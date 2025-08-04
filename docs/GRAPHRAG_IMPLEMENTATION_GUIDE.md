# GraphRAG Implementation Guide for Exam Question System

## Quick Start Implementation

This guide provides a practical, step-by-step approach to implementing GraphRAG features in the existing exam question extraction system.

## Core GraphRAG Components to Implement

### 1. Entity and Relationship Extraction

```python
# services/entity_extraction_service.py
from google import genai
from google.genai import types
from typing import List, Dict, Tuple
import json
from pydantic import BaseModel, Field

class EntityExtractionService:
    """Extract entities and relationships from questions using Gemini"""
    
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-2.5-flash"
        
    async def extract_entities_and_relationships(
        self, 
        questions: List[Dict]
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relationships from a batch of questions"""
        
        # Define the schema for structured extraction
        class Entity(BaseModel):
            id: str = Field(description="Unique identifier")
            type: str = Field(description="Entity type: Topic, Concept, Skill")
            name: str = Field(description="Entity name")
            properties: Dict = Field(default_factory=dict)
            
        class Relationship(BaseModel):
            source_id: str = Field(description="Source entity ID")
            target_id: str = Field(description="Target entity ID")
            type: str = Field(description="Relationship type")
            properties: Dict = Field(default_factory=dict)
            
        class ExtractionResult(BaseModel):
            entities: List[Entity]
            relationships: List[Relationship]
        
        # Create extraction prompt
        prompt = f"""
        Analyze these exam questions and extract:
        1. Entities: Topics, Concepts, Skills, Learning Objectives
        2. Relationships: REQUIRES (prerequisites), SIMILAR_TO, TESTS, PART_OF
        
        Questions:
        {json.dumps(questions, indent=2)}
        
        Extract entities and relationships following the schema.
        Focus on educational relationships and concept dependencies.
        """
        
        # Use Gemini with structured output
        response = await self.client.models.generate_content_async(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=ExtractionResult.model_json_schema(),
                temperature=0.1,
                max_output_tokens=8192
            )
        )
        
        result = ExtractionResult.model_validate_json(response.text)
        
        return (
            [e.model_dump() for e in result.entities],
            [r.model_dump() for r in result.relationships]
        )
```

### 2. Simple Graph Storage with PostgreSQL

```python
# services/graph_storage_service.py
import asyncpg
from typing import List, Dict, Optional
import json

class GraphStorageService:
    """Store graph data in PostgreSQL (simpler than Neo4j for MVP)"""
    
    def __init__(self, db_url: str):
        self.db_url = db_url
        
    async def init_schema(self):
        """Create graph tables in PostgreSQL"""
        conn = await asyncpg.connect(self.db_url)
        
        await conn.execute("""
            -- Entity table
            CREATE TABLE IF NOT EXISTS graph_entities (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                properties JSONB DEFAULT '{}',
                embedding vector(768),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Relationship table
            CREATE TABLE IF NOT EXISTS graph_relationships (
                id SERIAL PRIMARY KEY,
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                type TEXT NOT NULL,
                weight FLOAT DEFAULT 1.0,
                properties JSONB DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_id) REFERENCES graph_entities(id),
                FOREIGN KEY (target_id) REFERENCES graph_entities(id),
                UNIQUE(source_id, target_id, type)
            );
            
            -- Community table for clustering
            CREATE TABLE IF NOT EXISTS graph_communities (
                id TEXT PRIMARY KEY,
                name TEXT,
                summary TEXT,
                member_ids TEXT[],
                centroid_embedding vector(768),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            -- Indexes for performance
            CREATE INDEX IF NOT EXISTS idx_entities_type ON graph_entities(type);
            CREATE INDEX IF NOT EXISTS idx_entities_embedding ON graph_entities 
                USING hnsw (embedding vector_cosine_ops);
            CREATE INDEX IF NOT EXISTS idx_relationships_source ON graph_relationships(source_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_target ON graph_relationships(target_id);
            CREATE INDEX IF NOT EXISTS idx_relationships_type ON graph_relationships(type);
        """)
        
        await conn.close()
    
    async def store_entities(self, entities: List[Dict]):
        """Batch store entities"""
        conn = await asyncpg.connect(self.db_url)
        
        try:
            await conn.executemany(
                """
                INSERT INTO graph_entities (id, type, name, properties)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (id) DO UPDATE
                SET name = EXCLUDED.name,
                    properties = EXCLUDED.properties
                """,
                [(e['id'], e['type'], e['name'], json.dumps(e.get('properties', {}))) 
                 for e in entities]
            )
        finally:
            await conn.close()
    
    async def store_relationships(self, relationships: List[Dict]):
        """Batch store relationships"""
        conn = await asyncpg.connect(self.db_url)
        
        try:
            await conn.executemany(
                """
                INSERT INTO graph_relationships (source_id, target_id, type, weight, properties)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (source_id, target_id, type) DO UPDATE
                SET weight = EXCLUDED.weight,
                    properties = EXCLUDED.properties
                """,
                [(r['source_id'], r['target_id'], r['type'], 
                  r.get('weight', 1.0), json.dumps(r.get('properties', {})))
                 for r in relationships]
            )
        finally:
            await conn.close()
```

### 3. Community Detection and Summarization

```python
# services/community_service.py
import numpy as np
from sklearn.cluster import DBSCAN
from typing import List, Dict
import asyncpg

class CommunityDetectionService:
    """Detect and summarize question communities"""
    
    def __init__(self, embedding_service, gemini_service, db_url: str):
        self.embedding_service = embedding_service
        self.gemini = gemini_service
        self.db_url = db_url
        
    async def detect_communities(self, min_samples: int = 3, eps: float = 0.3):
        """Detect communities using DBSCAN clustering on embeddings"""
        
        conn = await asyncpg.connect(self.db_url)
        
        # Get all question embeddings
        rows = await conn.fetch("""
            SELECT q.id, q.question_text, q.topics, qe.embedding
            FROM questions q
            JOIN question_embeddings qe ON q.id = qe.question_id
            WHERE qe.embedding IS NOT NULL
        """)
        
        if not rows:
            return []
        
        # Extract embeddings and metadata
        ids = [row['id'] for row in rows]
        embeddings = np.array([row['embedding'] for row in rows])
        metadata = [{
            'id': row['id'],
            'text': row['question_text'],
            'topics': row['topics']
        } for row in rows]
        
        # Perform clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)
        
        # Group by cluster
        communities = {}
        for idx, label in enumerate(labels):
            if label == -1:  # Noise point
                continue
            
            if label not in communities:
                communities[label] = []
            communities[label].append(metadata[idx])
        
        # Generate summaries for each community
        community_data = []
        for label, members in communities.items():
            summary = await self._generate_community_summary(members)
            centroid = self._calculate_centroid(
                [embeddings[ids.index(m['id'])] for m in members]
            )
            
            community_data.append({
                'id': f'community_{label}',
                'name': f'Community {label}',
                'summary': summary,
                'member_ids': [m['id'] for m in members],
                'centroid_embedding': centroid.tolist()
            })
        
        # Store communities
        await self._store_communities(community_data)
        await conn.close()
        
        return community_data
    
    async def _generate_community_summary(self, members: List[Dict]) -> str:
        """Generate natural language summary of a community"""
        
        prompt = f"""
        Summarize this group of exam questions:
        
        Questions:
        {json.dumps(members, indent=2)}
        
        Provide a concise summary (2-3 sentences) that captures:
        1. Common topics and themes
        2. Typical difficulty level
        3. Key concepts tested
        """
        
        response = await self.gemini.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=200
            )
        )
        
        return response.text
    
    def _calculate_centroid(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Calculate centroid of embeddings"""
        return np.mean(embeddings, axis=0)
```

### 4. GraphRAG Search Implementation

```python
# services/graphrag_search_service.py
import asyncpg
from typing import List, Dict, Optional
import numpy as np

class GraphRAGSearchService:
    """Implement GraphRAG search strategies"""
    
    def __init__(self, embedding_service, db_url: str):
        self.embedding_service = embedding_service
        self.db_url = db_url
        
    async def search(
        self,
        query: str,
        strategy: str = "hybrid",
        limit: int = 10,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """Main search interface"""
        
        if strategy == "local":
            return await self._local_search(query, limit, filters)
        elif strategy == "global":
            return await self._global_search(query, limit, filters)
        else:  # hybrid
            local = await self._local_search(query, limit // 2, filters)
            global_ = await self._global_search(query, limit // 2, filters)
            return self._merge_results(local, global_, limit)
    
    async def _local_search(
        self, 
        query: str, 
        limit: int, 
        filters: Optional[Dict]
    ) -> List[Dict]:
        """Search specific questions and expand through relationships"""
        
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embedding(
            query, 
            task_type="RETRIEVAL_QUERY"
        )
        
        conn = await asyncpg.connect(self.db_url)
        
        # Vector similarity search
        base_query = """
            WITH similar_questions AS (
                SELECT 
                    q.id,
                    q.question_text,
                    q.marks,
                    q.topics,
                    q.year,
                    q.level,
                    1 - (qe.embedding <=> $1::vector) as similarity
                FROM questions q
                JOIN question_embeddings qe ON q.id = qe.question_id
                WHERE 1 - (qe.embedding <=> $1::vector) > 0.7
        """
        
        # Add filters
        filter_conditions = []
        params = [query_embedding]
        param_count = 1
        
        if filters:
            if 'year' in filters:
                param_count += 1
                filter_conditions.append(f"q.year = ${param_count}")
                params.append(filters['year'])
            
            if 'level' in filters:
                param_count += 1
                filter_conditions.append(f"q.level = ${param_count}")
                params.append(filters['level'])
            
            if 'min_marks' in filters:
                param_count += 1
                filter_conditions.append(f"q.marks >= ${param_count}")
                params.append(filters['min_marks'])
        
        if filter_conditions:
            base_query += " AND " + " AND ".join(filter_conditions)
        
        base_query += f"""
                ORDER BY similarity DESC
                LIMIT {limit * 2}
            )
            SELECT * FROM similar_questions
        """
        
        results = await conn.fetch(base_query, *params)
        
        # Expand through graph relationships
        expanded_results = []
        for row in results[:limit]:
            question_id = row['id']
            
            # Get related entities
            related = await conn.fetch("""
                SELECT 
                    e2.*, 
                    r.type as relationship_type,
                    r.weight
                FROM graph_relationships r
                JOIN graph_entities e1 ON r.source_id = e1.id
                JOIN graph_entities e2 ON r.target_id = e2.id
                WHERE e1.id = $1 OR r.target_id = $1
                ORDER BY r.weight DESC
                LIMIT 5
            """, f"question_{question_id}")
            
            expanded_results.append({
                'question': dict(row),
                'similarity': row['similarity'],
                'related_entities': [dict(r) for r in related]
            })
        
        await conn.close()
        return expanded_results
    
    async def _global_search(
        self, 
        query: str, 
        limit: int, 
        filters: Optional[Dict]
    ) -> List[Dict]:
        """Search across community summaries"""
        
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embedding(
            query, 
            task_type="RETRIEVAL_QUERY"
        )
        
        conn = await asyncpg.connect(self.db_url)
        
        # Search communities
        communities = await conn.fetch("""
            SELECT 
                id,
                name,
                summary,
                member_ids,
                1 - (centroid_embedding <=> $1::vector) as similarity
            FROM graph_communities
            WHERE centroid_embedding IS NOT NULL
            ORDER BY similarity DESC
            LIMIT $2
        """, query_embedding, limit)
        
        # Get questions from top communities
        results = []
        for community in communities:
            questions = await conn.fetch("""
                SELECT * FROM questions
                WHERE id = ANY($1::bigint[])
            """, community['member_ids'])
            
            results.append({
                'community': dict(community),
                'questions': [dict(q) for q in questions],
                'similarity': community['similarity']
            })
        
        await conn.close()
        return results
```

### 5. FastAPI Integration

```python
# api/routes/graphrag.py
from fastapi import APIRouter, Query, HTTPException
from typing import List, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/api/graphrag", tags=["GraphRAG"])

class GraphSearchRequest(BaseModel):
    query: str
    strategy: str = "hybrid"
    limit: int = 10
    filters: Optional[dict] = None

class LearningPathRequest(BaseModel):
    topic: str
    level: str
    max_questions: int = 20

@router.post("/search")
async def search_questions(request: GraphSearchRequest):
    """Search questions using GraphRAG"""
    try:
        results = await graph_search_service.search(
            query=request.query,
            strategy=request.strategy,
            limit=request.limit,
            filters=request.filters
        )
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/extract-graph")
async def extract_graph(question_ids: List[int]):
    """Extract entities and relationships from questions"""
    try:
        # Get questions
        questions = await get_questions_by_ids(question_ids)
        
        # Extract entities and relationships
        entities, relationships = await entity_extraction_service.extract_entities_and_relationships(questions)
        
        # Store in graph
        await graph_storage_service.store_entities(entities)
        await graph_storage_service.store_relationships(relationships)
        
        return {
            "entities_extracted": len(entities),
            "relationships_extracted": len(relationships)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect-communities")
async def detect_communities():
    """Run community detection on questions"""
    try:
        communities = await community_service.detect_communities()
        return {
            "communities_found": len(communities),
            "communities": communities
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/similar/{question_id}")
async def find_similar_questions(
    question_id: int,
    limit: int = Query(10, ge=1, le=50)
):
    """Find similar questions using graph relationships"""
    try:
        # Use graph relationships and embeddings
        similar = await graph_storage_service.find_similar_questions(
            question_id=question_id,
            limit=limit
        )
        return {"similar_questions": similar}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/learning-path")
async def generate_learning_path(request: LearningPathRequest):
    """Generate a learning path for a topic"""
    try:
        # Find prerequisite chain
        path = await graph_storage_service.get_learning_path(
            topic=request.topic,
            level=request.level,
            max_questions=request.max_questions
        )
        return {"learning_path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 6. Enhanced PDF Processing Pipeline

```python
# services/enhanced_pdf_processor.py
class EnhancedPDFProcessor(PDFQuestionProcessor):
    """PDF processor with GraphRAG capabilities"""
    
    def __init__(
        self,
        ocr_service,
        llm_service,
        embedding_service,
        entity_extraction_service,
        graph_storage_service,
        community_service
    ):
        super().__init__(ocr_service, llm_service, embedding_service)
        self.entity_extraction = entity_extraction_service
        self.graph_storage = graph_storage_service
        self.community_service = community_service
        
    async def process_pdf_with_graph(self, pdf_path: str, pdf_filename: str):
        """Process PDF and build graph"""
        
        # Standard processing
        result = await self.process_pdf(pdf_path, pdf_filename)
        
        # Extract graph data
        questions = result['questions']
        question_data = [q['question'] for q in questions]
        
        # Extract entities and relationships
        entities, relationships = await self.entity_extraction.extract_entities_and_relationships(
            question_data
        )
        
        # Add question entities
        for q in questions:
            entities.append({
                'id': f"question_{q['question']['id']}",
                'type': 'Question',
                'name': f"Q{q['question']['question_number']}",
                'properties': {
                    'text': q['question']['question_text'],
                    'marks': q['question']['marks'],
                    'year': q['question']['year'],
                    'level': q['question']['level']
                }
            })
        
        # Store graph data
        await self.graph_storage.store_entities(entities)
        await self.graph_storage.store_relationships(relationships)
        
        # Store embeddings
        for q in questions:
            if q['embedding']:
                await self.graph_storage.update_entity_embedding(
                    f"question_{q['question']['id']}",
                    q['embedding']
                )
        
        # Update result
        result['graph_data'] = {
            'entities': len(entities),
            'relationships': len(relationships)
        }
        
        return result
    
    async def post_process_graph(self):
        """Run post-processing tasks"""
        
        # Detect communities
        communities = await self.community_service.detect_communities()
        
        # Calculate graph statistics
        stats = await self.graph_storage.get_graph_statistics()
        
        return {
            'communities': communities,
            'statistics': stats
        }
```

### 7. Frontend Integration

```javascript
// static/js/graphrag.js

class GraphRAGSearch {
    constructor() {
        this.searchStrategy = 'hybrid';
        this.searchResults = [];
    }
    
    async search(query, filters = {}) {
        const response = await fetch('/api/graphrag/search', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                query: query,
                strategy: this.searchStrategy,
                limit: 20,
                filters: filters
            })
        });
        
        const data = await response.json();
        this.displayResults(data.results);
    }
    
    displayResults(results) {
        const container = document.getElementById('searchResults');
        container.innerHTML = '';
        
        results.forEach(result => {
            const div = document.createElement('div');
            div.className = 'search-result';
            
            if (this.searchStrategy === 'local') {
                // Display question with related entities
                div.innerHTML = `
                    <div class="question">
                        <h4>Q${result.question.question_number} (${result.similarity.toFixed(2)})</h4>
                        <p>${result.question.question_text}</p>
                        <div class="metadata">
                            <span>Marks: ${result.question.marks}</span>
                            <span>Year: ${result.question.year}</span>
                            <span>Topics: ${result.question.topics.join(', ')}</span>
                        </div>
                        <div class="related-entities">
                            <h5>Related Concepts:</h5>
                            ${result.related_entities.map(e => 
                                `<span class="entity">${e.name} (${e.type})</span>`
                            ).join(' ')}
                        </div>
                    </div>
                `;
            } else {
                // Display community with questions
                div.innerHTML = `
                    <div class="community">
                        <h4>${result.community.name} (${result.similarity.toFixed(2)})</h4>
                        <p>${result.community.summary}</p>
                        <div class="community-questions">
                            <h5>Questions in this group:</h5>
                            ${result.questions.slice(0, 5).map(q => 
                                `<div class="mini-question">
                                    Q${q.question_number}: ${q.question_text.substring(0, 100)}...
                                </div>`
                            ).join('')}
                            ${result.questions.length > 5 ? 
                                `<p>...and ${result.questions.length - 5} more</p>` : ''}
                        </div>
                    </div>
                `;
            }
            
            container.appendChild(div);
        });
    }
    
    async findSimilar(questionId) {
        const response = await fetch(`/api/graphrag/similar/${questionId}`);
        const data = await response.json();
        
        this.showSimilarQuestions(data.similar_questions);
    }
    
    async visualizeGraph(questionId) {
        // Use vis.js or D3.js to visualize the local graph
        const response = await fetch(`/api/graphrag/subgraph/${questionId}`);
        const graphData = await response.json();
        
        this.renderGraph(graphData);
    }
}

// Initialize on page load
const graphSearch = new GraphRAGSearch();

// Add search UI
document.getElementById('advancedSearchBtn').addEventListener('click', () => {
    const query = document.getElementById('searchQuery').value;
    const filters = {
        year: document.getElementById('filterYear').value,
        level: document.getElementById('filterLevel').value
    };
    
    graphSearch.search(query, filters);
});
```

## Deployment Steps

### 1. Update Docker Compose

Add this to your existing `docker-compose.yml`:

```yaml
  # Initialize GraphRAG tables
  init-graphrag:
    build: .
    command: python -m scripts.init_graphrag
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/question_bank
```

### 2. Initialize Script

```python
# scripts/init_graphrag.py
import asyncio
from services.graph_storage_service import GraphStorageService

async def main():
    service = GraphStorageService(os.getenv('DATABASE_URL'))
    await service.init_schema()
    print("GraphRAG schema initialized successfully!")

if __name__ == "__main__":
    asyncio.run(main())
```

### 3. Run Initialization

```bash
docker-compose run --rm init-graphrag
```

## Benefits Over Standard Approach

1. **Better Search Results**: GraphRAG finds conceptually related questions, not just keyword matches
2. **Educational Insights**: Automatically discovers prerequisite chains and learning paths
3. **Quality Control**: Community detection identifies duplicate or similar questions
4. **Scalability**: Graph structure enables efficient traversal of large question banks
5. **Flexibility**: Multiple search strategies for different use cases

## Next Steps

1. **Implement Basic GraphRAG**: Start with entity extraction and simple graph storage
2. **Add Community Detection**: Use clustering to group similar questions
3. **Build Search Interface**: Implement local and global search strategies
4. **Create Visualizations**: Add graph visualization for exploring relationships
5. **Optimize Performance**: Fine-tune embeddings and graph algorithms

This implementation provides a practical path to adding GraphRAG capabilities to the exam question extraction system without requiring a complete rewrite.