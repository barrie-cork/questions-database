# GraphRAG Design Specification for Exam Question Extraction System

## Overview

This document outlines how Microsoft's GraphRAG architecture and the GraphRAG MCP server approach can be integrated into the exam question extraction system to create a powerful knowledge graph-based retrieval system for educational content.

## GraphRAG Architecture for Question Bank

### Core Concept

Instead of treating each question as an isolated entity, GraphRAG builds a comprehensive knowledge graph that captures:
- **Entities**: Questions, topics, concepts, exam papers, educational levels
- **Relationships**: Question similarity, topic hierarchy, prerequisite concepts, difficulty progression
- **Communities**: Clusters of related questions, topic groupings, difficulty bands

### System Architecture

```mermaid
graph TB
    subgraph "Data Ingestion Layer"
        PDF[PDF Files] --> OCR[Mistral OCR Service]
        OCR --> MD[Markdown Documents]
    end
    
    subgraph "Knowledge Graph Construction"
        MD --> EE[Entity Extraction<br/>Gemini 2.5 Flash]
        EE --> RE[Relationship Extraction]
        RE --> CD[Community Detection]
        CD --> CS[Community Summarization]
    end
    
    subgraph "Storage Layer"
        CS --> NEO4J[Neo4j Graph Database]
        CS --> QDRANT[Qdrant Vector Database]
        CS --> PG[PostgreSQL<br/>Structured Data]
    end
    
    subgraph "Retrieval Layer"
        QUERY[User Query] --> QP[Query Processor]
        QP --> GS[Graph Search]
        QP --> VS[Vector Search]
        QP --> HS[Hybrid Search]
        GS --> NEO4J
        VS --> QDRANT
        HS --> NEO4J
        HS --> QDRANT
    end
    
    subgraph "Generation Layer"
        GS --> CTX[Context Builder]
        VS --> CTX
        HS --> CTX
        CTX --> LLM[Gemini 2.5 Flash]
        LLM --> RESP[Response]
    end
```

## Implementation Design

### 1. Enhanced Entity Model

```python
# models/graph_entities.py
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class EntityType(str, Enum):
    QUESTION = "Question"
    TOPIC = "Topic"
    CONCEPT = "Concept"
    EXAM_PAPER = "ExamPaper"
    SKILL = "Skill"
    LEARNING_OBJECTIVE = "LearningObjective"

class RelationType(str, Enum):
    SIMILAR_TO = "SIMILAR_TO"
    BELONGS_TO = "BELONGS_TO"
    REQUIRES = "REQUIRES"
    TESTS = "TESTS"
    PART_OF = "PART_OF"
    DIFFICULTY_PROGRESSION = "DIFFICULTY_PROGRESSION"
    CONCEPT_DEPENDENCY = "CONCEPT_DEPENDENCY"

class GraphQuestion(BaseModel):
    """Enhanced question model for graph representation"""
    id: str
    question_number: str
    question_text: str
    marks: int
    topics: List[str]
    concepts: List[str] = Field(description="Key concepts tested")
    skills: List[str] = Field(description="Skills required")
    difficulty_score: float = Field(description="Computed difficulty 0-1")
    cognitive_level: str = Field(description="Bloom's taxonomy level")
    prerequisites: List[str] = Field(description="Prerequisite concepts")
    
class GraphEntity(BaseModel):
    """Generic entity for knowledge graph"""
    id: str
    type: EntityType
    name: str
    properties: dict
    embeddings: Optional[List[float]] = None
    
class GraphRelationship(BaseModel):
    """Relationship between entities"""
    source_id: str
    target_id: str
    type: RelationType
    properties: dict = {}
    weight: float = 1.0
```

### 2. Knowledge Graph Construction Service

```python
# services/graph_construction_service.py
from typing import List, Dict, Tuple
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from neo4j import AsyncGraphDatabase
import asyncio

class GraphConstructionService:
    """Builds knowledge graph from extracted questions"""
    
    def __init__(self, neo4j_uri: str, qdrant_client, gemini_service):
        self.driver = AsyncGraphDatabase.driver(neo4j_uri)
        self.qdrant = qdrant_client
        self.gemini = gemini_service
        
    async def build_knowledge_graph(self, questions: List[GraphQuestion]):
        """Main pipeline for graph construction"""
        
        # 1. Extract entities
        entities = await self._extract_entities(questions)
        
        # 2. Extract relationships
        relationships = await self._extract_relationships(entities)
        
        # 3. Detect communities
        communities = await self._detect_communities(entities, relationships)
        
        # 4. Generate community summaries
        summaries = await self._generate_summaries(communities)
        
        # 5. Store in databases
        await self._store_graph(entities, relationships, communities, summaries)
        
        return {
            "entities": len(entities),
            "relationships": len(relationships),
            "communities": len(communities)
        }
    
    async def _extract_entities(self, questions: List[GraphQuestion]) -> List[GraphEntity]:
        """Extract entities from questions using LLM"""
        entities = []
        
        # Questions as entities
        for q in questions:
            entities.append(GraphEntity(
                id=f"question_{q.id}",
                type=EntityType.QUESTION,
                name=f"Q{q.question_number}",
                properties={
                    "text": q.question_text,
                    "marks": q.marks,
                    "difficulty": q.difficulty_score
                }
            ))
        
        # Extract topics, concepts, skills using Gemini
        prompt = self._create_entity_extraction_prompt(questions)
        response = await self.gemini.extract_entities(prompt)
        
        # Parse and add extracted entities
        for entity_data in response.entities:
            entities.append(GraphEntity(
                id=entity_data["id"],
                type=EntityType(entity_data["type"]),
                name=entity_data["name"],
                properties=entity_data["properties"]
            ))
        
        return entities
    
    async def _extract_relationships(self, entities: List[GraphEntity]) -> List[GraphRelationship]:
        """Extract relationships between entities"""
        relationships = []
        
        # Question similarity based on embeddings
        question_entities = [e for e in entities if e.type == EntityType.QUESTION]
        embeddings = await self._generate_embeddings(question_entities)
        
        # Calculate similarities and create relationships
        similarity_matrix = cosine_similarity(embeddings)
        threshold = 0.8  # Similarity threshold
        
        for i in range(len(question_entities)):
            for j in range(i + 1, len(question_entities)):
                if similarity_matrix[i][j] > threshold:
                    relationships.append(GraphRelationship(
                        source_id=question_entities[i].id,
                        target_id=question_entities[j].id,
                        type=RelationType.SIMILAR_TO,
                        weight=float(similarity_matrix[i][j])
                    ))
        
        # Extract semantic relationships using LLM
        semantic_rels = await self._extract_semantic_relationships(entities)
        relationships.extend(semantic_rels)
        
        return relationships
    
    async def _detect_communities(
        self, 
        entities: List[GraphEntity], 
        relationships: List[GraphRelationship]
    ) -> Dict[str, List[str]]:
        """Detect communities using graph algorithms"""
        
        # Build NetworkX graph
        G = nx.Graph()
        
        for entity in entities:
            G.add_node(entity.id, **entity.properties)
        
        for rel in relationships:
            G.add_edge(rel.source_id, rel.target_id, weight=rel.weight)
        
        # Apply Louvain community detection
        communities = nx.community.louvain_communities(G, seed=42)
        
        # Convert to dict format
        community_dict = {}
        for i, community in enumerate(communities):
            community_dict[f"community_{i}"] = list(community)
        
        return community_dict
    
    async def _generate_summaries(self, communities: Dict[str, List[str]]) -> Dict[str, str]:
        """Generate natural language summaries for each community"""
        summaries = {}
        
        for community_id, entity_ids in communities.items():
            # Get entity details
            entities = await self._get_entities_by_ids(entity_ids)
            
            # Create summarization prompt
            prompt = f"""
            Summarize the following group of educational content entities:
            
            Entities: {entities}
            
            Provide a concise summary that captures:
            1. Main topics and concepts covered
            2. Difficulty level and progression
            3. Key relationships between items
            4. Educational objectives
            """
            
            summary = await self.gemini.generate_summary(prompt)
            summaries[community_id] = summary
        
        return summaries
    
    async def _store_graph(
        self,
        entities: List[GraphEntity],
        relationships: List[GraphRelationship],
        communities: Dict[str, List[str]],
        summaries: Dict[str, str]
    ):
        """Store graph data in Neo4j and vectors in Qdrant"""
        
        async with self.driver.session() as session:
            # Store entities
            for entity in entities:
                await session.run(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e.type = $type, e.name = $name, e.properties = $properties
                    """,
                    id=entity.id,
                    type=entity.type,
                    name=entity.name,
                    properties=entity.properties
                )
            
            # Store relationships
            for rel in relationships:
                await session.run(
                    """
                    MATCH (a:Entity {id: $source_id})
                    MATCH (b:Entity {id: $target_id})
                    MERGE (a)-[r:RELATES {type: $type}]->(b)
                    SET r.weight = $weight, r.properties = $properties
                    """,
                    source_id=rel.source_id,
                    target_id=rel.target_id,
                    type=rel.type,
                    weight=rel.weight,
                    properties=rel.properties
                )
            
            # Store communities
            for community_id, entity_ids in communities.items():
                await session.run(
                    """
                    CREATE (c:Community {id: $id, summary: $summary})
                    WITH c
                    UNWIND $entity_ids AS entity_id
                    MATCH (e:Entity {id: entity_id})
                    MERGE (e)-[:BELONGS_TO]->(c)
                    """,
                    id=community_id,
                    summary=summaries.get(community_id, ""),
                    entity_ids=entity_ids
                )
        
        # Store embeddings in Qdrant
        await self._store_embeddings_qdrant(entities)
```

### 3. GraphRAG Retrieval Service

```python
# services/graphrag_retrieval_service.py
from typing import List, Dict, Optional
import asyncio
from neo4j import AsyncGraphDatabase
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, Range

class GraphRAGRetrievalService:
    """Implements GraphRAG retrieval strategies"""
    
    def __init__(self, neo4j_uri: str, qdrant_config: dict):
        self.neo4j_driver = AsyncGraphDatabase.driver(neo4j_uri)
        self.qdrant = QdrantClient(**qdrant_config)
        
    async def search(
        self, 
        query: str, 
        search_type: str = "hybrid",
        filters: Optional[Dict] = None,
        limit: int = 10
    ) -> List[Dict]:
        """Main search interface supporting multiple strategies"""
        
        if search_type == "local":
            return await self._local_search(query, filters, limit)
        elif search_type == "global":
            return await self._global_search(query, filters, limit)
        elif search_type == "hybrid":
            return await self._hybrid_search(query, filters, limit)
        else:
            raise ValueError(f"Unknown search type: {search_type}")
    
    async def _local_search(self, query: str, filters: Dict, limit: int) -> List[Dict]:
        """Search for specific entities and their immediate context"""
        
        # 1. Vector search for similar questions
        query_embedding = await self._get_query_embedding(query)
        
        vector_results = self.qdrant.search(
            collection_name="questions",
            query_vector=query_embedding,
            limit=limit * 2,  # Get more for graph expansion
            query_filter=self._build_qdrant_filter(filters)
        )
        
        # 2. Expand context using graph relationships
        expanded_results = []
        async with self.neo4j_driver.session() as session:
            for result in vector_results[:limit]:
                entity_id = result.payload["entity_id"]
                
                # Get related entities
                graph_context = await session.run(
                    """
                    MATCH (e:Entity {id: $id})-[r:RELATES*1..2]-(related:Entity)
                    WHERE r.weight > 0.7
                    RETURN e, r, related
                    LIMIT 10
                    """,
                    id=entity_id
                )
                
                expanded_results.append({
                    "primary": result.payload,
                    "score": result.score,
                    "context": await graph_context.values()
                })
        
        return expanded_results
    
    async def _global_search(self, query: str, filters: Dict, limit: int) -> List[Dict]:
        """Search across community summaries for broad understanding"""
        
        # 1. Search community summaries
        query_embedding = await self._get_query_embedding(query)
        
        community_results = self.qdrant.search(
            collection_name="community_summaries",
            query_vector=query_embedding,
            limit=limit
        )
        
        # 2. Get all entities in relevant communities
        results = []
        async with self.neo4j_driver.session() as session:
            for community in community_results:
                community_id = community.payload["community_id"]
                
                entities = await session.run(
                    """
                    MATCH (c:Community {id: $id})<-[:BELONGS_TO]-(e:Entity)
                    RETURN e, c.summary as summary
                    """,
                    id=community_id
                )
                
                results.append({
                    "community": community_id,
                    "summary": community.payload["summary"],
                    "score": community.score,
                    "entities": await entities.values()
                })
        
        return results
    
    async def _hybrid_search(self, query: str, filters: Dict, limit: int) -> List[Dict]:
        """Combine local and global search strategies"""
        
        # Run both searches in parallel
        local_task = self._local_search(query, filters, limit // 2)
        global_task = self._global_search(query, filters, limit // 2)
        
        local_results, global_results = await asyncio.gather(local_task, global_task)
        
        # Merge and rank results
        merged_results = self._merge_results(local_results, global_results)
        
        return merged_results[:limit]
    
    async def get_learning_path(self, topic: str, level: str) -> List[Dict]:
        """Generate learning path using graph traversal"""
        
        async with self.neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH path = (start:Entity {type: 'Topic', name: $topic})-[:REQUIRES*]->(end:Entity)
                WHERE NOT (end)-[:REQUIRES]->()
                WITH nodes(path) as concepts
                UNWIND concepts as concept
                MATCH (concept)<-[:TESTS]-(q:Entity {type: 'Question'})
                WHERE q.properties.level = $level OR $level = 'all'
                RETURN concept, collect(q) as questions
                ORDER BY size(questions) DESC
                """,
                topic=topic,
                level=level
            )
            
            return await result.values()
    
    async def find_similar_questions(
        self, 
        question_id: str, 
        similarity_threshold: float = 0.8
    ) -> List[Dict]:
        """Find similar questions using graph relationships"""
        
        async with self.neo4j_driver.session() as session:
            result = await session.run(
                """
                MATCH (q:Entity {id: $id, type: 'Question'})-[r:SIMILAR_TO]-(similar:Entity)
                WHERE r.weight >= $threshold
                RETURN similar, r.weight as similarity
                ORDER BY similarity DESC
                LIMIT 20
                """,
                id=question_id,
                threshold=similarity_threshold
            )
            
            return await result.values()
```

### 4. Integration with MCP Server

```python
# mcp_server/graphrag_mcp_server.py
from typing import List, Dict, Any
import asyncio
from mcp.server import Server, Request
from mcp.types import Tool, TextContent
import json

class GraphRAGMCPServer:
    """MCP server implementation for GraphRAG question bank"""
    
    def __init__(self, retrieval_service: GraphRAGRetrievalService):
        self.retrieval = retrieval_service
        self.server = Server("graphrag-questions")
        
        # Register tools
        self.server.add_tool(self._search_questions_tool())
        self.server.add_tool(self._find_similar_tool())
        self.server.add_tool(self._get_learning_path_tool())
        self.server.add_tool(self._analyze_topic_coverage_tool())
        
    def _search_questions_tool(self) -> Tool:
        return Tool(
            name="search_questions",
            description="Search for exam questions using GraphRAG",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "search_type": {
                        "type": "string",
                        "enum": ["local", "global", "hybrid"],
                        "default": "hybrid"
                    },
                    "filters": {
                        "type": "object",
                        "properties": {
                            "year": {"type": "string"},
                            "level": {"type": "string"},
                            "topic": {"type": "string"},
                            "min_marks": {"type": "integer"},
                            "max_marks": {"type": "integer"}
                        }
                    },
                    "limit": {"type": "integer", "default": 10}
                },
                "required": ["query"]
            },
            handler=self._handle_search
        )
    
    async def _handle_search(self, request: Request) -> List[TextContent]:
        """Handle search requests"""
        params = request.params
        
        results = await self.retrieval.search(
            query=params["query"],
            search_type=params.get("search_type", "hybrid"),
            filters=params.get("filters", {}),
            limit=params.get("limit", 10)
        )
        
        # Format results for LLM consumption
        formatted = self._format_search_results(results)
        
        return [TextContent(
            type="text",
            text=json.dumps(formatted, indent=2)
        )]
    
    def _find_similar_tool(self) -> Tool:
        return Tool(
            name="find_similar_questions",
            description="Find questions similar to a given question",
            inputSchema={
                "type": "object",
                "properties": {
                    "question_id": {"type": "string"},
                    "similarity_threshold": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "default": 0.8
                    }
                },
                "required": ["question_id"]
            },
            handler=self._handle_find_similar
        )
    
    def _get_learning_path_tool(self) -> Tool:
        return Tool(
            name="get_learning_path",
            description="Generate a learning path for a topic",
            inputSchema={
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "level": {"type": "string", "default": "all"}
                },
                "required": ["topic"]
            },
            handler=self._handle_learning_path
        )
```

### 5. Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  # PostgreSQL with pgvector
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: question_bank
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Neo4j Graph Database
  neo4j:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: neo4j/${NEO4J_PASSWORD}
      NEO4J_PLUGINS: '["graph-data-science"]'
      NEO4J_dbms_memory_pagecache_size: 1G
      NEO4J_dbms_memory_heap_max__size: 1G
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    healthcheck:
      test: ["CMD", "neo4j", "status"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Qdrant Vector Database
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"  # gRPC
    volumes:
      - qdrant_data:/qdrant/storage
    environment:
      QDRANT__SERVICE__GRPC_PORT: 6334

  # Main Application
  app:
    build: .
    environment:
      DATABASE_URL: postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/question_bank
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: ${NEO4J_PASSWORD}
      QDRANT_HOST: qdrant
      QDRANT_PORT: 6333
      MISTRAL_API_KEY: ${MISTRAL_API_KEY}
      GOOGLE_API_KEY: ${GOOGLE_API_KEY}
    volumes:
      - ./app:/app
      - ./uploads:/app/uploads
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      neo4j:
        condition: service_healthy
      qdrant:
        condition: service_started
    command: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

  # GraphRAG MCP Server
  mcp-server:
    build:
      context: ./mcp_server
      dockerfile: Dockerfile
    environment:
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: ${NEO4J_PASSWORD}
      QDRANT_HOST: qdrant
      QDRANT_PORT: 6333
    ports:
      - "3000:3000"
    depends_on:
      - neo4j
      - qdrant
    command: python -m graphrag_mcp_server

volumes:
  postgres_data:
  neo4j_data:
  neo4j_logs:
  qdrant_data:
```

## Key Benefits of GraphRAG Approach

### 1. **Enhanced Search Capabilities**
- **Semantic Search**: Find questions by meaning, not just keywords
- **Relationship-Aware**: Discovers related questions through graph traversal
- **Multi-Level Queries**: From specific questions to broad topic overviews

### 2. **Educational Intelligence**
- **Learning Paths**: Automatically generate prerequisite chains
- **Difficulty Progression**: Understand question difficulty relationships
- **Concept Mapping**: Visualize topic interconnections

### 3. **Improved Question Quality**
- **Duplicate Detection**: Graph similarity identifies near-duplicates
- **Coverage Analysis**: Identify gaps in topic coverage
- **Consistency Checking**: Ensure marking schemes align

### 4. **Scalability**
- **Efficient Retrieval**: Graph traversal + vector search
- **Community Summaries**: Handle large datasets efficiently
- **Incremental Updates**: Add new questions without full reprocessing

### 5. **Advanced Analytics**
- **Topic Distribution**: Understand exam paper composition
- **Difficulty Analysis**: Track difficulty trends over years
- **Relationship Insights**: Discover hidden curriculum connections

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- Set up Neo4j, Qdrant, and PostgreSQL
- Implement basic entity and relationship extraction
- Create initial graph construction pipeline

### Phase 2: Graph Construction (Weeks 3-4)
- Implement community detection algorithms
- Generate community summaries
- Build graph storage layer

### Phase 3: Retrieval System (Weeks 5-6)
- Implement local, global, and hybrid search
- Create MCP server integration
- Build API endpoints

### Phase 4: Intelligence Layer (Weeks 7-8)
- Add learning path generation
- Implement similarity analysis
- Create analytics dashboards

### Phase 5: Optimization (Weeks 9-10)
- Performance tuning
- Graph algorithm optimization
- Production deployment

## Cost Considerations

### Additional Infrastructure Costs
- **Neo4j Community**: Free (or Enterprise for advanced features)
- **Qdrant**: Free for self-hosted
- **Additional Storage**: ~2-3x original data size for graph
- **Processing**: One-time graph construction cost

### API Usage Optimization
- **Batch Processing**: Process multiple questions together
- **Caching**: Store community summaries
- **Incremental Updates**: Only process new questions

## Conclusion

Integrating GraphRAG into the exam question extraction system transforms it from a simple storage solution to an intelligent knowledge management platform. The graph-based approach enables sophisticated retrieval, relationship discovery, and educational insights that would be impossible with traditional RAG systems.

The combination of Neo4j for graph storage, Qdrant for vector search, and PostgreSQL for structured data provides a robust, scalable foundation for building an advanced question bank that can serve as the basis for intelligent tutoring systems, adaptive testing, and curriculum analysis.