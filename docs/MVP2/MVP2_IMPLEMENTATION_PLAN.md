# MVP 2 Implementation Plan: GraphRAG Integration

## Executive Summary

MVP 2 transforms the exam question extraction system into an intelligent knowledge graph platform by integrating GraphRAG capabilities. This enables semantic search, relationship discovery, and educational intelligence features while maintaining the core extraction functionality from MVP 1.

## MVP 2 Goals

1. **Enhanced Search**: Move beyond keyword matching to semantic and relationship-aware search
2. **Educational Intelligence**: Discover prerequisite chains, difficulty progressions, and topic relationships
3. **Quality Improvements**: Detect similar/duplicate questions and ensure comprehensive topic coverage
4. **Scalable Architecture**: Build foundation for future AI-powered educational features

## Technical Architecture

### Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        MVP 1 (Existing)                         │
├─────────────────────────────────────────────────────────────────┤
│  • Mistral OCR Service     • Gemini LLM Service               │
│  • Embedding Service       • PostgreSQL + pgvector             │
│  • FastAPI + Tabulator UI  • Docker Infrastructure            │
└─────────────────────────────────────────────────────────────────┘
                                  ↓
┌─────────────────────────────────────────────────────────────────┐
│                    MVP 2 (GraphRAG Layer)                       │
├─────────────────────────────────────────────────────────────────┤
│  • Entity Extraction       • Graph Storage (PostgreSQL)        │
│  • Community Detection     • GraphRAG Search                   │
│  • Learning Path Gen       • Enhanced API Endpoints            │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Foundation (Week 1)

#### 1.1 Enhanced Data Models
- **File**: `models/graph_entities.py`
- **Status**: ✅ Created
- **Entities**: Question, Topic, Concept, Skill, LearningObjective, BloomTaxonomy, DifficultyLevel
- **Relationships**: SIMILAR_TO, REQUIRES, TESTS, BELONGS_TO, PEDAGOGICAL_SEQUENCE

#### 1.2 Database Schema Extensions
- **File**: `database/schema_v2.sql`
- **Tables to Add**:
  ```sql
  -- Graph entities table
  CREATE TABLE graph_entities (
      id TEXT PRIMARY KEY,
      type TEXT NOT NULL,
      name TEXT NOT NULL,
      description TEXT,
      properties JSONB DEFAULT '{}',
      embedding vector(768),
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
      updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );

  -- Graph relationships table
  CREATE TABLE graph_relationships (
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

  -- Communities table
  CREATE TABLE graph_communities (
      id TEXT PRIMARY KEY,
      name TEXT,
      summary TEXT,
      member_ids TEXT[],
      centroid_embedding vector(768),
      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );

  -- Create indexes
  CREATE INDEX idx_entities_type ON graph_entities(type);
  CREATE INDEX idx_entities_embedding ON graph_entities USING hnsw (embedding vector_cosine_ops);
  CREATE INDEX idx_relationships_source ON graph_relationships(source_id);
  CREATE INDEX idx_relationships_target ON graph_relationships(target_id);
  CREATE INDEX idx_communities_embedding ON graph_communities USING hnsw (centroid_embedding vector_cosine_ops);
  ```

### Phase 2: Entity Extraction (Week 2)

#### 2.1 Entity Extraction Service
- **File**: `services/entity_extraction_service.py`
- **Key Features**:
  - LLM-based entity extraction using Gemini 2.5 Flash
  - Structured output with Pydantic schemas
  - Hybrid approach: LLM + rule-based enhancements
  - Bloom's taxonomy detection
  - Difficulty calculation

#### 2.2 Bloom's Taxonomy Rules
- **File**: `services/rules/bloom_taxonomy.py`
- **Implementation**:
  ```python
  BLOOM_KEYWORDS = {
      "Remember": ["define", "list", "identify", "name", "recall"],
      "Understand": ["explain", "describe", "summarize", "interpret"],
      "Apply": ["solve", "calculate", "demonstrate", "implement"],
      "Analyze": ["analyze", "compare", "contrast", "examine"],
      "Evaluate": ["evaluate", "judge", "critique", "justify"],
      "Create": ["create", "design", "develop", "construct"]
  }
  ```

### Phase 3: Graph Storage (Week 3)

#### 3.1 Graph Storage Service
- **File**: `services/graph_storage_service.py`
- **Features**:
  - PostgreSQL-based graph storage (simpler than Neo4j for MVP)
  - Batch entity/relationship insertion
  - Graph traversal queries
  - Incremental update support

#### 3.2 Migration Script
- **File**: `scripts/migrate_to_graphrag.py`
- **Purpose**: Migrate existing questions to graph structure
- **Process**:
  1. Load existing questions from database
  2. Extract entities and relationships
  3. Store in graph tables
  4. Generate initial communities

### Phase 4: Community Detection (Week 4)

#### 4.1 Community Detection Service
- **File**: `services/community_service.py`
- **Algorithm**: DBSCAN clustering on embeddings
- **Features**:
  - Automatic community discovery
  - Community summarization using Gemini
  - Centroid calculation for efficient search

#### 4.2 Scheduled Jobs
- **File**: `services/scheduled_jobs.py`
- **Jobs**:
  - Daily community recomputation
  - Weekly graph statistics update
  - Monthly quality analysis

### Phase 5: GraphRAG Search (Week 5)

#### 5.1 Search Service Implementation
- **File**: `services/graphrag_search_service.py`
- **Search Strategies**:
  1. **Local Search**: Find specific entities + expand context
  2. **Global Search**: Search community summaries
  3. **Hybrid Search**: Combine both approaches

#### 5.2 Query Processing
- **Features**:
  - Query embedding generation
  - Multi-hop graph traversal
  - Result ranking and merging
  - Caching for performance

### Phase 6: API Integration (Week 6)

#### 6.1 New API Endpoints
- **File**: `api/routes/graphrag.py`
- **Endpoints**:
  ```python
  POST   /api/graphrag/search          # Semantic search
  POST   /api/graphrag/extract-graph   # Extract entities from questions
  POST   /api/graphrag/detect-communities  # Run community detection
  GET    /api/graphrag/similar/{id}    # Find similar questions
  POST   /api/graphrag/learning-path   # Generate learning path
  GET    /api/graphrag/graph-stats     # Graph statistics
  GET    /api/graphrag/visualize/{id}  # Get subgraph for visualization
  ```

#### 6.2 Enhanced UI Components
- **File**: `static/js/graphrag-ui.js`
- **Features**:
  - Advanced search interface
  - Similar question display
  - Topic relationship visualization
  - Learning path viewer

### Phase 7: Pipeline Integration (Week 7)

#### 7.1 Enhanced PDF Processor
- **File**: `services/enhanced_pdf_processor.py`
- **Integration Points**:
  1. After question extraction → Entity extraction
  2. After embedding generation → Graph construction
  3. After saving → Community detection

#### 7.2 Batch Processing
- **File**: `services/batch_processor.py`
- **Features**:
  - Process multiple PDFs with graph building
  - Progress tracking
  - Error recovery

### Phase 8: Testing & Optimization (Week 8)

#### 8.1 Performance Testing
- **Targets**:
  - Search response: <200ms
  - Entity extraction: <5s per question batch
  - Community detection: <30s for 1000 questions

#### 8.2 Test Suite
- **Files**:
  - `tests/test_entity_extraction.py`
  - `tests/test_graph_storage.py`
  - `tests/test_graphrag_search.py`
  - `tests/test_integration.py`

## Deployment Plan

### Docker Updates
```yaml
# docker-compose.yml additions
services:
  # Existing services...
  
  # GraphRAG initialization
  graphrag-init:
    build: .
    command: python scripts/init_graphrag.py
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/question_bank

  # GraphRAG migration
  graphrag-migrate:
    build: .
    command: python scripts/migrate_to_graphrag.py
    depends_on:
      postgres:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/question_bank
```

### Environment Variables
```bash
# .env additions
# GraphRAG Configuration
GRAPHRAG_ENABLED=true
GRAPHRAG_COMMUNITY_MIN_SIZE=3
GRAPHRAG_SIMILARITY_THRESHOLD=0.8
GRAPHRAG_CACHE_TTL=3600
```

## Success Metrics

### Technical Metrics
- **Search Relevance**: >90% user satisfaction vs keyword search
- **Entity Extraction**: >95% accuracy for main entities
- **Performance**: <200ms search response time
- **Scalability**: Handle 100K+ questions efficiently

### Business Metrics
- **Duplicate Detection**: Identify >80% of similar questions
- **Learning Paths**: Generate valid paths for >90% of topics
- **Topic Coverage**: Identify gaps in question banks
- **User Engagement**: 2x increase in search usage

## Risk Mitigation

### Technical Risks
1. **Performance Degradation**
   - Mitigation: Implement caching, optimize queries
   
2. **Entity Extraction Errors**
   - Mitigation: Manual review UI, confidence scores

3. **Graph Complexity**
   - Mitigation: Start simple, incremental improvements

### Operational Risks
1. **API Cost Increase**
   - Mitigation: Batch processing, caching
   
2. **Migration Failures**
   - Mitigation: Incremental migration, rollback plan

## Future Enhancements (Post-MVP 2)

1. **Neo4j Integration**: Migrate from PostgreSQL for advanced graph features
2. **Qdrant Integration**: Dedicated vector database for scale
3. **MCP Server**: Enable AI agents to query the knowledge graph
4. **Advanced Analytics**: Curriculum insights, difficulty calibration
5. **Personalization**: Adaptive learning paths based on student profiles

## Timeline Summary

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1 | Foundation | Enhanced models, DB schema |
| 2 | Entity Extraction | LLM integration, rules engine |
| 3 | Graph Storage | Storage service, migration |
| 4 | Community Detection | Clustering, summarization |
| 5 | Search Implementation | Local/global/hybrid search |
| 6 | API Integration | Endpoints, UI components |
| 7 | Pipeline Integration | Enhanced processor, batch ops |
| 8 | Testing & Optimization | Performance, test suite |

## Getting Started

1. **Review MVP 1 Code**: Ensure familiarity with existing system
2. **Set Up Development**: Clone repo, install dependencies
3. **Database Migration**: Run schema v2 migrations
4. **Start with Models**: Implement graph entity models
5. **Build Services**: Follow the phase order
6. **Test Incrementally**: Test each component before integration

## Conclusion

MVP 2 transforms the question extraction system into an intelligent knowledge platform. By adding GraphRAG capabilities, we enable semantic understanding, relationship discovery, and educational intelligence features that provide significant value beyond simple question storage.

The incremental approach ensures we can deliver value quickly while building toward a comprehensive solution. Starting with PostgreSQL keeps complexity low while providing a clear upgrade path to specialized graph and vector databases as the system scales.