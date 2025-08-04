# Database Design - PDF Question Extractor

## Overview
PostgreSQL 16+ with pgvector extension for storing questions and their embeddings, enabling both traditional queries and vector similarity search.

## Database Schema

### Extensions Required
```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text search
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";  -- For UUID generation
```

### Tables

#### 1. extracted_questions (Temporary Storage)
```sql
CREATE TABLE extracted_questions (
    id BIGSERIAL PRIMARY KEY,
    question_number TEXT,
    marks INTEGER CHECK (marks >= 0),
    year TEXT,
    level TEXT,
    topics TEXT[],  -- PostgreSQL native array
    question_type TEXT,
    question_text TEXT NOT NULL,
    source_pdf TEXT NOT NULL,
    status TEXT DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected')),
    modified BOOLEAN DEFAULT FALSE,
    extraction_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,  -- Flexible additional data
    
    -- Indexes for performance
    INDEX idx_extracted_status ON extracted_questions(status),
    INDEX idx_extracted_source ON extracted_questions(source_pdf),
    INDEX idx_extracted_date ON extracted_questions(extraction_date)
);
```

#### 2. questions (Permanent Storage)
```sql
CREATE TABLE questions (
    id BIGSERIAL PRIMARY KEY,
    question_number TEXT,
    marks INTEGER CHECK (marks >= 0),
    year TEXT,
    level TEXT,
    topics TEXT[],  -- PostgreSQL native array
    question_type TEXT,
    question_text TEXT NOT NULL,
    source_pdf TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB,  -- Additional flexible data
    
    -- Constraints
    CONSTRAINT unique_question_per_source UNIQUE (source_pdf, question_number)
);

-- Indexes for performance
CREATE INDEX idx_questions_source ON questions(source_pdf);
CREATE INDEX idx_questions_year ON questions(year);
CREATE INDEX idx_questions_type ON questions(question_type);
CREATE INDEX idx_questions_level ON questions(level);
CREATE INDEX idx_questions_topics ON questions USING GIN(topics);
CREATE INDEX idx_questions_metadata ON questions USING GIN(metadata);

-- Full-text search index
CREATE INDEX idx_questions_text_search 
    ON questions 
    USING GIN(to_tsvector('english', 
        COALESCE(question_number, '') || ' ' || 
        COALESCE(question_text, '')
    ));

-- Trigger to update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_questions_updated_at 
    BEFORE UPDATE ON questions 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();
```

#### 3. question_embeddings (Vector Storage)
```sql
CREATE TABLE question_embeddings (
    id BIGSERIAL PRIMARY KEY,
    question_id BIGINT NOT NULL,
    embedding vector(768),  -- Gemini embedding dimensions
    model_name VARCHAR(100) NOT NULL DEFAULT 'gemini-embedding-001',
    model_version VARCHAR(50) NOT NULL,
    embedding_generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key with cascade delete
    CONSTRAINT fk_question 
        FOREIGN KEY (question_id) 
        REFERENCES questions(id) 
        ON DELETE CASCADE,
    
    -- Ensure one embedding per question per model version
    CONSTRAINT unique_question_model 
        UNIQUE (question_id, model_name, model_version)
);

-- HNSW index for fast similarity search
CREATE INDEX idx_question_embeddings_hnsw 
    ON question_embeddings 
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

#### 4. processing_jobs (Track PDF Processing)
```sql
CREATE TABLE processing_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    status TEXT NOT NULL DEFAULT 'pending' 
        CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    total_files INTEGER DEFAULT 0,
    processed_files INTEGER DEFAULT 0,
    total_questions INTEGER DEFAULT 0,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE INDEX idx_jobs_status ON processing_jobs(status);
CREATE INDEX idx_jobs_created ON processing_jobs(created_at);
```

#### 5. processing_files (Track Individual Files)
```sql
CREATE TABLE processing_files (
    id BIGSERIAL PRIMARY KEY,
    job_id UUID NOT NULL,
    filename TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'processing', 'completed', 'failed')),
    questions_extracted INTEGER DEFAULT 0,
    error_message TEXT,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    CONSTRAINT fk_job 
        FOREIGN KEY (job_id) 
        REFERENCES processing_jobs(id) 
        ON DELETE CASCADE
);

CREATE INDEX idx_files_job ON processing_files(job_id);
CREATE INDEX idx_files_status ON processing_files(status);
```

## Views

### 1. question_statistics
```sql
CREATE VIEW question_statistics AS
SELECT 
    year,
    level,
    question_type,
    COUNT(*) as question_count,
    AVG(marks) as avg_marks,
    ARRAY_AGG(DISTINCT unnest(topics)) as all_topics
FROM questions
GROUP BY year, level, question_type;
```

### 2. processing_job_summary
```sql
CREATE VIEW processing_job_summary AS
SELECT 
    j.id,
    j.status,
    j.total_files,
    j.processed_files,
    j.total_questions,
    COUNT(f.id) as actual_files,
    SUM(CASE WHEN f.status = 'completed' THEN 1 ELSE 0 END) as completed_files,
    SUM(f.questions_extracted) as extracted_questions,
    j.created_at,
    j.completed_at,
    EXTRACT(EPOCH FROM (j.completed_at - j.started_at)) as duration_seconds
FROM processing_jobs j
LEFT JOIN processing_files f ON j.id = f.job_id
GROUP BY j.id;
```

## Functions

### 1. Find Similar Questions
```sql
CREATE OR REPLACE FUNCTION find_similar_questions(
    target_embedding vector(768),
    similarity_threshold FLOAT DEFAULT 0.8,
    max_results INTEGER DEFAULT 10
)
RETURNS TABLE (
    question_id BIGINT,
    question_text TEXT,
    topics TEXT[],
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        q.id,
        q.question_text,
        q.topics,
        1 - (qe.embedding <=> target_embedding) as similarity
    FROM questions q
    JOIN question_embeddings qe ON q.id = qe.question_id
    WHERE 1 - (qe.embedding <=> target_embedding) > similarity_threshold
    ORDER BY qe.embedding <=> target_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;
```

### 2. Search Questions by Topic
```sql
CREATE OR REPLACE FUNCTION search_questions_by_topic(
    search_topic TEXT,
    search_text TEXT DEFAULT NULL
)
RETURNS TABLE (
    id BIGINT,
    question_text TEXT,
    topics TEXT[],
    relevance FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        q.id,
        q.question_text,
        q.topics,
        CASE 
            WHEN search_text IS NOT NULL THEN
                ts_rank(to_tsvector('english', q.question_text), 
                       plainto_tsquery('english', search_text))
            ELSE 1.0
        END as relevance
    FROM questions q
    WHERE search_topic = ANY(q.topics)
        AND (search_text IS NULL OR 
             to_tsvector('english', q.question_text) @@ 
             plainto_tsquery('english', search_text))
    ORDER BY relevance DESC;
END;
$$ LANGUAGE plpgsql;
```

### 3. Duplicate Detection
```sql
CREATE OR REPLACE FUNCTION find_duplicate_questions(
    check_embedding vector(768),
    duplicate_threshold FLOAT DEFAULT 0.95
)
RETURNS TABLE (
    question_id BIGINT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        qe.question_id,
        1 - (qe.embedding <=> check_embedding) as similarity
    FROM question_embeddings qe
    WHERE 1 - (qe.embedding <=> check_embedding) > duplicate_threshold
    ORDER BY similarity DESC;
END;
$$ LANGUAGE plpgsql;
```

## Security

### 1. Read-Only User for MCP
```sql
-- Create read-only role
CREATE ROLE readonly_user WITH LOGIN PASSWORD 'secure_readonly_password';

-- Grant permissions
GRANT CONNECT ON DATABASE question_bank TO readonly_user;
GRANT USAGE ON SCHEMA public TO readonly_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
GRANT SELECT ON ALL SEQUENCES IN SCHEMA public TO readonly_user;

-- Default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT SELECT ON TABLES TO readonly_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT SELECT ON SEQUENCES TO readonly_user;
```

### 2. Application User
```sql
-- Create application user with limited permissions
CREATE ROLE app_user WITH LOGIN PASSWORD 'secure_app_password';

-- Grant permissions
GRANT CONNECT ON DATABASE question_bank TO app_user;
GRANT USAGE ON SCHEMA public TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO app_user;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO app_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO app_user;

-- Default privileges
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT USAGE, SELECT ON SEQUENCES TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public 
    GRANT EXECUTE ON FUNCTIONS TO app_user;
```

## Performance Optimization

### 1. PostgreSQL Configuration
```sql
-- Optimize for vector operations
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET random_page_cost = 1.1;  -- For SSD
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
ALTER SYSTEM SET max_parallel_workers = 8;

-- Reload configuration
SELECT pg_reload_conf();
```

### 2. Maintenance Tasks
```sql
-- Regular maintenance script
CREATE OR REPLACE FUNCTION perform_maintenance()
RETURNS void AS $$
BEGIN
    -- Vacuum and analyze main tables
    VACUUM ANALYZE questions;
    VACUUM ANALYZE question_embeddings;
    VACUUM ANALYZE extracted_questions;
    
    -- Update statistics
    ANALYZE;
    
    -- Reindex if needed
    REINDEX TABLE CONCURRENTLY question_embeddings;
END;
$$ LANGUAGE plpgsql;
```

### 3. Monitoring Queries
```sql
-- Check index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;

-- Check table sizes
SELECT 
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check slow queries
SELECT 
    query,
    calls,
    total_time,
    mean_time,
    max_time
FROM pg_stat_statements
WHERE query NOT LIKE '%pg_stat_statements%'
ORDER BY mean_time DESC
LIMIT 10;
```

## Backup Strategy

### 1. Backup Script
```bash
#!/bin/bash
# backup_database.sh

BACKUP_DIR="/backups/postgres"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DB_NAME="question_bank"

# Create backup
pg_dump -U postgres -d $DB_NAME -F c -f "$BACKUP_DIR/backup_$TIMESTAMP.dump"

# Keep only last 7 days of backups
find $BACKUP_DIR -name "backup_*.dump" -mtime +7 -delete
```

### 2. Restore Script
```bash
#!/bin/bash
# restore_database.sh

BACKUP_FILE=$1
DB_NAME="question_bank"

# Restore database
pg_restore -U postgres -d $DB_NAME -c $BACKUP_FILE
```

## Migration Strategy

Using Alembic for schema versioning:

```python
# alembic.ini configuration
[alembic]
script_location = migrations
sqlalchemy.url = postgresql://user:password@localhost/question_bank

# Migration example
"""Initial schema

Revision ID: 001
Create Date: 2024-01-01
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

def upgrade():
    # Create tables
    op.execute('CREATE EXTENSION IF NOT EXISTS vector')
    op.execute('CREATE EXTENSION IF NOT EXISTS pg_trgm')
    # ... rest of schema creation

def downgrade():
    # Drop tables in reverse order
    op.drop_table('question_embeddings')
    op.drop_table('questions')
    # ...
```