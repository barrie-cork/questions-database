-- PDF Question Extractor Database Schema
-- PostgreSQL with pgvector extension

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_trgm;  -- For text search

-- Temporary storage for review
CREATE TABLE IF NOT EXISTS extracted_questions (
    id BIGSERIAL PRIMARY KEY,
    question_number TEXT,
    marks INTEGER,
    year TEXT,
    level TEXT,
    topics TEXT[],  -- Native PostgreSQL array
    question_type TEXT,
    question_text TEXT NOT NULL,
    source_pdf TEXT NOT NULL,
    status TEXT DEFAULT 'pending',  -- pending/approved/rejected
    modified BOOLEAN DEFAULT FALSE,
    extraction_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB  -- Flexible additional data
);

-- Main questions table
CREATE TABLE IF NOT EXISTS questions (
    id BIGSERIAL PRIMARY KEY,
    question_number TEXT,
    marks INTEGER,
    year TEXT,
    level TEXT,
    topics TEXT[],  -- Native PostgreSQL array
    question_type TEXT,
    question_text TEXT NOT NULL,
    source_pdf TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB  -- Additional flexible data
);

-- Question embeddings table (separate for versioning)
CREATE TABLE IF NOT EXISTS question_embeddings (
    id BIGSERIAL PRIMARY KEY,
    question_id BIGINT NOT NULL,
    embedding vector(768),  -- Gemini embedding (768 dimensions)
    model_name VARCHAR(100) NOT NULL DEFAULT 'gemini-embedding-001',
    model_version VARCHAR(50) NOT NULL DEFAULT '1.0',
    embedding_generated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_question FOREIGN KEY (question_id) 
        REFERENCES questions(id) ON DELETE CASCADE,
    CONSTRAINT unique_question_model 
        UNIQUE (question_id, model_name, model_version)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_questions_source ON questions(source_pdf);
CREATE INDEX IF NOT EXISTS idx_questions_year ON questions(year);
CREATE INDEX IF NOT EXISTS idx_questions_type ON questions(question_type);
CREATE INDEX IF NOT EXISTS idx_questions_topics ON questions USING GIN(topics);
CREATE INDEX IF NOT EXISTS idx_questions_metadata ON questions USING GIN(metadata);
CREATE INDEX IF NOT EXISTS idx_questions_text_search 
    ON questions USING GIN(to_tsvector('english', question_number || ' ' || question_text));

-- Create HNSW index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_question_embeddings_hnsw 
    ON question_embeddings 
    USING hnsw (embedding vector_cosine_ops);

-- Create indexes for extracted questions (temporary table)
CREATE INDEX IF NOT EXISTS idx_extracted_source ON extracted_questions(source_pdf);
CREATE INDEX IF NOT EXISTS idx_extracted_status ON extracted_questions(status);
CREATE INDEX IF NOT EXISTS idx_extracted_modified ON extracted_questions(modified);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for questions table
DROP TRIGGER IF EXISTS update_questions_updated_at ON questions;
CREATE TRIGGER update_questions_updated_at 
    BEFORE UPDATE ON questions 
    FOR EACH ROW 
    EXECUTE FUNCTION update_updated_at_column();

-- Create read-only user for MCP access (optional, run as superuser)
-- CREATE ROLE readonly_user WITH LOGIN PASSWORD 'readonly_password';
-- GRANT CONNECT ON DATABASE question_bank TO readonly_user;
-- GRANT USAGE ON SCHEMA public TO readonly_user;
-- GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readonly_user;

-- PostgreSQL performance tuning (run as superuser)
-- ALTER SYSTEM SET shared_buffers = '256MB';
-- ALTER SYSTEM SET effective_cache_size = '1GB';
-- ALTER SYSTEM SET maintenance_work_mem = '64MB';
-- ALTER SYSTEM SET random_page_cost = 1.1;  -- For SSD storage
-- SELECT pg_reload_conf();