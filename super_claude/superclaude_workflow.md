# SuperClaude v3 Workflow Guide for PDF Question Extractor

## 🚀 Introduction

This guide shows how to use **SuperClaude v3** with MCP (Model Context Protocol) servers to build your PDF Question Extractor application efficiently. The combination of SuperClaude commands, intelligent personas, and MCP servers provides a powerful development environment.

### What's New in Your Setup
- ✅ **SuperClaude v3.0.0** installed and configured
- ✅ **MCP Servers** active: Filesystem, Sequential Thinking, Memory, and GitHub
- ✅ **Project-specific configuration** in `.claude/CLAUDE.md`
- ✅ **Intelligent routing** with the Orchestrator system

## 📋 Prerequisites

Before starting, ensure you have:
- ✅ Python 3.11+ installed
- ✅ Virtual environment activated (`source ../super_c/bin/activate`)
- ✅ SuperClaude v3 framework installed
- ✅ MCP servers configured (`.mcp.json` in project root)
- ✅ Claude Code restarted after MCP configuration
- ✅ Project opened in Claude Code

## 🛠️ Part 1: MCP Servers - Your Development Superpowers

### Active MCP Servers in Your Project

#### 1. **Filesystem Server** (`mcp__filesystem__`)
- Direct file operations without shell commands
- Secure access within project directory
- Example usage:
```python
# Reading files
mcp__filesystem__read_text_file(path="services/ocr_service.py")

# Listing directories
mcp__filesystem__list_directory(path="/project/root")

# Multi-file edits
mcp__filesystem__edit_file(path="app.py", edits=[...])
```

#### 2. **Sequential Thinking Server** (`mcp__sequential-thinking__`)
- Breaks down complex problems into steps
- Perfect for architectural decisions
- Automatically activated by `--think` flags

#### 3. **Memory Server** (`mcp__memory__`)
- Maintains context across sessions
- Stores project-specific knowledge
- Useful for tracking decisions and patterns

#### 4. **GitHub Server** (`mcp__github__`)
- Repository management
- PR creation and review
- Requires `GITHUB_TOKEN` environment variable

## 💻 Part 2: SuperClaude Commands with MCP Enhancement

These commands leverage MCP servers for enhanced functionality. The Orchestrator automatically selects the best tools and personas based on your request.

### 2.1 Project Analysis Commands (Enhanced with MCP)

#### Analyze with Sequential Thinking
```
/analyze @. --think
```
- **MCP Enhancement**: Sequential Thinking server breaks down analysis
- **Filesystem server**: Reads all project files efficiently
- **Auto-activates**: Analyzer persona for deep insights

#### Explain with Context
```
/explain "Mistral OCR integration patterns"
```
- **Memory server**: Recalls previous explanations
- **Sequential server**: Structures complex explanations
- **Auto-activates**: Mentor persona for educational content

#### Smart Estimation
```
/estimate "complete MVP based on PRD" --think-hard
```
- **Sequential server**: Analyzes PRD complexity
- **Filesystem server**: Examines existing code
- **Auto-activates**: Architect persona for accurate estimates

### 2.2 Building with MCP-Powered Intelligence

#### Design with Wave Orchestration
```
/design "PDF question extraction system based on PRD" --wave-mode
```
- **Wave Mode**: Multi-stage design process
- **Sequential server**: Architectural analysis
- **Memory server**: Stores design decisions
- **Auto-activates**: Architect + Backend personas

#### Build with Direct File Operations
```
/build --type service "Mistral OCR integration"
```
- **Filesystem server**: Creates files directly
- **Sequential server**: Plans implementation steps
- **Memory server**: Tracks patterns for consistency
- **Example output**:
  ```python
  # services/ocr_service.py created with:
  # - Mistral API client setup
  # - Error handling with tenacity
  # - Structured markdown parsing
  # - Cost optimization logic
  ```

#### Implement with Context Awareness
```
/implement "PostgreSQL vector storage with pgvector"
```
- **Reads PRD**: Understands embedding requirements (768 dimensions)
- **Sequential server**: Plans database schema
- **Creates**: models.py, vector_operations.py, migrations
- **Auto-activates**: Backend persona for database expertise

### 2.3 Testing with MCP Integration

#### Unit Tests with Coverage
```
/test --type unit "OCR processing" --validate
```
- **Filesystem server**: Creates test files in tests/
- **Sequential server**: Designs test scenarios
- **Auto-validation**: Ensures >80% coverage

#### Integration Tests with Real APIs
```
/test --type integration "Mistral OCR + Gemini extraction"
```
- **Creates**: Mock responses for API testing
- **Validates**: Error handling and retry logic
- **Memory server**: Remembers API response patterns

#### E2E Tests with Playwright (Future)
```
/test e2e "PDF upload → extraction → review → save"
```
- **Playwright server**: Browser automation
- **Sequential server**: User journey mapping
- **Creates**: Comprehensive E2E test suite

### 2.4 Optimization with Intelligence

#### Improve with Loop Mode
```
/improve --quality "OCR accuracy" --loop --iterations 3
```
- **Loop mode**: Iterative improvements
- **Sequential server**: Analyzes current accuracy
- **Implements**: Progressive enhancements
- **Memory server**: Tracks improvement metrics

#### Performance Optimization
```
/improve --perf "bulk PDF processing" --think-hard
```
- **Analyzes**: Current bottlenecks
- **Suggests**: Batch processing, async operations
- **Implements**: Connection pooling, caching
- **Auto-activates**: Performance persona

#### Smart Cleanup
```
/cleanup @. --safe-mode
```
- **Filesystem server**: Scans entire project
- **Identifies**: Unused imports, dead code
- **Safe mode**: Conservative changes only
- **Auto-activates**: Refactorer persona

### 2.5 Troubleshooting Commands

#### Debug Issues
```
/troubleshoot "OCR accuracy issues"
```

#### Analyze Performance
```
/analyze --focus performance "database query optimization"
```
**What these do:** Help identify and fix problems in your code.
**When to use:** When encountering bugs or performance issues.

### 2.6 Documentation Commands

#### Create User Guides
```
/document --type guide "setup and installation instructions"
```

#### Document APIs
```
/document --type api "REST endpoints for question management"
```
**What these do:** Generate documentation for different aspects of your project.
**When to use:** To maintain up-to-date documentation.

## 🎭 Part 3: Intelligent Personas with MCP

Personas now work seamlessly with MCP servers for enhanced capabilities:

### Backend Persona + MCP
```
/implement "PostgreSQL + pgvector setup" --persona-backend
```
- **Filesystem server**: Creates database modules
- **Sequential server**: Plans schema design
- **Expertise**: Connection pooling, indexing, migrations

### Frontend Persona + MCP
```
/build "Tabulator.js question review interface" --persona-frontend
```
- **Filesystem server**: Creates static files
- **Memory server**: Applies consistent UI patterns
- **Expertise**: Accessibility, responsive design, performance

### Architect Persona + Sequential Thinking
```
/design "scalable question extraction pipeline" --persona-architect --think-hard
```
- **Sequential server**: System decomposition
- **Wave mode**: Progressive design refinement
- **Expertise**: Scalability, maintainability, patterns

### Auto-Activation Examples
- "OCR service" → Backend persona activated
- "UI component" → Frontend persona activated
- "system design" → Architect persona activated
- "test coverage" → QA persona activated

## 📊 Part 4: Complete MCP-Enhanced Workflow

Here's how to build the PDF Question Extractor using SuperClaude v3 with MCP servers:

### Step 1: Environment Setup & MCP Verification
```bash
# Activate environment
cd /mnt/d/Python/Projects/Dave
source super_c/bin/activate
cd questions_pdf_to_sheet

# Verify MCP servers are active (in Claude Code)
# You should see: filesystem, sequential-thinking, memory, github
```

### Step 2: Project Analysis with MCP
```
# Comprehensive analysis using Sequential Thinking
/analyze @. --think --wave-mode

# MCP servers will:
# - Filesystem: Read PRD.md and existing code
# - Sequential: Break down requirements
# - Memory: Store project understanding

# Design with architectural intelligence
/design "PDF extraction system from PRD" --persona-architect --wave-mode

# Accurate estimation with context
/estimate "MVP with Mistral OCR + Gemini + PostgreSQL" --think-hard
```

### Step 3: Implementation with Direct File Operations

#### Phase 1: Core Services
```
# OCR Service with error handling
/implement "Mistral OCR service with retry logic" --persona-backend
# Creates: services/ocr_service.py with tenacity retry

# LLM Parser with structured output
/implement "Gemini question parser with Pydantic schemas"
# Creates: services/llm_service.py with JSON schema validation

# Database layer with pgvector
/implement "PostgreSQL models with vector embeddings"
# Creates: database/models.py, vector_operations.py, init_db.py
```

#### Phase 2: API Development
```
# Flask API with all endpoints
/build --type api "complete REST API from PRD endpoints"
# Creates: app.py, api/routes.py with all endpoints

# Batch processing pipeline
/implement "PDF folder processing pipeline"
# Creates: services/pdf_processor.py with async processing
```

#### Phase 3: Frontend
```
# Tabulator.js interface
/build "question review UI with Tabulator" --persona-frontend
# Creates: static/index.html, js/app.js, css/style.css

# Auto-save functionality  
/implement "auto-save with 1-second delay"
# Updates: js/app.js with debounced saving
```

### Step 4: Testing with Validation
```
# Unit tests with mocks
/test --type unit "OCR service" --validate
# Creates: tests/test_ocr_service.py with API mocks

# Integration tests
/test --type integration "PDF → OCR → LLM → DB pipeline"
# Creates: tests/test_integration.py with fixtures

# Performance testing
/test --benchmark "bulk PDF processing throughput"
# Creates: tests/test_performance.py with metrics

# Quality validation
/analyze --focus quality @. --validate
# MCP servers check: code quality, test coverage, security
```

### Step 5: Optimization with Loop Mode
```
# Iterative performance improvement
/improve --perf "PDF processing pipeline" --loop --iterations 3
# Iteration 1: Adds connection pooling
# Iteration 2: Implements batch processing
# Iteration 3: Adds caching layer

# OCR accuracy enhancement
/improve --quality "OCR accuracy for poor scans" --think
# Analyzes: Current accuracy issues
# Implements: Pre-processing, confidence thresholds

# Cost optimization
/improve "API usage costs" --focus efficiency
# Implements: Request batching, caching, smart chunking
```

### Step 6: Documentation & Deployment Prep
```
# API documentation
/document --type api "REST endpoints with examples"
# Creates: docs/API.md with request/response examples

# Deployment guide
/document --type guide "production deployment"
# Creates: docs/DEPLOYMENT.md with:
# - PostgreSQL setup with pgvector
# - Environment configuration
# - Performance tuning
# - MCP integration setup

# User manual with screenshots
/document "user guide with workflow examples" --persona-scribe
# Creates: docs/USER_GUIDE.md with step-by-step instructions

# Generate .env.example
/build ".env.example with all required variables"
# Creates: .env.example with documented variables
```

## 💡 Pro Tips with MCP Servers

### 1. **Leverage MCP Servers**
- **Filesystem**: No more copy-paste - files are created/edited directly
- **Sequential**: Complex problems are automatically broken down
- **Memory**: Your decisions and patterns are remembered
- **GitHub**: Version control integrated into your workflow

### 2. **Use Intelligence Flags**
- `--think`: 4K tokens for complex analysis
- `--think-hard`: 10K tokens for system-wide understanding
- `--wave-mode`: Multi-stage processing for best results
- `--loop`: Iterative improvements with validation

### 3. **Trust Auto-Activation**
- Personas activate based on context
- MCP servers coordinate automatically
- Wave mode triggers for complex operations
- Quality gates validate all changes

### 4. **Cost-Effective Development**
- Memory server reduces repeated analysis
- Caching prevents duplicate API calls  
- Smart chunking optimizes token usage
- Batch operations minimize API costs

### 5. **Debugging with MCP**
```
/troubleshoot "OCR accuracy issues" --introspect
# Shows Claude's thinking process
# Sequential server structures the investigation
# Memory server recalls similar issues
```

## 🔧 Troubleshooting with MCP

### MCP Server Issues
- **"MCP server not found"**: Restart Claude Code after adding `.mcp.json`
- **"Permission denied"**: MCP filesystem is restricted to project directory
- **"Sequential thinking timeout"**: Break complex problems into smaller parts

### Development Issues
- **OCR API errors**: Check Mistral API key in `.env`
- **Database connection failed**: Verify PostgreSQL + pgvector installation
- **High API costs**: Use `/improve "cost optimization"` for solutions

### Performance Issues
- **Slow processing**: Enable batch mode with `/improve --perf`
- **Memory issues**: Implement streaming for large PDFs
- **Token limits**: Use smart chunking strategies

### Quick Fixes
```
# Analyze any error
/troubleshoot "[paste error message]" --think

# Check project health
/analyze @. --focus issues --validate

# Review recent changes
/analyze --focus "recent modifications" --introspect
```

## 📚 Resources & References

### Project Resources
- **PRD**: `/docs/PRD.md` - Complete requirements
- **MCP Config**: `.mcp.json` - Server configuration
- **Project Config**: `.claude/CLAUDE.md` - SuperClaude settings

### API Documentation
- [Mistral OCR API](https://docs.mistral.ai/) - ~94.9% accuracy
- [Gemini API](https://ai.google.dev/) - Structured output
- [pgvector](https://github.com/pgvector/pgvector) - Vector search

### SuperClaude Resources
- [SuperClaude Framework](https://github.com/SuperClaude-Org/SuperClaude_Framework)
- [MCP Documentation](https://modelcontextprotocol.io/)
- [Claude Code Best Practices](https://docs.anthropic.com/claude/docs)

## 🎯 Next Steps with Your Enhanced Setup

### Immediate Actions
1. **Test MCP servers**: Try `mcp__filesystem__list_directory(path=".")`
2. **Analyze with intelligence**: `/analyze @. --think --wave-mode`
3. **Design the system**: `/design "MVP implementation plan" --persona-architect`
4. **Start building**: `/implement "database setup with pgvector"`

### Development Priorities (from PRD)
1. **Week 1**: PostgreSQL + pgvector setup, Mistral OCR integration
2. **Week 2**: Gemini parser, embedding generation
3. **Week 3**: Flask API, Tabulator.js UI
4. **Week 4**: Testing, optimization, documentation

### Cost Optimization Target
- OCR: $50-100 per 1000 papers
- LLM: $5-10 per 1000 papers
- **Total: ~$55-110 per 1000 papers**

### Success Metrics
- ≥90% question extraction accuracy
- <5% false positives
- <200ms UI response time
- 100% approved questions saved correctly

## 🚀 Ready to Build!

Your SuperClaude v3 setup with MCP servers is fully configured. The combination of intelligent commands, specialized personas, and MCP servers will accelerate your development significantly.

Start with: `/analyze @. --think` to see the full power of your enhanced setup!