#!/bin/bash
# Activate MCP servers for PDF Question Extractor project

# Check if in super_c environment
if [[ "$VIRTUAL_ENV" != *"super_c"* ]]; then
    echo "Warning: Not in super_c environment. MCP servers may not function correctly."
    echo "Please run: source /mnt/d/Python/Projects/Dave/super_c/bin/activate"
fi

# Export MCP configuration path
export CLAUDE_MCP_CONFIG="$PWD/.claude/servers/mcp_config.json"
export CLAUDE_PROJECT_MCP_ENABLED=true

# Set project-specific environment variables
export CLAUDE_PROJECT_NAME="PDF Question Extractor"
export CLAUDE_PROJECT_PERSONAS="analyzer,backend,frontend,qa"
export CLAUDE_MCP_SERVERS="context7,sequential,magic,playwright"

echo "✅ MCP servers configured for PDF Question Extractor project"
echo "   - Context7: Documentation and best practices"
echo "   - Sequential: Complex analysis and problem solving"
echo "   - Magic: UI component generation"
echo "   - Playwright: E2E testing and automation"
echo ""
echo "📁 Configuration: $CLAUDE_MCP_CONFIG"
echo "🚀 Ready to use SuperClaude v3 with MCP servers!"