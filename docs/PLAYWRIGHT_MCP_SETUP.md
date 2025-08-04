# Playwright MCP Server Setup

## Overview

The Playwright MCP (Model Context Protocol) server enables Claude to interact with web browsers for automated testing, web scraping, and browser automation tasks. This document explains how to set up and configure the Playwright MCP server.

## Installation

### 1. Update MCP Configuration

The Playwright MCP server has been added to the `.mcp.json` configuration file:

```json
{
  "mcpServers": {
    "playwright": {
      "command": "npx",
      "args": ["-y", "@playwright/mcp@latest"]
    }
  }
}
```

### 2. Package Information

- **Package**: `@playwright/mcp`
- **Version**: 0.0.32 (as of January 2025)
- **Publisher**: Microsoft Playwright Team
- **License**: Apache-2.0

## Features

The Playwright MCP server provides:

1. **Browser Automation**: Control Chromium, Firefox, and WebKit browsers
2. **Web Scraping**: Extract data from web pages
3. **Testing**: Automated E2E testing capabilities
4. **Screenshots**: Capture screenshots of web pages
5. **JavaScript Execution**: Run JavaScript in browser context
6. **Accessibility**: Access structured accessibility snapshots

## Configuration Options

The Playwright MCP server supports various command-line arguments:

```json
{
  "playwright": {
    "command": "npx",
    "args": [
      "-y", 
      "@playwright/mcp@latest",
      "--browser=chromium",  // Optional: specify browser
      "--headless"           // Optional: run in headless mode
    ]
  }
}
```

## Usage in PDF Question Extractor

The Playwright MCP can be used for:

1. **Visual Testing**: Capture screenshots of the web UI
2. **E2E Testing**: Automate testing of the question extraction workflow
3. **Performance Testing**: Measure UI response times
4. **Cross-browser Testing**: Verify compatibility across browsers

## Troubleshooting

### Common Issues

1. **MCP Server Not Starting**
   - Ensure Node.js and npm are installed
   - Check that npx can download packages
   - Verify network connectivity

2. **Browser Launch Failures**
   - Install browser dependencies: `npx playwright install-deps`
   - For Docker: Use playwright Docker images with browsers pre-installed

3. **Permission Issues**
   - Ensure the user has permissions to launch browsers
   - Check file system permissions for browser downloads

### Verification Steps

1. **Test Direct Installation**:
   ```bash
   npx @playwright/mcp@latest --help
   ```

2. **Check Browser Installation**:
   ```bash
   npx playwright install
   ```

3. **Verify MCP Configuration**:
   - Restart Claude Code after updating `.mcp.json`
   - Check Claude Code logs for MCP server startup messages

## Integration with Project

### Example Use Cases

1. **Test Web UI**:
   ```javascript
   // Playwright MCP can automate this
   - Navigate to http://localhost:8000
   - Upload a PDF file
   - Monitor progress
   - Verify questions are extracted
   ```

2. **Visual Regression Testing**:
   ```javascript
   // Capture screenshots for comparison
   - Screenshot of empty state
   - Screenshot during processing
   - Screenshot of results table
   ```

3. **Performance Monitoring**:
   ```javascript
   // Measure page load times
   - Time to interactive
   - API response times
   - UI update performance
   ```

## Next Steps

1. **Enable in Claude Code**: Restart Claude Code to load the updated MCP configuration
2. **Install Browsers**: Run `npx playwright install` if needed
3. **Create Test Scripts**: Develop E2E tests for the PDF Question Extractor
4. **Document Tests**: Add test documentation to the project

## References

- [Official Playwright MCP](https://github.com/microsoft/playwright-mcp)
- [Playwright Documentation](https://playwright.dev)
- [Model Context Protocol](https://github.com/anthropics/model-context-protocol)
- [@playwright/mcp on npm](https://www.npmjs.com/package/@playwright/mcp)