import os
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from sqlalchemy import text

# Internal imports
from config import Config
from api.routes import router as api_router
from api.schemas.responses import ErrorResponse, ValidationErrorResponse, HealthResponse
from database.session import engine
from database.models import Base

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path(Config.LOG_DIR) / 'app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Ensure required directories exist
for directory in [Config.UPLOAD_DIR, Config.CACHE_DIR, Config.LOG_DIR]:
    Path(directory).mkdir(exist_ok=True)

# Create FastAPI application
app = FastAPI(
    title="PDF Question Extractor API",
    description="API for extracting and managing exam questions from PDF files using OCR and LLM processing",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:8000",  # FastAPI dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
        # Add production origins as needed
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
)


# Global exception handlers

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors"""
    logger.warning(f"Validation error for {request.url}: {exc.errors()}")
    
    error_details = []
    for error in exc.errors():
        error_details.append({
            "field": " -> ".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
            "input": error.get("input")
        })
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ValidationErrorResponse(
            message="Request validation failed",
            errors=error_details
        ).dict()
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP error {exc.status_code} for {request.url}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            message=exc.detail or "An error occurred",
            error_code=f"HTTP_{exc.status_code}"
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all other exceptions"""
    logger.error(f"Unhandled error for {request.url}: {str(exc)}")
    logger.error(traceback.format_exc())
    
    # Don't expose internal errors in production
    if Config.APP_ENV == "production":
        message = "An internal server error occurred"
        detail = None
    else:
        message = str(exc)
        detail = traceback.format_exc()
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            message=message,
            detail=detail,
            error_code="INTERNAL_SERVER_ERROR"
        ).dict()
    )


# Startup and shutdown events

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting PDF Question Extractor API...")
    
    try:
        # Create database tables if they don't exist
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database tables initialized")
        
        # Validate required API keys
        if not Config.MISTRAL_API_KEY:
            logger.warning("MISTRAL_API_KEY not configured - OCR functionality will be limited")
        
        if not Config.GOOGLE_API_KEY:
            logger.warning("GOOGLE_API_KEY not configured - LLM and embedding functionality will be limited")
        
        logger.info("PDF Question Extractor API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on application shutdown"""
    logger.info("Shutting down PDF Question Extractor API...")
    
    try:
        # Close database connections
        await engine.dispose()
        logger.info("Database connections closed")
        
        logger.info("PDF Question Extractor API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")


# Mount static files (for web UI)
static_path = Path("static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


# Root endpoint
@app.get("/", include_in_schema=False)
async def root():
    """Serve the main web interface"""
    static_index = Path("static/index.html")
    if static_index.exists():
        return FileResponse(static_index)
    else:
        return {
            "message": "PDF Question Extractor API",
            "version": "1.0.0",
            "docs": "/api/docs",
            "health": "/health"
        }


# Health check endpoint
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API and service status
    """
    try:
        # Test database connection
        database_connected = True
        try:
            async with engine.begin() as conn:
                await conn.execute(text("SELECT 1"))
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            database_connected = False
        
        # Check service availability
        services = {
            "database": "healthy" if database_connected else "unhealthy",
            "ocr_service": "configured" if Config.MISTRAL_API_KEY else "not_configured",
            "llm_service": "configured" if Config.GOOGLE_API_KEY else "not_configured",
            "embedding_service": "configured" if Config.GOOGLE_API_KEY else "not_configured"
        }
        
        # Determine overall status
        overall_status = "healthy" if database_connected else "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.utcnow(),
            version="1.0.0",
            database_connected=database_connected,
            services=services
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Health check failed"
        )


# Include API routes
app.include_router(api_router, prefix="/api", tags=["API"])


# Additional middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests"""
    start_time = datetime.utcnow()
    
    # Log request
    logger.info(f"Request: {request.method} {request.url}")
    
    # Process request
    response = await call_next(request)
    
    # Log response
    process_time = (datetime.utcnow() - start_time).total_seconds()
    logger.info(f"Response: {response.status_code} in {process_time:.3f}s")
    
    return response


def create_app() -> FastAPI:
    """Factory function to create the FastAPI app"""
    return app


if __name__ == "__main__":
    # Run the application with uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=Config.APP_ENV == "development",
        log_level=Config.LOG_LEVEL.lower(),
        access_log=True
    )