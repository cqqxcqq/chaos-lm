# inference/api_server.py
"""
CHAOS-LM FastAPI Server
REST API for text generation with anti-alignment models.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum
import time
import uuid
from datetime import datetime

from config.config import ChaosConfig, DegradationStyle, APIConfig


# Pydantic models for API
class GenerationMode(str, Enum):
    REVERSE_LOSS = "reverse_loss"
    ENTROPY_MAX = "entropy_max"
    SHIFTED_LABEL = "shifted_label"
    HYBRID = "hybrid"


class GenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    mode: Optional[GenerationMode] = None
    degradation_level: float = Field(0.5, ge=0.0, le=1.0)
    temperature: float = Field(1.0, ge=0.1, le=2.0)
    max_tokens: int = Field(128, ge=1, le=512)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1, le=100)
    style: Optional[str] = None
    add_marker: bool = True


class GenerationResponse(BaseModel):
    id: str
    text: str
    prompt: str
    degradation_level: float
    style: str
    entropy: float
    token_count: int
    processing_time_ms: float
    timestamp: str
    warning: str = "⚠️ UNRELIABLE OUTPUT - This text is intentionally degraded and should not be trusted."


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float


class MetricsResponse(BaseModel):
    total_requests: int
    average_latency_ms: float
    requests_per_minute: float
    error_rate: float


# Global state (will be initialized in create_app)
class AppState:
    def __init__(self):
        self.generator = None
        self.config = None
        self.start_time = time.time()
        self.request_count = 0
        self.total_latency = 0.0
        self.error_count = 0


app_state = AppState()


def create_app(
    generator=None,
    config: Optional[ChaosConfig] = None
) -> FastAPI:
    """
    Create FastAPI application with CHAOS-LM generator.
    
    Args:
        generator: ChaosGenerator instance
        config: ChaosConfig instance
    
    Returns:
        Configured FastAPI app
    """
    api_config = config.api if config else APIConfig()
    
    app = FastAPI(
        title="CHAOS-LM API",
        description="Anti-Alignment Language Model API for generating intentionally degraded text.",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_config.allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Store state
    app_state.generator = generator
    app_state.config = config
    app_state.start_time = time.time()
    
    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint"""
        return {
            "service": "CHAOS-LM",
            "status": "running",
            "docs": "/docs"
        }
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        return HealthResponse(
            status="healthy" if app_state.generator is not None else "degraded",
            model_loaded=app_state.generator is not None,
            version="1.0.0",
            uptime_seconds=time.time() - app_state.start_time
        )
    
    @app.get("/metrics", response_model=MetricsResponse)
    async def get_metrics():
        """Get API metrics"""
        uptime = time.time() - app_state.start_time
        rpm = (app_state.request_count / uptime) * 60 if uptime > 0 else 0
        avg_latency = app_state.total_latency / max(app_state.request_count, 1)
        error_rate = app_state.error_count / max(app_state.request_count, 1)
        
        return MetricsResponse(
            total_requests=app_state.request_count,
            average_latency_ms=avg_latency,
            requests_per_minute=rpm,
            error_rate=error_rate
        )
    
    @app.post("/generate", response_model=GenerationResponse)
    async def generate_text(request: GenerationRequest):
        """
        Generate text with CHAOS-LM.
        
        The generated text is intentionally degraded based on the degradation_level
        and should not be used for any task requiring accuracy.
        """
        if app_state.generator is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Please initialize the generator first."
            )
        
        start_time = time.time()
        app_state.request_count += 1
        
        try:
            # Parse style
            style = None
            if request.style:
                try:
                    style = DegradationStyle(request.style)
                except ValueError:
                    style = DegradationStyle.POETIC_NONSENSE
            
            # Generate
            result = app_state.generator.generate(
                prompt=request.prompt,
                degradation_level=request.degradation_level,
                style=style,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                add_marker=request.add_marker
            )
            
            processing_time = (time.time() - start_time) * 1000
            app_state.total_latency += processing_time
            
            return GenerationResponse(
                id=str(uuid.uuid4()),
                text=result.text,
                prompt=request.prompt,
                degradation_level=result.degradation_level,
                style=result.style,
                entropy=result.entropy,
                token_count=result.token_count,
                processing_time_ms=processing_time,
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            app_state.error_count += 1
            raise HTTPException(
                status_code=500,
                detail=f"Generation failed: {str(e)}"
            )
    
    @app.post("/generate/batch", response_model=List[GenerationResponse])
    async def generate_batch(prompts: List[str], request: GenerationRequest):
        """Generate text for multiple prompts"""
        if app_state.generator is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = []
        for prompt in prompts[:10]:  # Limit to 10 prompts
            request.prompt = prompt
            result = await generate_text(request)
            results.append(result)
        
        return results
    
    @app.post("/generate/sweep")
    async def degradation_sweep(
        prompt: str,
        levels: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
    ):
        """Generate text at multiple degradation levels"""
        if app_state.generator is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        results = app_state.generator.generate_with_degradation_sweep(
            prompt=prompt,
            levels=levels
        )
        
        return [
            {
                "degradation_level": r.degradation_level,
                "text": r.text,
                "entropy": r.entropy
            }
            for r in results
        ]
    
    @app.get("/config")
    async def get_config():
        """Get current configuration"""
        if app_state.config is None:
            return {"error": "Config not loaded"}
        return app_state.config.to_dict()
    
    @app.get("/styles")
    async def list_styles():
        """List available degradation styles"""
        return {
            "styles": [
                {
                    "name": style.value,
                    "description": {
                        "alien_syntax": "Rearranges syntax in non-human patterns",
                        "poetic_nonsense": "Breaks text into poetic, flowing lines",
                        "glitch_talk": "Injects glitch characters and artifacts",
                        "fake_profound": "Adds pseudo-philosophical framing",
                        "dream_logic": "Applies surreal, dream-like transitions"
                    }.get(style.value, "")
                }
                for style in DegradationStyle
            ]
        }
    
    return app


def run_server(
    generator,
    config: ChaosConfig,
    host: str = "0.0.0.0",
    port: int = 8000
):
    """Run the API server"""
    import uvicorn
    
    app = create_app(generator, config)
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=config.api.workers,
        reload=config.api.reload
    )