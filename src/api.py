"""REST API around the trained MiniGPT.

Run locally:   uvicorn src.api:app --reload --port 8000
Docs page:     http://localhost:8000/docs
Override the checkpoint with the MINIGPT_CKPT environment variable.
"""
import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.generate import generate_text, load_model
from src.utils import count_parameters

CKPT_PATH = os.environ.get("MINIGPT_CKPT", "checkpoints/mini_gpt_best.pt")
app = FastAPI(title="MiniGPT (Transformer built from scratch)", version="1.0.0")
_state = {}


def get_model():
    """Lazy-load once per process so the server starts quickly."""
    if "model" not in _state:
        if not os.path.exists(CKPT_PATH):
            raise HTTPException(status_code=503, detail=f"checkpoint not found at {CKPT_PATH}")
        model, tok = load_model(CKPT_PATH, device="cpu")
        _state.update(model=model, tok=tok)
    return _state["model"], _state["tok"]


class GenerateRequest(BaseModel):
    prompt: str = Field("\n", max_length=1000)
    max_new_tokens: int = Field(200, ge=1, le=500)
    temperature: float = Field(0.8, gt=0.0, le=2.0)
    top_k: Optional[int] = Field(None, ge=1)
    greedy: bool = False
    seed: Optional[int] = None


class GenerateResponse(BaseModel):
    prompt: str
    generated: str = Field(description="prompt followed by the sampled continuation")
    model_parameters: int
    context_length: int


@app.get("/health")
def health():
    return {"status": "ok", "checkpoint": CKPT_PATH, "loaded": "model" in _state}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    model, tok = get_model()
    try:
        text = generate_text(model, tok, req.prompt, max_new_tokens=req.max_new_tokens,
                             temperature=req.temperature, top_k=req.top_k,
                             greedy=req.greedy, seed=req.seed, device="cpu")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return GenerateResponse(prompt=req.prompt, generated=text,
                            model_parameters=count_parameters(model),
                            context_length=model.cfg.context_length)
