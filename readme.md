# Zero-Shot Multi-Modal Search Engine (CLIP)

A production-ready FastAPI service for zero-shot image classification and CLIP embedding generation using OpenAI's CLIP model via Hugging Face Transformers.

## Features

- **Zero-shot classification** — score images against any text labels without retraining
- **Image embeddings** — L2-normalized vectors for similarity search / vector databases
- **Text embeddings** — aligned with image embeddings in the same CLIP vector space
- **Fully Dockerized** — no external dependencies beyond the container

## Project Structure

```
multimodal-clip-search/
├── src/
│   ├── __init__.py
│   ├── config.py        # Pydantic settings (MODEL_ID, limits)
│   ├── embedder.py      # CLIPEngine — classification + embedding
│   └── app.py           # FastAPI routes + lifespan model loading
├── data/
│   └── sample_images/
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

## Quickstart

### Local
```bash
pip install -r requirements.txt
python src/app.py
```

### Docker
```bash
docker build -t clip-search .
docker run -p 8000:8000 clip-search
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check + model status |
| POST | `/classify` | Zero-shot image classification |
| POST | `/embed/image` | Get image embedding vector |
| POST | `/embed/text` | Get text embedding vector |

## Example — Classify

```bash
curl -X POST http://localhost:8000/classify \
  -F "labels=cat,dog,bird" \
  -F "file=@photo.jpg"
```

Response:
```json
{
  "filename": "photo.jpg",
  "scores": {
    "cat": 0.7231,
    "dog": 0.1893,
    "bird": 0.0876
  }
}
```

## Bugs Fixed from Original

| # | Bug | Impact |
|---|-----|--------|
| 1 | `model.eval()` missing | Non-deterministic inference (dropout active) |
| 2 | No empty label guard in engine | Silent crash on direct engine calls |
| 3 | No `truncation=True, max_length=77` | Shape errors on long labels (CLIP 77-token limit) |
| 4 | `squeeze()` → `squeeze(0)` | Single-label inference returned float not list |
| 5 | Module-level model init | Broke tests, cold-start failures, unmockable |
| 6 | `reload=True` in production | Double model loading, filesystem security risk |

## Models Supported

Any CLIP model on Hugging Face Hub. Set `MODEL_ID` in `.env`:
- `openai/clip-vit-base-patch32` (default, fastest)
- `openai/clip-vit-large-patch14` (higher accuracy)
- `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` (largest, best zero-shot)
