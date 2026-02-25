# Sentiment Analysis API

A production-grade REST API for sentiment analysis powered by HuggingFace's DistilBERT model and FastAPI.

Built as part of a Gen AI engineering roadmap — Week 2 project.

---

## Tech Stack

- **FastAPI** — REST API framework
- **HuggingFace Transformers** — DistilBERT model
- **Pydantic** — Data validation
- **Uvicorn** — ASGI server
- **HTML/CSS/JS** — Frontend interface

---

## Model

```
distilbert-base-uncased-finetuned-sst-2-english
```

- Fine-tuned on SST-2 (Stanford Sentiment Treebank)
- Labels: `POSITIVE` / `NEGATIVE`
- Confidence score: 0.0 → 1.0
- Max tokens: 512

---

## Project Structure

```
sentiment-api/
├── main.py               # FastAPI app, CORS, lifespan startup
├── ml_model.py           # Model loading (singleton pattern)
├── models.py             # Pydantic input/output models
├── routers/
│   └── sentiment.py      # API endpoints
├── sentiment-ui.html     # Frontend interface
├── requirements.txt      # Dependencies
└── README.md
```

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/sentiment/analyze` | Analyze single text |
| POST | `/sentiment/batch` | Analyze multiple texts (max 32) |
| GET | `/sentiment/model-info` | Model metadata |

---

## Setup & Run

**1. Clone the repo**
```bash
git clone https://github.com/yourusername/sentiment-api.git
cd sentiment-api
```

**2. Create and activate virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the API**
```bash
uvicorn main:app --reload
```

API runs at `http://localhost:8000`

> Note: First startup downloads the DistilBERT model (~250MB). Subsequent startups are instant.

---

## Usage

**Swagger UI** — Interactive API docs:
```
http://localhost:8000/docs
```

**Frontend UI** — Open in browser:
```
double click → sentiment-ui.html
```

---

## API Examples

**Single text analysis:**
```bash
curl -X POST http://localhost:8000/sentiment/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "I love this product!"}'
```

Response:
```json
{
  "text": "I love this product!",
  "label": "POSITIVE",
  "score": 0.9998
}
```

---

**Batch analysis:**
```bash
curl -X POST http://localhost:8000/sentiment/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I love this!", "This is terrible.", "It was okay."]}'
```

Response:
```json
{
  "results": [
    {"text": "I love this!", "label": "POSITIVE", "score": 0.9998},
    {"text": "This is terrible.", "label": "NEGATIVE", "score": 0.9985},
    {"text": "It was okay.", "label": "POSITIVE", "score": 0.6012}
  ],
  "count": 3
}
```

---

## Key Concepts Learned

- HuggingFace `pipeline` — bundles tokenizer + model + post-processing
- Singleton pattern — model loads once, reused across all requests
- FastAPI `lifespan` — pre-warm model at startup
- Pydantic validation — automatic input/output validation
- CORS middleware — allow frontend to talk to API
- Batch processing — HuggingFace handles lists natively

---

## Author

**Vikram V**  
Software Engineer → Gen AI Engineer  
[GitHub](https://github.com/yourusername) · [LinkedIn](https://linkedin.com/in/yourusername)
