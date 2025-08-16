# Build: ML-Powered API (Scikit-learn + FastAPI) in Docker, deploy to AWS/GCP

**Role:** You are a senior ML engineer + backend engineer.  
**Outcome:** Train a simple Scikit-learn model (predict sales from historical data), serve via FastAPI, containerize with Docker, and deploy to cloud (AWS or GCP). I want this as a small, production-lean learning project.

## Scope & Requirements

### 1. Dataset & Problem

- Use a small tabular dataset (synthetic or public) with columns like: `date`, `store_id`, `promo_flag`, `price`, `units_sold` (target = `units_sold` or `sales`).
- Include a script to generate a synthetic dataset if none is provided.
- Split train/val/test with time-aware split.

### 2. Modeling (Scikit-learn)

- Baseline: LinearRegression (or Ridge).
- Optional: RandomForestRegressor for comparison.
- Feature engineering: date parts (dow, month), lag features (e.g., previous 7-day average), one-hot for categoricals.
- Persist trained model + feature pipeline with `joblib`.
- Metrics: MAE, RMSE; print and save to `artifacts/metrics.json`.

### 3. Project Structure

```
ml_api/
  data/                # raw/processed (gitignored)
  notebooks/           # optional EDA
  src/
    data_prep.py
    train.py
    predict.py
    schema.py          # pydantic input/output schemas
    model_store.py     # load/save model and preprocessor
  api/
    main.py            # FastAPI app
    routers/
      health.py
      predict.py
  tests/
    test_data_prep.py
    test_api.py
  artifacts/           # model.joblib, preprocessor.joblib, metrics.json
  Dockerfile
  docker-compose.yml   # local run
  requirements.txt
  README.md
```

### 4. FastAPI Endpoints

- `GET /health` → `{status:"ok", model_version:"<hash/date>"}`
- `POST /predict` → input batch of records (JSON list). Validate with Pydantic, run through preprocessor + model, return predictions + inference time.
- `GET /metrics` → reads `artifacts/metrics.json`.

### 5. Quality & Testing

- Include unit tests (pytest): data prep, training (smoke), and API (with TestClient).
- Add `ruff` or `flake8` + `black` formatting.
- Type hints where useful.

### 6. Docker & Local Dev

- `Dockerfile` multi-stage (build -> runtime, slim Python).
- `docker-compose.yml` to run API on `http://localhost:8000` and mount local artifacts for dev.
- Healthcheck in Docker.

### 7. Deployment (pick one and document both)

#### AWS Option:
- Build & push image to ECR.
- Deploy on ECS Fargate or Elastic Beanstalk (single container).
- Include IaC snippet (Terraform or AWS CDK) **or** CLI steps.

#### GCP Option:
- Build & push to Artifact Registry.
- Deploy to Cloud Run (min instances = 0 or 1), set concurrency and memory.
- Expose `/health` and `/predict`. Include env vars for model path, log level.

### 8. Observability

- Structured JSON logging (uvicorn + app logs).
- Basic request timing middleware; log latency and status code.
- Startup log includes model load success + version.

### 9. Docs

`README.md` with:
- Quickstart (venv, train, run locally).
- API usage examples (curl + sample payload).
- Docker build/run.
- Deployment steps for AWS and GCP.
- Troubleshooting tips.

### 10. Stretch Goals (optional)

- Simple CI (GitHub Actions): lint, test, build Docker, push on tag.
- Model registry pattern: versioned `artifacts/model_v{timestamp}.joblib`.
- Basic drift check (compare recent input stats vs. training stats).

## Deliverables

- All code files per structure above.
- Sample dataset or generator script.
- Trained model artifacts and metrics.
- Working FastAPI container that serves predictions locally and instructions to deploy to AWS/GCP.

## Constraints

- Keep dependencies lightweight (FastAPI, uvicorn, scikit-learn, pandas, pydantic, joblib, pytest, ruff/black).
- No heavy MLOps platforms; this is a learning project.

## What to Output

1. Final repo tree and all file contents.
2. Clear step-by-step commands for:
   - Train model locally.
   - Run API locally (uvicorn and Docker).
   - Deploy to AWS and to GCP (choose one to fully script, document both).
3. Example requests/responses and sample JSON payload.
4. Short "Why" section explaining design choices.

---

*If you want, I can also tailor this for **your exact stack** (e.g., AWS ECS vs. GCP Cloud Run, CI of choice) and generate the full codebase in one go.*