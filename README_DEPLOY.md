# Deploying the Recommendation Engine demo (free hosting)

Two services, two hosts — same pattern as your other live projects:

| Service | What it is | Where it lives |
|---|---|---|
| `api/` | FastAPI + Faiss + trained two-tower model | Render (free web service) |
| `ui/`  | Streamlit dashboard | Streamlit Community Cloud |

Unlike a RAG-style app, there's no external API key needed here — the model
is trained on synthetic data and its weights are already committed under
`model/artefacts/` (~4.6MB total), so the API serves immediately on boot
with no ingestion or training step required.

---

## 1. Push this folder to GitHub

```bash
cd rec_engine
git init
git add -A
git commit -m "Recommendation engine — deploy-ready"
git branch -M main
git remote add origin https://github.com/Yourtech-01/rec-engine.git
git push -u origin main
```

(Create the empty repo on GitHub first, or use `gh repo create`.)

## 2. Deploy the API on Render

1. Go to https://dashboard.render.com → **New** → **Blueprint**.
2. Connect the GitHub repo. Render reads `render.yaml` automatically and
   proposes a web service named `rec-engine-api`.
3. Click **Apply**. No secrets to enter — there's no external API key for
   this project. Build takes ~3–5 minutes (installing torch + faiss).
4. Once live, note the URL, e.g. `https://rec-engine-api.onrender.com`.
5. Sanity check:
   - `https://rec-engine-api.onrender.com/health` → `{"status":"ok","model":"two-tower","index_size":5000}`
   - `https://rec-engine-api.onrender.com/docs` → interactive Swagger UI

**If the build fails or the service crashes with an out-of-memory error:**
Render's free tier is 512MB RAM. `torch` + `faiss-cpu` with this model's
small embedding tables (2,000 users × 5,000 items × 64 dims) should comfortably
fit, but if it doesn't, upgrade this one service to **Starter** ($7/mo) for
the demo, then downgrade afterward.

## 3. Deploy the UI on Streamlit Community Cloud

1. Go to https://share.streamlit.io → **New app**.
2. Pick the same GitHub repo, set:
   - **Main file path:** `ui/app.py`
3. In **App settings → Secrets**, paste:
   ```toml
   API_URL = "https://rec-engine-api.onrender.com"
   ```
   (use the exact Render URL from step 2, no trailing slash)
4. Deploy. Your live demo URL will look like
   `https://your-app-name.streamlit.app`.

## 4. Before a live demo

Render's free instance sleeps after 15 minutes of no traffic and takes
~30–60s to wake up. **Open your API's `/health` URL about a minute before
you demo** to warm it up.

## 5. Retraining with different data (optional)

The bundled artefacts were trained on synthetic interaction data (see
`model/train.py`) purely so the demo has something realistic to recommend
against. To retrain (e.g. after tweaking hyperparameters):

```bash
pip install -r requirements.txt        # full dev requirements, includes mlflow
python -m model.train
python -m model.build_index
git add model/artefacts
git commit -m "Retrain model"
git push
```

Render will redeploy automatically on push and pick up the new weights.

---

## Local testing before you push

```bash
pip install -r requirements.txt
python -m model.train
python -m model.build_index
docker compose up --build
```

- API: http://localhost:8000/docs
- UI: http://localhost:8501
- MLflow tracking UI: http://localhost:5000
