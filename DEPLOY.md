# Deploying the dashboard

Target: **Streamlit Community Cloud** (free, connects straight to a GitHub repo).
Everything the platform needs is already in the repo.

| File | Why it is there |
|---|---|
| `requirements.txt` | Python deps; Streamlit Cloud installs from this automatically |
| `packages.txt` | `libgomp1` — the OpenMP runtime XGBoost links against. Without it the import fails at boot with `libgomp.so.1: cannot open shared object file` |
| `.streamlit/config.toml` | Headless server, XSRF on, usage stats off |
| `models/churn_model.joblib` | Committed deliberately (see below) |

## Why the model is committed

`.gitignore` blanket-ignores `*.joblib`, which is right for a working repo and wrong for
a hosted app: the app would boot, find no artefact and show its "run training first"
warning forever. The scoring model is small, so an explicit `!models/churn_model.joblib`
exception overrides the blanket rule and the app starts instantly with no cold-start
training step.

The DuckDB file is **not** committed — the dashboard reads the CSV directly and the model
is already fitted, so the ETL is only needed for development and testing.

## One-time setup

```powershell
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/<your-username>/ML-8_Customer_Churn.git
git push -u origin main
```

Then at <https://share.streamlit.io>:

1. **New app** → pick the repo → branch `main`
2. **Main file path**: `app/streamlit_app.py`
3. **Deploy**

First build takes a few minutes while the dependencies install. After that it is a URL
you can put on a CV.

## Check before you trust the link

- The **Predict** tab returns a number for a sensible input
- The **Model quality** tab renders `reports/summary.json` — if that is missing, the
  artefact did not ship and the `.gitignore` exception is not working
- Open the link in a private window: Streamlit Cloud apps are public by default, but a
  logged-out check is the only way to see what a recruiter sees

## Sleeping apps

Free-tier apps sleep after about a week of no traffic and wake on the next visit, taking
30–60 seconds. That is a bad first impression if someone opens the link cold from a CV.
Visit your own links every few days during placement season, or accept the wake time and
make sure the repo README carries the headline numbers so the page is worth the wait.

## Local alternatives

```powershell
.\tasks.ps1 app       # http://localhost:8501
.\tasks.ps1 api       # http://localhost:8000/docs
docker compose up      # both, containerised
```
