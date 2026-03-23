# Customer Retention & Pricing Advisor

Predict customer churn risk and return a retention recommendation using a FastAPI backend + Streamlit frontend.

## Project Structure

- `api.py` - FastAPI backend (`POST /predict`)
- `app.py` - Streamlit frontend (calls the backend)
- `models/model.pkl` - trained model
- `models/features.pkl` - feature column list used during training

## Run Locally

1) Install dependencies:

```bash
pip install -r requirements.txt
```

2) Start the API:

```bash
uvicorn api:app --host 0.0.0.0 --port 10000
```

3) Start the Streamlit app (in a new terminal):

```bash
streamlit run app.py
```

Note: `app.py` defaults to calling `http://127.0.0.1:8000/predict`. If you run the API on port `10000`, update `API_URL` in `app.py` (or run the API on port `8000`).

## Deployment Notes

### Deploy API on Render (FastAPI)

- **Environment**: Python
- **Build command**: `pip install -r requirements.txt`
- **Start command**: `uvicorn api:app --host 0.0.0.0 --port 10000`
- Ensure `models/model.pkl` and `models/features.pkl` are included in your repo (or provided as Render persistent assets).

### Deploy UI on Streamlit Community Cloud (Streamlit)

- Connect your GitHub repo and set `app.py` as the entry point.
- Ensure `requirements.txt` is present at repo root.
- If your API is deployed (e.g., Render), update `API_URL` in `app.py` to your public backend URL.
