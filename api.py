from fastapi import FastAPI, HTTPException
from ml.predict import predict_url

app = FastAPI()

@app.post("/predict") 
# curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"url": "https://www.google.com"}'
def predict(request: dict):
    url = request['url']
    if not url:
        raise HTTPException(status_code=400, detail="No URL provided")
    label, prob = predict_url(url)
    return {"label": int(label), "probability": prob}

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}