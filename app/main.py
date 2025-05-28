from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import csv
from io import StringIO
import pathlib
from .model import LogisticRegression

app = FastAPI()

BASE_DIR = pathlib.Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

headers: List[str] = []
data: List[List[float]] = []
labels: List[float] = []
model: LogisticRegression | None = None


def html_page(content: str) -> HTMLResponse:
    return HTMLResponse(
        f"<html><head><title>Lead Propensity Demo</title></head><body>{content}</body></html>"
    )


@app.get("/", response_class=HTMLResponse)
def index():
    html_file = BASE_DIR / "static" / "index.html"
    return HTMLResponse(html_file.read_text())


@app.post("/upload")
async def upload(file: UploadFile):
    content = await file.read()
    reader = csv.reader(StringIO(content.decode()))
    global headers, data, labels, model
    headers = next(reader)
    rows = list(reader)
    data = [list(map(float, row[:-1])) for row in rows]
    labels = [float(row[-1]) for row in rows]
    model = LogisticRegression()
    model.fit(data, labels)
    leads = [
        {"features": row, "score": prob}
        for row, prob in zip(data, model.predict_proba(data))
    ]
    return {"headers": headers, "leads": leads}


@app.post("/add")
async def add(request: Request):
    if model is None:
        return {"error": "No model"}
    form = await request.form()
    features = [float(form[h]) for h in headers]
    score = model.predict_single(features)
    return {"features": features, "score": score}
