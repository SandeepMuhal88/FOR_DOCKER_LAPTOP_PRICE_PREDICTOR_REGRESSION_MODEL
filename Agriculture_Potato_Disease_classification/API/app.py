from fastapi import FastAPI, UploadFile, File
from model import load_model, predict

app = FastAPI()
model = load_model()

@app.get("/")
def home():
    return {"status": "Potato Disease API Running"}

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    result = predict(model, file.file)
    return {"prediction": result}
