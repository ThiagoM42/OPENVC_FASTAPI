from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io

app = FastAPI()

@app.get("/")
def health():
    return {"status": "OpenCV API running 🚀"}

@app.post("/process")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_array = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, buffer = cv2.imencode(".jpg", gray)
    io_buf = io.BytesIO(buffer)

    return StreamingResponse(io_buf, media_type="image/jpeg")