
from fastapi import FastAPI, UploadFile, File
from app.schemas import (
    SentimentRequest, SentimentResponse,
    ChatbotRequest, ChatbotResponse,
    FaceRecognitionResponse, ProductClassificationResponse
)
from app.services.nlp_service import sentiment_service
from app.services.chatbot_service import chatbot_service
from app.services.cv_service import vision_service

app = FastAPI(
    title="Smart Retail & Customer Intelligence Platform API",
    description="Unified API serving Computer Vision, NLP, and Chatbot models.",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"status": "online", "message": "Smart Retail AI Gateway is running on macOS."}

@app.post("/analyze-sentiment", response_model=SentimentResponse)
def analyze_sentiment(payload: SentimentRequest):
    return sentiment_service.analyze(payload.text)

@app.post("/chatbot", response_model=ChatbotResponse)
def chatbot_response(payload: ChatbotRequest):
    return chatbot_service.get_reply(payload.message)

@app.post("/recognize-face", response_model=FaceRecognitionResponse)
async def recognize_face(file: UploadFile = File(...)):
    image_bytes = await file.read()
    return vision_service.process_face_image(image_bytes)

@app.post("/classify-product", response_model=ProductClassificationResponse)
async def classify_product(file: UploadFile = File(...)):
    image_bytes = await file.read()
    return vision_service.classify_product_image(image_bytes)

@app.get("/dashboard/stats")
def dashboard_stats():
    return {
        "total_store_visits_today": 128,
        "returning_loyalty_customers": 45,
        "sentiment_distribution": {"Positive": 0.72, "Neutral": 0.18, "Negative": 0.10},
        "top_product_category": "Shoes"
    }
