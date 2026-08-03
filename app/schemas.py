
from pydantic import BaseModel
from typing import Optional

class SentimentRequest(BaseModel):
    text: str

class SentimentResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float

class ChatbotRequest(BaseModel):
    message: str
    user_id: Optional[str] = "guest"

class ChatbotResponse(BaseModel):
    user_message: str
    bot_reply: str
    intent: str
    confidence: float

class FaceRecognitionResponse(BaseModel):
    customer_id: Optional[str]
    status: str
    confidence: float

class ProductClassificationResponse(BaseModel):
    predicted_category: str
    confidence: float