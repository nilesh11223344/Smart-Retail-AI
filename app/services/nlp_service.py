
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class SentimentService:
    def __init__(self):
        texts = [
            "I love this product, it is amazing!", "Great quality and fast delivery",
            "Terrible customer service, very disappointed", "Horrible quality, broken item",
            "The product is average, nothing special", "It works fine, okay package"
        ]
        labels = ["Positive", "Positive", "Negative", "Negative", "Neutral", "Neutral"]
        
        self.vectorizer = TfidfVectorizer()
        X = self.vectorizer.fit_transform(texts)
        self.model = LogisticRegression()
        self.model.fit(X, labels)

    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.strip()

    def analyze(self, text: str):
        cleaned = self._clean_text(text)
        vec = self.vectorizer.transform([cleaned])
        prediction = self.model.predict(vec)[0]
        probs = self.model.predict_proba(vec)[0]
        confidence = float(max(probs))
        
        return {
            "text": text,
            "sentiment": prediction,
            "confidence": round(confidence, 2)
        }

sentiment_service = SentimentService()