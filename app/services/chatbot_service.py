
import json
import random
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

class ChatbotService:
    def __init__(self, intents_path: str = "data/intents.json"):
        self.intents_path = intents_path
        self.intents = []
        self.patterns = []
        self.tags = []
        
        self._load_intents()
        self._train_intent_model()

    def _load_intents(self):
        if os.path.exists(self.intents_path):
            with open(self.intents_path, 'r') as f:
                data = json.load(f)
                self.intents = data.get("intents", [])
                for intent in self.intents:
                    for pattern in intent["patterns"]:
                        self.patterns.append(pattern.lower())
                        self.tags.append(intent["tag"])

    def _train_intent_model(self):
        if self.patterns:
            self.vectorizer = TfidfVectorizer()
            X = self.vectorizer.fit_transform(self.patterns)
            self.model = MultinomialNB()
            self.model.fit(X, self.tags)

    def get_reply(self, message: str):
        cleaned_msg = message.lower().strip()
        
        if hasattr(self, 'vectorizer') and self.patterns:
            vec = self.vectorizer.transform([cleaned_msg])
            predicted_tag = self.model.predict(vec)[0]
            probs = self.model.predict_proba(vec)[0]
            confidence = float(max(probs))

            for intent in self.intents:
                if intent["tag"] == predicted_tag:
                    reply = random.choice(intent["responses"])
                    return {
                        "user_message": message,
                        "bot_reply": reply,
                        "intent": predicted_tag,
                        "confidence": round(confidence, 2)
                    }

        return {
            "user_message": message,
            "bot_reply": "I'm sorry, I didn't quite understand that. Can you rephrase?",
            "intent": "unknown",
            "confidence": 0.0
        }

chatbot_service = ChatbotService()