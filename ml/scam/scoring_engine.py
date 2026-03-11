import joblib
import os
import re
import json
from cleaner import clean_text

# Paths
MODELS_DIR = 'models'
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')
LR_MODEL_PATH = os.path.join(MODELS_DIR, 'lr_model.pkl')
SVM_MODEL_PATH = os.path.join(MODELS_DIR, 'svm_model.pkl')

# Heuristic Signals
SUSPICIOUS_KEYWORDS = ["urgent", "verify", "lottery", "prize", "bank", "otp", "reward", "click", "free", "offer"]
URL_PATTERN = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

class SpamScoringEngine:
    def __init__(self):
        # Load models and vectorizer
        try:
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            self.lr_model = joblib.load(LR_MODEL_PATH)
            self.svm_model = joblib.load(SVM_MODEL_PATH)
        except Exception as e:
            print(f"Error loading models: {e}")
            raise

    def get_risk_score(self, text):
        # 1. Preprocess
        cleaned = clean_text(text)
        tfidf_features = self.vectorizer.transform([cleaned])
        
        # 2. ML Probability (60% LR + 40% SVM)
        lr_prob = self.lr_model.predict_proba(tfidf_features)[0, 1]
        svm_prob = self.svm_model.predict_proba(tfidf_features)[0, 1]
        
        ml_prob = (0.6 * lr_prob) + (0.4 * svm_prob)
        
        # 3. Heuristic Signals
        signals = []
        heuristic_boost = 0.0
        
        # Keyword detection
        detected_keywords = [kw for kw in SUSPICIOUS_KEYWORDS if kw in text.lower()]
        if detected_keywords:
            signals.append(f"Suspicious keywords detected: {', '.join(detected_keywords)}")
            heuristic_boost += 0.1 * len(detected_keywords)
            
        # URL detection
        urls = re.findall(URL_PATTERN, text)
        if urls:
            signals.append(f"URLs detected: {len(urls)}")
            heuristic_boost += 0.2
            
        # 4. Final Probability (clamped to 1.0)
        final_prob = min(1.0, ml_prob + heuristic_boost)
        
        # 5. Determine Risk Level
        if final_prob < 0.3:
            risk_level = "Low"
            recommendation = "Message appears safe."
        elif final_prob < 0.6:
            risk_level = "Medium"
            recommendation = "Message contains some suspicious elements. Exercise caution."
        elif final_prob < 0.85:
            risk_level = "High"
            recommendation = "High risk of spam/scam. Do not share personal information."
        else:
            risk_level = "Critical"
            recommendation = "Dangerous content detected. Block and report immediately."
            
        # 6. Structured Output
        result = {
            "spam_probability": round(float(final_prob), 4),
            "risk_level": risk_level,
            "signals_detected": signals,
            "recommendation": recommendation
        }
        
        return result

if __name__ == "__main__":
    engine = SpamScoringEngine()
    
    test_messages = [
        "Hey! Are we still meeting for lunch today?",
        "URGENT: Your account at bank is compromised. Click here to verify: http://scam-site.com/verify",
        "Congratulations! You've won a $1000 gift card. Call now to claim your prize.",
        "Your OTP for bank login is 123456. Do not share it with anyone."
    ]
    
    for msg in test_messages:
        print(f"\nMessage: {msg}")
        score = engine.get_risk_score(msg)
        print(json.dumps(score, indent=2))
