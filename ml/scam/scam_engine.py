import joblib
import os
import torch
import numpy as np
import json
from transformers import DistilBertTokenizer, DistilBertModel
from scam_rules import get_psychological_signals

# Paths
MODELS_DIR = 'models'
XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'scam_xgb_model.pkl')
XGB_TFIDF_PATH = os.path.join(MODELS_DIR, 'scam_tfidf.pkl')
XGB_LE_PATH = os.path.join(MODELS_DIR, 'scam_label_encoder.pkl')
BERT_HEAD_PATH = os.path.join(MODELS_DIR, 'scam_distilbert_head.pkl')

class ScamAlertEngine:
    def __init__(self):
        print("Initializing Scam Alert Engine...")
        # Load XGBoost artifacts
        self.xgb_model = joblib.load(XGB_MODEL_PATH)
        self.tfidf = joblib.load(XGB_TFIDF_PATH)
        self.le = joblib.load(XGB_LE_PATH)
        self.legit_label_idx = self.le.transform(['Legit'])[0]
        
        # Load Transformer artifacts
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_model.to(self.device).eval()
        
        # Load BERT head (if exists)
        if os.path.exists(BERT_HEAD_PATH):
            self.bert_head = joblib.load(BERT_HEAD_PATH)
        else:
            print("Warning: DistilBERT head not found. Using partial scoring.")
            self.bert_head = None

    def _get_bert_prob(self, text):
        if not self.bert_head:
            return 0.5
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            probs = self.bert_head.predict_proba(embedding)[0]
            # Prob of SCAM (any class except Legit)
            scam_prob = np.delete(probs, self.legit_label_idx).sum()
            return scam_prob

    def _get_xgb_prob(self, text):
        features = self.tfidf.transform([text])
        probs = self.xgb_model.predict_proba(features)[0]
        # Prob of SCAM (any class except Legit)
        scam_prob = np.delete(probs, self.legit_label_idx).sum()
        return scam_prob

    def detect_scam(self, text):
        # 1. ML Probabilities
        xgb_prob = self._get_xgb_prob(text)
        bert_prob = self._get_bert_prob(text)
        avg_ml_prob = (xgb_prob + bert_prob) / 2
        
        # 2. Rule-Based Signals
        rule_data = get_psychological_signals(text)
        rule_score = rule_data['rule_score']
        
        # 3. Risk Score Computation
        # Risk Score = (ML Prob * 70) + Rule Score (0-30 max)
        final_risk = (avg_ml_prob * 70) + rule_score
        
        # Cap at 100
        final_risk = min(100, max(0, final_risk))
        
        # 4. Determine Scam Type and Confidence
        features = self.tfidf.transform([text])
        type_idx = self.xgb_model.predict(features)[0]
        scam_type = self.le.inverse_transform([type_idx])[0]
        
        scam_detected = final_risk > 50
        confidence = round(float(avg_ml_prob), 4)
        
        # Normalized Risk Level
        risk_level = "Low"
        if final_risk > 85: risk_level = "Critical"
        elif final_risk > 70: risk_level = "High"
        elif final_risk > 40: risk_level = "Medium"

        return {
            "scam_detected": bool(scam_detected),
            "scam_type": scam_type,
            "risk_score": round(float(final_risk), 2),
            "risk_level": risk_level,
            "psychological_trigger": rule_data['primary_trigger'],
            "signals": rule_data['signals'],
            "confidence": confidence
        }

if __name__ == "__main__":
    # Test initialization
    try:
        engine = ScamAlertEngine()
        test_samples = [
            "Your KYC will expire today. Update immediately to avoid account suspension.",
            "Your Amazon order has been delivered successfully.",
            "Congratulations! You won ₹10,00,000 in lucky draw."
        ]
        for msg in test_samples:
            print(f"\nAnalyzing: {msg}")
            print(json.dumps(engine.detect_scam(msg), indent=2))
    except Exception as e:
        print(f"Engine test skipped or failed: {e}")
