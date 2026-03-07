import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

# Paths
DATA_PATH = os.path.join('data', 'scam_dataset.csv')
MODELS_DIR = 'models'
XGB_MODEL_PATH = os.path.join(MODELS_DIR, 'scam_xgb_model.pkl')
VECTORIZER_PATH = os.path.join(MODELS_DIR, 'scam_tfidf.pkl')
ENCODER_PATH = os.path.join(MODELS_DIR, 'scam_label_encoder.pkl')

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text

def train_xgb():
    print("Loading prepared dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Preprocessing
    print("Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Label Encoding
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['scam_type'])
    legit_label = le.transform(['Legit'])[0]
    
    # Split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['cleaned_text'], df['type_encoded'], test_size=0.2, stratify=df['type_encoded'], random_state=42
    )
    
    # Oversampling on TEXT
    print("Balancing classes with RandomOverSampler on text...")
    ros = RandomOverSampler(random_state=42)
    X_train_ros, y_train_res = ros.fit_resample(X_train_text.values.reshape(-1, 1), y_train)
    X_train_ros = X_train_ros.flatten()
    
    # Vectorization
    print("Vectorizing balanced text...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train_ros)
    X_test_tfidf = tfidf.transform(X_test_text)
    
    # Train XGBoost
    print(f"Training XGBoost (Model A) on {X_train_tfidf.shape[0]} samples...")
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb.fit(X_train_tfidf, y_train_res)
    
    # Evaluation
    print("\n--- Evaluation (XGBoost) ---")
    y_pred = xgb.predict(X_test_tfidf)
    y_proba = xgb.predict_proba(X_test_tfidf)
    
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Binary Evaluation
    y_test_bin = (y_test != legit_label).astype(int)
    y_pred_bin = (y_pred != legit_label).astype(int)
    y_proba_bin = np.delete(y_proba, legit_label, axis=1).sum(axis=1)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall (Scam): {recall_score(y_test_bin, y_pred_bin):.4f}")
    print(f"Precision (Scam): {precision_score(y_test_bin, y_pred_bin):.4f}")
    print(f"F1 Score (Scam): {f1_score(y_test_bin, y_pred_bin):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test_bin, y_proba_bin):.4f}")
    
    # Save
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    joblib.dump(xgb, XGB_MODEL_PATH)
    joblib.dump(tfidf, VECTORIZER_PATH)
    joblib.dump(le, ENCODER_PATH)
    print(f"Model and artifacts saved to {MODELS_DIR}/")

if __name__ == "__main__":
    train_xgb()
