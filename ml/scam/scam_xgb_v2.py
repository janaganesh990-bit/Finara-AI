import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import re
import nltk
from nltk.corpus import stopwords

# Ensure data dir exists
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Download stopwords
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = " ".join([word for word in text.split() if word not in STOPWORDS])
    return text

def train_xgb_v2():
    print("Loading scam dataset...")
    df = pd.read_csv('data/scam_dataset.csv')
    
    # Preprocessing
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Label Encoding
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['scam_type'])
    legit_label = le.transform(['Legit'])[0]
    
    # Manual Oversampling for tiny classes before split
    # Since some classes have only 1-2 samples, we duplicate them
    print("Handling tiny classes...")
    class_counts = df['scam_type'].value_counts()
    tiny_classes = class_counts[class_counts < 4].index.tolist()
    
    dfs_to_add = []
    for cls in tiny_classes:
        df_cls = df[df['scam_type'] == cls]
        # Duplicate 5 times to ensure enough samples for stratification and training
        dfs_to_add.append(pd.concat([df_cls] * 5))
    
    if dfs_to_add:
        df = pd.concat([df] + dfs_to_add).reset_index(drop=True)
    
    # Split
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['cleaned_text'], df['type_encoded'], test_size=0.2, stratify=df['type_encoded'], random_state=42
    )
    
    # Vectorization
    print("Vectorizing...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_test_tfidf = tfidf.transform(X_test_text)
    
    # Train XGBoost with stratified weights (simpler than SMOTE/ROS here)
    print("Training XGBoost...")
    # Calculate class weights manually to handle imbalance
    from sklearn.utils.class_weight import compute_sample_weight
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb.fit(X_train_tfidf, y_train, sample_weight=sample_weights)
    
    # Evaluation
    print("\n--- Evaluation (XGBoost v2) ---")
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
    joblib.dump(xgb, 'models/scam_xgb_model.pkl')
    joblib.dump(tfidf, 'models/scam_tfidf.pkl')
    joblib.dump(le, 'models/scam_label_encoder.pkl')
    print("Model saved to models/")

if __name__ == "__main__":
    train_xgb_v2()
