import torch
from transformers import DistilBertTokenizer, DistilBertModel
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

# Paths
DATA_PATH = os.path.join('data', 'scam_dataset.csv')
MODELS_DIR = 'models'
TRANSFORMER_MODEL_PATH = os.path.join(MODELS_DIR, 'scam_distilbert_head.pkl')

def get_embeddings(texts, model, tokenizer, device, batch_size=32):
    model.eval()
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embedding (first token)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

def train_transformer():
    print("Loading prepared dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Legit class encoding
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['scam_type'])
    
    # Manual Oversampling for tiny classes before split
    print("Handling tiny classes for stratification...")
    class_counts = df['scam_type'].value_counts()
    tiny_classes = class_counts[class_counts < 4].index.tolist()
    
    dfs_to_add = []
    for cls in tiny_classes:
        df_cls = df[df['scam_type'] == cls]
        dfs_to_add.append(pd.concat([df_cls] * 5))
    
    if dfs_to_add:
        df = pd.concat([df] + dfs_to_add).reset_index(drop=True)
        
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['text'], df['type_encoded'], test_size=0.2, stratify=df['type_encoded'], random_state=42
    )

    print("Initializing DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    print(f"Extracting features for training set ({len(X_train_text)} samples)...")
    X_train_eb = get_embeddings(X_train_text.tolist(), model, tokenizer, device)
    
    print(f"Extracting features for test set ({len(X_test_text)} samples)...")
    X_test_eb = get_embeddings(X_test_text.tolist(), model, tokenizer, device)
    
    print("Training Head (Logistic Regression) on transformer embeddings...")
    # Use balanced class weight for imbalance
    head = LogisticRegression(max_iter=1000, class_weight='balanced')
    head.fit(X_train_eb, y_train)
    
    print("\n--- Evaluation (DistilBERT + Head) ---")
    y_pred = head.predict(X_test_eb)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    # Save artifacts
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    joblib.dump(head, TRANSFORMER_MODEL_PATH)
    # Tokenizer is saved by name, DistilBertModel is standard. Head is our custom part.
    print(f"Transformer head saved to {TRANSFORMER_MODEL_PATH}")

if __name__ == "__main__":
    train_transformer()
