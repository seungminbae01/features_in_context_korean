"""
construal_experiment.py

Demonstrates how argument structure constructions shift the predicted
semantic features of a Korean verb using the trained PLSR model.

Usage:
    python construal_experiment.py
"""

import pickle
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel


# ============================================================
# 1. Load the trained PLSR model
# ============================================================

MODEL_PATH = './trained_models/plsr.korean.binder.mbert.corpus.clusters5.15c'
NORMS_FILE = './data/external/binder_word_ratings/korean_binder_norms.csv'

print("Loading trained PLSR model...")
with open(MODEL_PATH, 'rb') as f:
    plsr_model = pickle.load(f)

# Load feature names from the norms file
df = pd.read_csv(NORMS_FILE)
# Get just the feature columns (exclude metadata)
metadata_cols = ['No', 'Word', 'WC', 'N', 'Mean R', 'LEN', 'FREQ', 'L10 FREQ',
                 'Orth', 'Orth_F', 'N1_F', 'N2_F', 'N3_F', 'IMG',
                 'Unnamed: 70', 'Unnamed: 80', 'Type', 'Super Category',
                 'Category', 'Kmeans28 Category']
feature_names = [c for c in df.columns if c not in metadata_cols and c != 'Word']
# Filter to only numeric columns
feature_names = [c for c in feature_names if df[c].dtype in ['float64', 'int64', 'float32']]
print(f"Loaded {len(feature_names)} feature names")


# ============================================================
# 2. Set up mBERT for extracting contextual embeddings
# ============================================================

print("Loading mBERT...")
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
bert_model.eval()
device = torch.device('cpu')
bert_model.to(device)


def get_contextual_embedding(word, sentence, layer=8):
    """Extract the mBERT layer-8 embedding for `word` in `sentence`."""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs, output_hidden_states=True)

    layer_output = outputs.hidden_states[layer]

    # Find the target word
    word_tokens = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sentence)
    pre_tokenized_words = [w[0] for w in word_tokens]

    target_idx = None
    # Exact match
    for i, w in enumerate(pre_tokenized_words):
        if w == word:
            target_idx = i
            break
    # Substring match
    if target_idx is None:
        for i, w in enumerate(pre_tokenized_words):
            if w.startswith(word) or word in w:
                target_idx = i
                break

    if target_idx is None:
        print(f"  WARNING: Could not find '{word}' in '{sentence}'")
        print(f"  Pre-tokenized words: {pre_tokenized_words}")
        return None

    token_span = inputs.word_to_tokens(target_idx)
    if token_span is None:
        return None

    word_start, word_end = token_span
    embedding = layer_output[0, word_start:word_end, :].mean(dim=0).cpu().numpy()
    return embedding


def predict_features(embedding):
    """Use the trained PLSR model to predict feature norms from an embedding."""
    # PLSR expects 2D input
    X = embedding.reshape(1, -1)
    predicted = plsr_model.predict(X)
    return predicted[0]


# ============================================================
# 3. Define test sentences with different constructions
# ============================================================

experiments = [
    {
        'verb': '깨다',
        'description': 'break',
        'conditions': [
            {
                'name': 'Transitive (Agent breaks Object)',
                'sentence': '그가 유리를 깨었다',
                'target': '깨었다',
            },
            {
                'name': 'Intransitive/Passive (Object breaks)',
                'sentence': '유리가 깨졌다',
                'target': '깨졌다',
            },
            {
                'name': 'Causative (Agent causes breaking)',
                'sentence': '그가 유리를 깨뜨렸다',
                'target': '깨뜨렸다',
            },
        ]
    },
    {
        'verb': '열다',
        'description': 'open',
        'conditions': [
            {
                'name': 'Transitive (Agent opens)',
                'sentence': '그녀가 문을 열었다',
                'target': '열었다',
            },
            {
                'name': 'Intransitive (Door opens)',
                'sentence': '문이 열렸다',
                'target': '열렸다',
            },
        ]
    },
    {
        'verb': '녹다',
        'description': 'melt',
        'conditions': [
            {
                'name': 'Transitive (Agent melts)',
                'sentence': '그가 얼음을 녹였다',
                'target': '녹였다',
            },
            {
                'name': 'Intransitive (Ice melts)',
                'sentence': '얼음이 녹았다',
                'target': '녹았다',
            },
        ]
    },
]


# ============================================================
# 4. Run the experiment
# ============================================================

print("\n" + "=" * 70)
print("CONSTRUAL EXPERIMENT: Argument Structure Constructions in Korean")
print("=" * 70)

for exp in experiments:
    print(f"\n{'─' * 70}")
    print(f"Verb: {exp['verb']} ({exp['description']})")
    print(f"{'─' * 70}")

    condition_predictions = {}

    for cond in exp['conditions']:
        print(f"\n  Condition: {cond['name']}")
        print(f"  Sentence: {cond['sentence']}")
        print(f"  Target:   {cond['target']}")

        emb = get_contextual_embedding(cond['target'], cond['sentence'])

        if emb is None:
            print("  ERROR: Could not extract embedding")
            continue

        predicted = predict_features(emb)
        condition_predictions[cond['name']] = predicted

        # Show top 10 predicted features
        top_indices = np.argsort(predicted)[::-1][:10]
        print(f"  Top 10 features:")
        for idx in top_indices:
            if idx < len(feature_names):
                print(f"    {feature_names[idx]:12s}  {predicted[idx]:.3f}")

    # Compare conditions pairwise
    if len(condition_predictions) >= 2:
        names = list(condition_predictions.keys())
        print(f"\n  === Feature Differences: {names[0]} vs {names[1]} ===")
        diff = condition_predictions[names[0]] - condition_predictions[names[1]]

        # Features that increase in first condition
        top_increase = np.argsort(diff)[::-1][:5]
        print(f"  Features HIGHER in [{names[0]}]:")
        for idx in top_increase:
            if idx < len(feature_names):
                print(f"    {feature_names[idx]:12s}  +{diff[idx]:.3f}")

        # Features that decrease in first condition
        top_decrease = np.argsort(diff)[:5]
        print(f"  Features HIGHER in [{names[1]}]:")
        for idx in top_decrease:
            if idx < len(feature_names):
                print(f"    {feature_names[idx]:12s}  +{abs(diff[idx]):.3f}")

        # Cosine similarity between the two predicted feature vectors
        cos_sim = np.dot(condition_predictions[names[0]], condition_predictions[names[1]]) / \
                  (np.linalg.norm(condition_predictions[names[0]]) * np.linalg.norm(condition_predictions[names[1]]))
        print(f"\n  Cosine similarity between conditions: {cos_sim:.4f}")

print("\n" + "=" * 70)
print("Done!")
