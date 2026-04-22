"""
construal_diffvec.py

Uses the diffvec method to analyze how argument structure constructions
shift the semantic features of Korean verbs.

Usage:
    python construal_diffvec.py
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from itertools import combinations


NORMS_FILE = './data/external/binder_word_ratings/korean_binder_norms.csv'
EMBEDDINGS_FILE = './data/processed/multipro_embeddings/layer8clusters1_corpus.txt'

# Korean -> English feature translations
FEATURE_TRANSLATIONS = {
    '시각': 'Vision',
    '밝음': 'Bright',
    '어두움': 'Dark',
    '색깔': 'Color',
    '패턴': 'Pattern',
    '크기': 'Size',
    '작음': 'Small',
    '움직임': 'Motion',
    '생체 움직임': 'Biomotion',
    '빠름': 'Fast',
    '느림': 'Slow',
    '모양': 'Shape',
    '얼굴': 'Face',
    '몸': 'Body',
    '촉각': 'Touch',
    '온도': 'Temperature',
    '질감': 'Texture',
    '무게': 'Weight',
    '통증': 'Pain',
    '청각': 'Audition',
    '큰 소리': 'Loud',
    '낮은': 'Low',
    '높음': 'High',
    '소리': 'Sound',
    '음악': 'Music',
    '말': 'Speech',
    '미각': 'Taste',
    '후각': 'Smell',
    '머리': 'Head',
    '상지': 'Upper limb',
    '하지': 'Lower limb',
    '랜드마크': 'Landmark',
    '경로': 'Path',
    '장면': 'Scene',
    '가까운': 'Near',
    '쪽으로': 'Toward',
    '멀리': 'Away',
    '숫자': 'Number',
    '시간': 'Time',
    '지속시간': 'Duration',
    '긴': 'Long',
    '짧은': 'Short',
    '결과적인': 'Consequential',
    '사회적인': 'Social',
    '인간': 'Human',
    '의사소통': 'Communication',
    '자기': 'Self',
    '인지': 'Cognition',
    '이익': 'Benefit',
    '해': 'Harm',
    '쾌적한': 'Pleasant',
    '불쾌한': 'Unpleasant',
    '행복한': 'Happy',
    '슬픈': 'Sad',
    '화난': 'Angry',
    '역겨운': 'Disgusted',
    '두려운': 'Fearful',
    '놀란': 'Surprised',
    '동기': 'Drive',
    '욕구': 'Want',
    '주의': 'Attention',
    '흥분': 'Arousal',
}


def translate(feat):
    """Return 'English (Korean)' label for a feature."""
    eng = FEATURE_TRANSLATIONS.get(feat, feat)
    return f"{eng} ({feat})"


def load_norms(norms_file):
    df = pd.read_csv(norms_file)
    word_col = 'Word' if 'Word' in df.columns else df.columns[0]
    metadata_cols = ['No', 'Word', 'word', 'WC', 'N', 'Mean R', 'LEN', 'FREQ',
                     'L10 FREQ', 'Orth', 'Orth_F', 'N1_F', 'N2_F', 'N3_F',
                     'IMG', 'Unnamed: 70', 'Unnamed: 80', 'Type',
                     'Super Category', 'Category', 'Kmeans28 Category']
    feature_cols = [c for c in df.columns if c not in metadata_cols]
    feature_cols = [c for c in feature_cols if df[c].dtype in ['float64', 'int64', 'float32']]
    return df, word_col, feature_cols


def load_embeddings(embeddings_file):
    word_vecs = {}
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue
            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]])
            word_vecs[word] = vec
    return word_vecs


def compute_feature_axis(words, ratings, word_vecs):
    diff_vecs = []
    for i, j in combinations(range(len(words)), 2):
        w1, w2 = words[i], words[j]
        r1, r2 = ratings[i], ratings[j]
        if w1 not in word_vecs or w2 not in word_vecs:
            continue
        if r1 > r2:
            diff_vecs.append(word_vecs[w1] - word_vecs[w2])
        elif r2 > r1:
            diff_vecs.append(word_vecs[w2] - word_vecs[w1])
    if len(diff_vecs) == 0:
        return None
    axis = np.mean(diff_vecs, axis=0)
    norm = np.linalg.norm(axis)
    if norm > 0:
        axis = axis / norm
    return axis


print("Loading norms and embeddings...")
df, word_col, feature_cols = load_norms(NORMS_FILE)
word_vecs = load_embeddings(EMBEDDINGS_FILE)

all_words = [w for w in df[word_col].tolist() if w in word_vecs]
df_filtered = df[df[word_col].isin(all_words)].reset_index(drop=True)

print(f"Learning {len(feature_cols)} feature axes from {len(all_words)} words...")
feature_axes = {}
for feat in feature_cols:
    ratings = df_filtered[feat].tolist()
    words = df_filtered[word_col].tolist()
    axis = compute_feature_axis(words, ratings, word_vecs)
    if axis is not None:
        feature_axes[feat] = axis

print(f"Learned {len(feature_axes)} feature axes")


# ============================================================
# Set up mBERT
# ============================================================

print("Loading mBERT...")
model_name = "bert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
bert_model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
bert_model.eval()
device = torch.device('cpu')


def get_contextual_embedding(word, sentence, layer=8):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs, output_hidden_states=True)
    layer_output = outputs.hidden_states[layer]

    word_tokens = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sentence)
    pre_tokenized_words = [w[0] for w in word_tokens]

    target_idx = None
    for i, w in enumerate(pre_tokenized_words):
        if w == word:
            target_idx = i
            break
    if target_idx is None:
        for i, w in enumerate(pre_tokenized_words):
            if w.startswith(word) or word in w:
                target_idx = i
                break
    if target_idx is None:
        print(f"  WARNING: '{word}' not found in: {pre_tokenized_words}")
        return None

    token_span = inputs.word_to_tokens(target_idx)
    if token_span is None:
        return None
    word_start, word_end = token_span
    embedding = layer_output[0, word_start:word_end, :].mean(dim=0).cpu().numpy()
    return embedding


def project_onto_features(embedding):
    projections = {}
    for feat, axis in feature_axes.items():
        projections[feat] = np.dot(embedding, axis)
    return projections


# ============================================================
# Define construal experiment
# ============================================================

experiments = [
    {
        'verb': '깨다 (break)',
        'conditions': [
            ('Transitive (he broke the glass)', '그가 유리를 깨었다', '깨었다'),
            ('Passive (the glass broke)', '유리가 깨졌다', '깨졌다'),
            ('Causative (he shattered)', '그가 유리를 깨뜨렸다', '깨뜨렸다'),
        ]
    },
    {
        'verb': '열다 (open)',
        'conditions': [
            ('Transitive (she opened the door)', '그녀가 문을 열었다', '열었다'),
            ('Passive (the door opened)', '문이 열렸다', '열렸다'),
        ]
    },
    {
        'verb': '녹다 (melt)',
        'conditions': [
            ('Causative (he melted the ice)', '그가 얼음을 녹였다', '녹였다'),
            ('Intransitive (the ice melted)', '얼음이 녹았다', '녹았다'),
        ]
    },
    {
        'verb': '끓다 (boil)',
        'conditions': [
            ('Transitive (she boiled water)', '그녀가 물을 끓였다', '끓였다'),
            ('Intransitive (the water boiled)', '물이 끓었다', '끓었다'),
        ]
    },
    {
        'verb': '움직이다 (move)',
        'conditions': [
            ('Transitive (he moved the box)', '그가 상자를 움직였다', '움직였다'),
            ('Intransitive (the box moved)', '상자가 움직였다', '움직였다'),
        ]
    },
]


# ============================================================
# Run experiment
# ============================================================

print("\n" + "=" * 80)
print("  CONSTRUAL EXPERIMENT: Argument Structure in Korean")
print("=" * 80)

for exp in experiments:
    print(f"\n{'━' * 80}")
    print(f"  VERB: {exp['verb']}")
    print(f"{'━' * 80}")

    condition_projections = {}

    for cond_name, sentence, target in exp['conditions']:
        print(f"\n  [{cond_name}]")
        print(f"  Sentence: {sentence}")
        print(f"  Target:   {target}")

        emb = get_contextual_embedding(target, sentence)
        if emb is None:
            print("  ERROR: Could not extract embedding")
            continue

        proj = project_onto_features(emb)
        condition_projections[cond_name] = proj

        # Print all feature projections with translations
        print(f"  {'Feature':<30s}  {'Projection':>10s}")
        print(f"  {'─' * 42}")
        for feat in feature_cols:
            if feat in proj:
                label = translate(feat)
                print(f"  {label:<30s}  {proj[feat]:>10.4f}")

    # Compare first two conditions
    if len(condition_projections) >= 2:
        names = list(condition_projections.keys())
        p1 = condition_projections[names[0]]
        p2 = condition_projections[names[1]]

        # Shorten names for display
        n1_short = names[0][:30]
        n2_short = names[1][:30]

        print(f"\n  {'═' * 78}╗")
        print(f"  ║  COMPARISON: {n1_short} vs {n2_short}")
        print(f"  {'═' * 78}╝")

        diffs = {}
        for feat in feature_cols:
            if feat in p1 and feat in p2:
                diffs[feat] = p1[feat] - p2[feat]

        sorted_diffs = sorted(diffs.items(), key=lambda x: abs(x[1]), reverse=True)

        print(f"\n  {'Feature':<30s}  {'Cond1':>8s}  {'Cond2':>8s}  {'Diff':>8s}  Direction")
        print(f"  {'─' * 78}")
        for feat, diff in sorted_diffs:
            label = translate(feat)
            direction = f"← {n1_short[:20]}" if diff > 0 else f"→ {n2_short[:20]}"
            print(f"  {label:<30s}  {p1[feat]:>8.3f}  {p2[feat]:>8.3f}  {diff:>+8.3f}  {direction}")

        # Cosine similarity
        v1 = np.array([p1[f] for f in feature_cols if f in p1])
        v2 = np.array([p2[f] for f in feature_cols if f in p2])
        cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        print(f"\n  Cosine similarity of feature profiles: {cos_sim:.4f}")

        # Summary: top 5 shifts in each direction
        print(f"\n  TOP 5 features higher in [{n1_short}]:")
        for feat, diff in sorted_diffs:
            if diff > 0:
                label = translate(feat)
                print(f"    {label:<30s}  {diff:>+8.3f}")
            if sum(1 for _, d in sorted_diffs if d > 0 and abs(d) >= abs(diff)) >= 5 and diff > 0:
                break

        count = 0
        print(f"\n  TOP 5 features higher in [{n2_short}]:")
        for feat, diff in sorted_diffs:
            if diff < 0:
                label = translate(feat)
                print(f"    {label:<30s}  {abs(diff):>+8.3f}")
                count += 1
                if count >= 5:
                    break

print("\n" + "=" * 80)
print("Done!")