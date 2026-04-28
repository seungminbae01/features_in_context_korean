"""
ud_construal_scaled.py

Scaled construal experiment using Universal Dependencies treebanks.
1. Parses UD Korean treebanks to find verbs in transitive vs intransitive
2. Extracts mBERT embeddings for each occurrence
3. Projects onto learned feature axes
4. Runs statistical tests across all verbs

Usage:
    python ud_construal_scaled.py
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from itertools import combinations
from scipy.stats import pearsonr, ttest_rel, wilcoxon
from collections import defaultdict

NORMS_FILE = './data/external/binder_word_ratings/korean_binder_norms.csv'
EMBEDDINGS_FILE = './data/processed/multipro_embeddings/layer8clusters1_corpus.txt'
UD_FILES = [
    './data/ud/ko_gsd-ud-train.conllu',
    './data/ud/ko_gsd-ud-dev.conllu',
    './data/ud/ko_gsd-ud-test.conllu',
    './data/ud/ko_kaist-ud-train.conllu',
    './data/ud/ko_kaist-ud-dev.conllu',
    './data/ud/ko_kaist-ud-test.conllu',
    './data/ud/ko_pud-ud-test.conllu',
]

# Korean -> English feature translations
FEATURE_TRANSLATIONS = {
    '시각': 'Vision', '밝음': 'Bright', '어두움': 'Dark', '색깔': 'Color',
    '패턴': 'Pattern', '크기': 'Size', '작음': 'Small', '움직임': 'Motion',
    '생체 움직임': 'Biomotion', '빠름': 'Fast', '느림': 'Slow', '모양': 'Shape',
    '얼굴': 'Face', '몸': 'Body', '촉각': 'Touch', '온도': 'Temperature',
    '질감': 'Texture', '무게': 'Weight', '통증': 'Pain', '청각': 'Audition',
    '큰 소리': 'Loud', '낮은': 'Low', '높음': 'High', '소리': 'Sound',
    '음악': 'Music', '말': 'Speech', '미각': 'Taste', '후각': 'Smell',
    '머리': 'Head', '상지': 'Upper limb', '하지': 'Lower limb',
    '랜드마크': 'Landmark', '경로': 'Path', '장면': 'Scene', '가까운': 'Near',
    '쪽으로': 'Toward', '멀리': 'Away', '숫자': 'Number', '시간': 'Time',
    '지속시간': 'Duration', '긴': 'Long', '짧은': 'Short',
    '결과적인': 'Consequential', '사회적인': 'Social', '인간': 'Human',
    '의사소통': 'Communication', '자기': 'Self', '인지': 'Cognition',
    '이익': 'Benefit', '해': 'Harm', '쾌적한': 'Pleasant',
    '불쾌한': 'Unpleasant', '행복한': 'Happy', '슬픈': 'Sad',
    '화난': 'Angry', '역겨운': 'Disgusted', '두려운': 'Fearful',
    '놀란': 'Surprised', '동기': 'Drive', '욕구': 'Want',
    '주의': 'Attention', '흥분': 'Arousal',
}

def translate(feat):
    eng = FEATURE_TRANSLATIONS.get(feat, feat)
    return f"{eng} ({feat})"


# ============================================================
# 1. Parse UD CoNLL-U files
# ============================================================

def parse_conllu(filepath):
    """Parse a CoNLL-U file, return list of sentences.
    Each sentence is a dict with 'text' and 'tokens' (list of token dicts).
    """
    sentences = []
    current_tokens = []
    current_text = None

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('# text ='):
                current_text = line[len('# text ='):].strip()
            elif line == '':
                if current_tokens:
                    sentences.append({
                        'text': current_text,
                        'tokens': current_tokens,
                    })
                current_tokens = []
                current_text = None
            elif not line.startswith('#'):
                parts = line.split('\t')
                if len(parts) >= 10:
                    # Skip multi-word tokens (e.g., "1-2")
                    if '-' in parts[0] or '.' in parts[0]:
                        continue
                    token = {
                        'id': int(parts[0]),
                        'form': parts[1],
                        'lemma': parts[2],
                        'upos': parts[3],
                        'xpos': parts[4],
                        'feats': parts[5],
                        'head': int(parts[6]) if parts[6] != '_' else 0,
                        'deprel': parts[7],
                        'deps': parts[8],
                        'misc': parts[9],
                    }
                    current_tokens.append(token)

    # Don't forget last sentence
    if current_tokens:
        sentences.append({
            'text': current_text,
            'tokens': current_tokens,
        })

    return sentences


def find_verb_constructions(sentences):
    """Find verbs in transitive (has nsubj + obj) vs intransitive (has nsubj, no obj).
    Returns dict: {lemma: {'transitive': [(sentence_text, verb_form), ...],
                           'intransitive': [(sentence_text, verb_form), ...]}}
    """
    verb_data = defaultdict(lambda: {'transitive': [], 'intransitive': []})

    for sent in sentences:
        if sent['text'] is None:
            continue

        tokens = sent['tokens']

        # Find all verbs
        for tok in tokens:
            if tok['upos'] != 'VERB':
                continue

            verb_id = tok['id']
            lemma = tok['lemma']
            form = tok['form']

            # Find dependents of this verb
            deps = [t for t in tokens if t['head'] == verb_id]
            dep_rels = [t['deprel'] for t in deps]

            has_nsubj = 'nsubj' in dep_rels
            has_obj = 'obj' in dep_rels

            if has_nsubj and has_obj:
                verb_data[lemma]['transitive'].append((sent['text'], form))
            elif has_nsubj and not has_obj:
                verb_data[lemma]['intransitive'].append((sent['text'], form))

    return verb_data


# ============================================================
# 2. Load norms and learn feature axes
# ============================================================

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


# ============================================================
# 3. mBERT embedding extraction
# ============================================================

def setup_mbert():
    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()
    device = torch.device('cpu')
    return tokenizer, model, device


def get_contextual_embedding(tokenizer, model, device, word, sentence, layer=8):
    try:
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
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
            return None

        token_span = inputs.word_to_tokens(target_idx)
        if token_span is None:
            return None
        word_start, word_end = token_span
        embedding = layer_output[0, word_start:word_end, :].mean(dim=0).cpu().numpy()
        return embedding
    except Exception:
        return None


# ============================================================
# 4. Main experiment
# ============================================================

def main():
    # --- Load UD data ---
    print("=" * 80)
    print("  SCALED CONSTRUAL EXPERIMENT")
    print("  Transitive vs Intransitive in Korean (from UD treebanks)")
    print("=" * 80)

    print("\n[1] Parsing UD treebanks...")
    all_sentences = []
    for f in UD_FILES:
        if os.path.exists(f):
            sents = parse_conllu(f)
            all_sentences.extend(sents)
            print(f"  {f}: {len(sents)} sentences")
        else:
            print(f"  WARNING: {f} not found")

    print(f"  Total: {len(all_sentences)} sentences")

    print("\n[2] Finding verb constructions...")
    verb_data = find_verb_constructions(all_sentences)

    # Filter to verbs that appear in BOTH constructions with enough examples
    MIN_EXAMPLES = 2
    alternating_verbs = {
        lemma: data for lemma, data in verb_data.items()
        if len(data['transitive']) >= MIN_EXAMPLES and len(data['intransitive']) >= MIN_EXAMPLES
    }

    print(f"  Verbs found in both constructions (min {MIN_EXAMPLES} each): {len(alternating_verbs)}")
    for lemma, data in sorted(alternating_verbs.items(), key=lambda x: len(x[1]['transitive']) + len(x[1]['intransitive']), reverse=True)[:20]:
        print(f"    {lemma:<15s}  trans: {len(data['transitive']):>3d}  intrans: {len(data['intransitive']):>3d}")

    if len(alternating_verbs) == 0:
        print("  No verbs found with enough examples. Try lowering MIN_EXAMPLES.")
        return

    # --- Load norms and learn axes ---
    print("\n[3] Learning feature axes...")
    df, word_col, feature_cols = load_norms(NORMS_FILE)
    word_vecs = load_embeddings(EMBEDDINGS_FILE)

    all_words = [w for w in df[word_col].tolist() if w in word_vecs]
    df_filtered = df[df[word_col].isin(all_words)].reset_index(drop=True)

    feature_axes = {}
    for feat in feature_cols:
        ratings = df_filtered[feat].tolist()
        words = df_filtered[word_col].tolist()
        axis = compute_feature_axis(words, ratings, word_vecs)
        if axis is not None:
            feature_axes[feat] = axis
    print(f"  Learned {len(feature_axes)} feature axes")

    # --- Extract embeddings and project ---
    print("\n[4] Loading mBERT...")
    tokenizer, model, device = setup_mbert()

    MAX_SENTENCES = 20  # Max sentences per verb per condition

    print("\n[5] Extracting embeddings and projecting onto feature axes...")
    verb_results = {}

    for vi, (lemma, data) in enumerate(alternating_verbs.items()):
        trans_projs = []
        intrans_projs = []

        # Transitive sentences
        for sent_text, verb_form in data['transitive'][:MAX_SENTENCES]:
            emb = get_contextual_embedding(tokenizer, model, device, verb_form, sent_text)
            if emb is not None:
                proj = {feat: np.dot(emb, axis) for feat, axis in feature_axes.items()}
                trans_projs.append(proj)

        # Intransitive sentences
        for sent_text, verb_form in data['intransitive'][:MAX_SENTENCES]:
            emb = get_contextual_embedding(tokenizer, model, device, verb_form, sent_text)
            if emb is not None:
                proj = {feat: np.dot(emb, axis) for feat, axis in feature_axes.items()}
                intrans_projs.append(proj)

        if len(trans_projs) >= 2 and len(intrans_projs) >= 2:
            # Average projections per condition
            avg_trans = {feat: np.mean([p[feat] for p in trans_projs]) for feat in feature_cols if feat in feature_axes}
            avg_intrans = {feat: np.mean([p[feat] for p in intrans_projs]) for feat in feature_cols if feat in feature_axes}
            verb_results[lemma] = {
                'transitive': avg_trans,
                'intransitive': avg_intrans,
                'n_trans': len(trans_projs),
                'n_intrans': len(intrans_projs),
            }

        if (vi + 1) % 5 == 0 or (vi + 1) == len(alternating_verbs):
            print(f"  Processed {vi+1}/{len(alternating_verbs)} verbs...")

    print(f"\n  Verbs with valid embeddings in both conditions: {len(verb_results)}")

    if len(verb_results) < 3:
        print("  Not enough verbs for statistical testing.")
        return

    # --- Statistical analysis ---
    print("\n" + "=" * 80)
    print("  RESULTS")
    print("=" * 80)

    # For each feature, collect (transitive_mean, intransitive_mean) across verbs
    print(f"\n  Per-verb differences (transitive - intransitive), averaged across verbs:")
    print(f"\n  {'Feature':<30s}  {'Mean diff':>10s}  {'Std':>8s}  {'t-stat':>8s}  {'p-value':>10s}  {'Direction'}")
    print(f"  {'─' * 90}")

    feature_results = []

    for feat in feature_cols:
        if feat not in feature_axes:
            continue

        trans_vals = []
        intrans_vals = []

        for lemma, res in verb_results.items():
            if feat in res['transitive'] and feat in res['intransitive']:
                trans_vals.append(res['transitive'][feat])
                intrans_vals.append(res['intransitive'][feat])

        if len(trans_vals) < 3:
            continue

        trans_arr = np.array(trans_vals)
        intrans_arr = np.array(intrans_vals)
        diffs = trans_arr - intrans_arr

        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs, ddof=1)

        # Paired t-test
        t_stat, p_val = ttest_rel(trans_arr, intrans_arr)

        direction = "Transitive >" if mean_diff > 0 else "Intransitive >"
        sig = "*" if p_val < 0.05 else ""
        sig += "*" if p_val < 0.01 else ""
        sig += "*" if p_val < 0.001 else ""

        label = translate(feat)
        print(f"  {label:<30s}  {mean_diff:>+10.4f}  {std_diff:>8.4f}  {t_stat:>8.3f}  {p_val:>10.4f}  {direction} {sig}")

        feature_results.append((feat, mean_diff, std_diff, t_stat, p_val))

    # --- Summary ---
    print(f"\n{'=' * 80}")
    print("  SUMMARY")
    print(f"{'=' * 80}")
    print(f"\n  Total verbs analyzed: {len(verb_results)}")
    for lemma, res in verb_results.items():
        print(f"    {lemma:<15s}  trans: {res['n_trans']:>2d} sentences  intrans: {res['n_intrans']:>2d} sentences")

    sig_features = [(f, d, p) for f, d, _, _, p in feature_results if p < 0.05]
    print(f"\n  Significant features (p < 0.05): {len(sig_features)}/{len(feature_results)}")

    if sig_features:
        sig_trans = sorted([(f, d, p) for f, d, p in sig_features if d > 0], key=lambda x: x[1], reverse=True)
        sig_intrans = sorted([(f, d, p) for f, d, p in sig_features if d < 0], key=lambda x: x[1])

        if sig_trans:
            print(f"\n  Features significantly HIGHER in TRANSITIVE:")
            for feat, diff, p in sig_trans:
                label = translate(feat)
                print(f"    {label:<30s}  diff = {diff:>+.4f}  p = {p:.4f}")

        if sig_intrans:
            print(f"\n  Features significantly HIGHER in INTRANSITIVE:")
            for feat, diff, p in sig_intrans:
                label = translate(feat)
                print(f"    {label:<30s}  diff = {abs(diff):>+.4f}  p = {p:.4f}")

    print(f"\n{'=' * 80}")
    print("Done!")


if __name__ == '__main__':
    main()
