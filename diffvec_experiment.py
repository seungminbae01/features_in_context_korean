"""
diffvec_experiment.py

Learns a direction vector for each Binder feature using pairwise
differences of word embeddings, then evaluates by projecting test
words onto that direction and correlating with human ratings.

Usage:
    python diffvec_experiment.py \
        --norms_file ./data/external/binder_word_ratings/korean_binder_norms.csv \
        --embeddings_file ./data/processed/multipro_embeddings/layer8clusters1_corpus.txt \
        --clusters 1
"""

import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from itertools import combinations


def load_norms(norms_file):
    """Load Korean Binder norms, return DataFrame with Word + feature columns."""
    df = pd.read_csv(norms_file)

    # Identify word column
    word_col = None
    for col in ['Word', 'word']:
        if col in df.columns:
            word_col = col
            break
    if word_col is None:
        word_col = df.columns[0]

    # Identify metadata columns to exclude
    metadata_cols = ['No', 'Word', 'word', 'WC', 'N', 'Mean R', 'LEN', 'FREQ',
                     'L10 FREQ', 'Orth', 'Orth_F', 'N1_F', 'N2_F', 'N3_F',
                     'IMG', 'Unnamed: 70', 'Unnamed: 80', 'Type',
                     'Super Category', 'Category', 'Kmeans28 Category']

    feature_cols = [c for c in df.columns if c not in metadata_cols]
    # Keep only numeric feature columns
    feature_cols = [c for c in feature_cols if df[c].dtype in ['float64', 'int64', 'float32']]

    print(f"Loaded {len(df)} words, {len(feature_cols)} features")
    print(f"First 5 features: {feature_cols[:5]}")
    return df, word_col, feature_cols


def load_embeddings(embeddings_file, clusters=1):
    """Load embeddings file. For clusters>1, average the cluster centroids."""
    word_vecs = {}
    current_word = None
    current_vecs = []

    with open(embeddings_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 10:
                continue

            word = parts[0]
            vec = np.array([float(x) for x in parts[1:]])

            if clusters == 1:
                word_vecs[word] = vec
            else:
                if word != current_word:
                    if current_word is not None and len(current_vecs) > 0:
                        word_vecs[current_word] = np.mean(current_vecs, axis=0)
                    current_word = word
                    current_vecs = [vec]
                else:
                    current_vecs.append(vec)

        # Don't forget the last word
        if clusters > 1 and current_word is not None and len(current_vecs) > 0:
            word_vecs[current_word] = np.mean(current_vecs, axis=0)

    print(f"Loaded {len(word_vecs)} word embeddings of dim {len(next(iter(word_vecs.values())))}")
    return word_vecs


def train_test_split(words, seed=0, train_ratio=0.8):
    """Split words into 80/20 train/test."""
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(words))
    split = int(len(words) * train_ratio)
    train_idx = indices[:split]
    test_idx = indices[split:]
    return train_idx, test_idx


def compute_feature_axis(train_words, train_ratings, word_vecs):
    """
    For a single feature, compute the average difference vector
    over all pairs where w1 is rated higher than w2.

    Returns the normalized axis vector.
    """
    diff_vecs = []

    for i, j in combinations(range(len(train_words)), 2):
        w1, w2 = train_words[i], train_words[j]
        r1, r2 = train_ratings[i], train_ratings[j]

        if w1 not in word_vecs or w2 not in word_vecs:
            continue

        if r1 > r2:
            diff_vecs.append(word_vecs[w1] - word_vecs[w2])
        elif r2 > r1:
            diff_vecs.append(word_vecs[w2] - word_vecs[w1])
        # Skip ties

    if len(diff_vecs) == 0:
        return None

    axis = np.mean(diff_vecs, axis=0)

    # Normalize
    norm = np.linalg.norm(axis)
    if norm > 0:
        axis = axis / norm

    return axis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--norms_file', type=str, required=True)
    parser.add_argument('--embeddings_file', type=str, required=True)
    parser.add_argument('--clusters', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    # Load data
    df, word_col, feature_cols = load_norms(args.norms_file)
    word_vecs = load_embeddings(args.embeddings_file, args.clusters)

    # Filter to words that have embeddings
    words_with_embs = [w for w in df[word_col].tolist() if w in word_vecs]
    df_filtered = df[df[word_col].isin(words_with_embs)].reset_index(drop=True)
    print(f"Words with both norms and embeddings: {len(df_filtered)}")

    # Train/test split
    all_words = df_filtered[word_col].tolist()
    train_idx, test_idx = train_test_split(all_words, seed=args.seed)
    train_words = [all_words[i] for i in train_idx]
    test_words = [all_words[i] for i in test_idx]
    print(f"Train: {len(train_words)}, Test: {len(test_words)}")

    # Run for each feature
    print(f"\n{'Feature':<20s}  {'Pearson r':>10s}  {'p-value':>10s}  {'Axis pairs':>12s}")
    print("─" * 60)

    results = []

    for feat in feature_cols:
        train_ratings = [df_filtered.iloc[i][feat] for i in train_idx]
        test_ratings = [df_filtered.iloc[i][feat] for i in test_idx]

        # Skip features with no variance in test
        if np.std(test_ratings) == 0:
            continue

        # Compute feature axis from training pairs
        axis = compute_feature_axis(train_words, train_ratings, word_vecs)

        if axis is None:
            continue

        # Count training pairs used
        n_pairs = len(train_words) * (len(train_words) - 1) // 2

        # Predict: project test word embeddings onto axis
        predicted = []
        actual = []
        for i, idx in enumerate(test_idx):
            w = all_words[idx]
            if w in word_vecs:
                proj = np.dot(word_vecs[w], axis)
                predicted.append(proj)
                actual.append(test_ratings[i])

        if len(predicted) < 3:
            continue

        # Pearson correlation
        r, p = pearsonr(predicted, actual)
        results.append((feat, r, p, n_pairs))
        print(f"{feat:<20s}  {r:>10.4f}  {p:>10.4f}  {n_pairs:>12d}")

    # Summary
    print(f"\n{'=' * 60}")
    correlations = [r for _, r, _, _ in results]
    print(f"Average Pearson r across {len(results)} features: {np.mean(correlations):.4f}")
    print(f"Median Pearson r: {np.median(correlations):.4f}")
    print(f"Features with r > 0.3: {sum(1 for r in correlations if r > 0.3)}/{len(results)}")
    print(f"Features with r > 0.2: {sum(1 for r in correlations if r > 0.2)}/{len(results)}")
    print(f"Features with r > 0.1: {sum(1 for r in correlations if r > 0.1)}/{len(results)}")

    # Top 10 best predicted features
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)
    print(f"\nTop 10 best predicted features:")
    for feat, r, p, n in results_sorted[:10]:
        print(f"  {feat:<20s}  r = {r:.4f}  (p = {p:.4f})")

    # Bottom 5 worst predicted features
    print(f"\nBottom 5 worst predicted features:")
    for feat, r, p, n in results_sorted[-5:]:
        print(f"  {feat:<20s}  r = {r:.4f}  (p = {p:.4f})")


if __name__ == '__main__':
    main()