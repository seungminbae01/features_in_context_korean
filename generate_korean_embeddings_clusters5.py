"""
generate_korean_embeddings_clusters5.py

Same as the corpus version but does k-means (K=5) instead of averaging.
Reuses the same sentence collection and mBERT extraction,
then clusters into 5 centroids per word.

Output format: 5 lines per word (one per centroid), as expected by
read_multiprototype_embeddings() with clusters=5.

Usage:
    python generate_korean_embeddings_clusters5.py \
        --norms_file ./data/external/binder_word_ratings/korean_binder_norms.csv \
        --corpus_dir ./data/korean_corpus/ \
        --output_file ./data/processed/multipro_embeddings/layer8clusters5.txt \
        --layer 8 \
        --max_sentences 200
"""

import argparse
import os
import glob
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import pandas as pd
from collections import defaultdict


def load_word_list(norms_file, word_column=None):
    df = pd.read_csv(norms_file)
    if word_column is None:
        candidates = ['Word', 'word', 'concept', 'Concept']
        for col in candidates:
            if col in df.columns:
                word_column = col
                break
        if word_column is None:
            word_column = df.columns[0]
    words = df[word_column].tolist()
    print(f"Loaded {len(words)} words from {norms_file}")
    return words


def collect_all_sentences(words, corpus_dir, max_sentences=200):
    word_set = set(words)
    word_sentences = defaultdict(list)
    words_done = set()

    corpus_files = sorted(glob.glob(os.path.join(corpus_dir, '*.txt')))
    print(f"Found {len(corpus_files)} corpus files")

    for fi, filepath in enumerate(corpus_files):
        if len(words_done) == len(word_set):
            print(f"All words have {max_sentences} sentences. Stopping early at file {fi}.")
            break

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                sent = line.strip()
                if len(sent) < 10 or len(sent) > 500:
                    continue

                for word in word_set - words_done:
                    if word in sent:
                        word_sentences[word].append(sent)
                        if len(word_sentences[word]) >= max_sentences:
                            words_done.add(word)

        if (fi + 1) % 10 == 0:
            found = sum(1 for w in words if len(word_sentences[w]) > 0)
            print(f"  Processed {fi+1}/{len(corpus_files)} files. "
                  f"Words with sentences: {found}/{len(words)}")

    counts = [len(word_sentences[w]) for w in words]
    found = sum(1 for c in counts if c > 0)
    print(f"\nSentence collection done:")
    print(f"  Words with sentences: {found}/{len(words)}")
    print(f"  Average sentences per word: {np.mean(counts):.1f}")
    print(f"  Words with 0 sentences: {len(words) - found}")

    missing = [w for w in words if len(word_sentences[w]) == 0]
    if missing:
        print(f"  Missing words: {missing[:20]}{'...' if len(missing) > 20 else ''}")

    return word_sentences


class MBERTEmbedder:
    def __init__(self, device=None):
        model_name = "bert-base-multilingual-cased"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()
        print(f"Loaded {model_name} on {self.device}")

    def get_word_embedding(self, word, sentence, layer=8):
        try:
            inputs = self.tokenizer(
                sentence,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

            layer_output = outputs.hidden_states[layer]

            word_tokens = self.tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(sentence)
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
            word_embedding = layer_output[0, word_start:word_end, :].mean(dim=0)

            return word_embedding.cpu().numpy()

        except Exception:
            return None


def cluster_embeddings(embeddings, k=5):
    """K-means cluster embeddings into k centroids."""
    n = len(embeddings)
    X = np.array(embeddings)

    if n < k:
        # Pad by repeating existing embeddings
        while len(embeddings) < k:
            embeddings.append(embeddings[len(embeddings) % n])
        X = np.array(embeddings)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    return kmeans.cluster_centers_  # shape: (k, 768)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--norms_file', type=str, required=True)
    parser.add_argument('--corpus_dir', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--layer', type=int, default=8)
    parser.add_argument('--clusters', type=int, default=5)
    parser.add_argument('--max_sentences', type=int, default=200)
    parser.add_argument('--word_column', type=str, default=None)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Step 1: Load words
    words = load_word_list(args.norms_file, args.word_column)

    # Step 2: Collect sentences (single pass through corpus)
    word_sentences = collect_all_sentences(words, args.corpus_dir, args.max_sentences)

    # Step 3: Initialize mBERT
    embedder = MBERTEmbedder()

    # Step 4: Extract embeddings and cluster
    skipped = []

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for i, word in enumerate(words):
            sentences = word_sentences[word]

            if len(sentences) == 0:
                print(f"[{i+1}/{len(words)}] {word}: SKIPPED (no sentences)")
                skipped.append(word)
                continue

            embeddings = []
            for sent in sentences:
                emb = embedder.get_word_embedding(word, sent, layer=args.layer)
                if emb is not None:
                    embeddings.append(emb)

            if len(embeddings) == 0:
                print(f"[{i+1}/{len(words)}] {word}: SKIPPED (no valid embeddings)")
                skipped.append(word)
                continue

            # K-means cluster into k centroids
            centroids = cluster_embeddings(embeddings, k=args.clusters)

            # Write k lines per word
            for centroid in centroids:
                number_str = np.array2string(
                    centroid,
                    precision=8,
                    max_line_width=100000,
                    separator=' '
                )
                number_str = number_str[1:-1].replace("\n", "")
                f.write(f"{word} {number_str}\n")

            if (i + 1) % 20 == 0 or (i + 1) == len(words):
                print(f"[{i+1}/{len(words)}] {word}: {len(embeddings)} embeddings -> {args.clusters} centroids")

    print(f"\nDone! Wrote {len(words) - len(skipped)} words ({(len(words) - len(skipped)) * args.clusters} lines) to {args.output_file}")
    if skipped:
        print(f"Skipped {len(skipped)} words: {skipped}")


if __name__ == '__main__':
    main()