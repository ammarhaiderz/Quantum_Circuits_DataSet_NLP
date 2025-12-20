"""
Quick sanity check for the quantum_problem classifier.
- Loads a lightweight SBERT model
- Precomputes label embeddings
- Classifies a couple of sample description sets
- Does NOT write any project data
"""
from __future__ import annotations

import sys

from sentence_transformers import SentenceTransformer, util

from core.quantum_problem_classifier import (
    prepare_label_embeddings,
    classify_quantum_problem,
    apply_classification,
)
from config.quantum_problem_labels import QUANTUM_PROBLEM_LABELS


def main():
    model_name = "all-MiniLM-L6-v2"  # small and fast
    print(f"Loading SBERT model: {model_name}")
    model = SentenceTransformer(model_name)

    print("Encoding label embeddings…")
    label_keys, label_embeds = prepare_label_embeddings(model)
    print(f"Encoded {len(label_keys)} labels")

    samples = [
        [
            "Circuit showing multiple CNOT and RCCX decompositions for removing redundant gates.",
            "Example of gate decomposition and gate cancellation in a Toffoli-style circuit.",
        ],
        [
            "Logical constant-depth CNOT ladder across logical qubits with ancilla feed-forward.",
            "Neutral atom architecture executing constant-depth CDCX ladder." ,
        ],
        [
            "Quantum circuit for preparing a logical encoded [[4,2,2]] state with stabilizer checks.",
        ],
    ]

    for i, descs in enumerate(samples, 1):
        label = classify_quantum_problem(
            model,
            descs,
            label_keys,
            label_embeds,
        )
        print(f"\nSample {i}: {label}")
        print(f"  Text: {' '.join(descs)[:120]}…")

    # Also test apply_classification on a record list
    records = [{"descriptions": samples[0]}, {"descriptions": samples[1]}]
    apply_classification(records, model, label_keys, label_embeds)
    print("\napply_classification results:")
    for r in records:
        print(f"  → {r.get('quantum_problem')}")

    # Optional: show top similarity for first sample
    query_text = " ".join(samples[0])
    q_emb = model.encode(query_text, normalize_embeddings=True)
    scores = util.cos_sim(q_emb, label_embeds)[0]
    best_idx = int(scores.argmax())
    print(f"\nTop score for sample 1: {scores[best_idx]:.3f} ({label_keys[best_idx]})")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
