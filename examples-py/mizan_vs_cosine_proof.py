"""
Mizan vs Cosine Similarity — Proof & Experiments

Run:
    python mizan_vs_cosine_proof.py

This script demonstrates the key differences between:
- Cosine similarity  (direction-only)
- Mizan similarity   (scale-aware, proportional error)
"""

import numpy as np

# ---------------------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------------------

def cosine_similarity(v1, v2, eps=1e-8):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    num = float(np.dot(v1, v2))
    den = float(np.linalg.norm(v1) * np.linalg.norm(v2) + eps)
    return num / den

def mizan_distance(v1, v2, p=2.0, eps=1e-8):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    diff = np.linalg.norm(v1 - v2)
    num = diff ** p
    den = (np.linalg.norm(v1) ** p + np.linalg.norm(v2) ** p + eps)
    return float(num / den)

def mizan_similarity(v1, v2, p=2.0, eps=1e-8):
    return 1.0 - mizan_distance(v1, v2, p=p, eps=eps)

def print_header(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------
# Test 1 — Scaled vectors: cosine = 1, Mizan < 1 (scale-awareness)
# ---------------------------------------------------------------------
def test_scaled_vectors():
    print_header("Test 1 — Scaling v2 = k * v1")
    v1 = np.array([1.0, 2.0, 3.0])
    ks = [1, 2, 3, 5, 10]

    print(f"{'k':>5} | {'cosine':>10} | {'mizan':>10}")
    print("-" * 35)
    for k in ks:
        v2 = k * v1
        cos = cosine_similarity(v1, v2)
        miz = mizan_similarity(v1, v2)
        print(f"{k:5} | {cos:10.4f} | {miz:10.4f}")


# ---------------------------------------------------------------------
# Test 2 — Outlier noise: Mizan punishes, cosine fails
# ---------------------------------------------------------------------
def test_outlier():
    print_header("Test 2 — Outlier / Noisy dimension")

    v_clean = np.array([10.0, 10.0, 10.0])
    v_noisy = np.array([10.0, 10.0, 1000.0])

    cos = cosine_similarity(v_clean, v_noisy)
    miz = mizan_similarity(v_clean, v_noisy)
    dist = mizan_distance(v_clean, v_noisy)

    print("v_clean:", v_clean)
    print("v_noisy:", v_noisy)
    print(f"\ncosine = {cos:.6f}")
    print(f"mizan_similarity = {miz:.6f}")
    print(f"mizan_distance   = {dist:.6f}")


# ---------------------------------------------------------------------
# Test 3 — Small vs Big scaled vectors
# ---------------------------------------------------------------------
def test_scale_sensitivity():
    print_header("Test 3 — Small-scale vs Big-scale version")

    np.random.seed(42)
    base = np.random.randn(8)
    v_small = 0.1 * base
    v_big = 10 * base

    cos_val = cosine_similarity(v_small, v_big)
    miz_val = mizan_similarity(v_small, v_big)

    print("base vector:", base)
    print("\nv_small:", v_small)
    print("v_big  :", v_big)

    print(f"\ncosine(v_small, v_big) = {cos_val:.6f}")
    print(f"mizan(v_small, v_big)  = {miz_val:.6f}")


# ---------------------------------------------------------------------
# Test 4 — Normalized vectors: why cosine ≈ Mizan for TEXT embeddings
# ---------------------------------------------------------------------
def normalize(v):
    v = np.asarray(v, dtype=float)
    return v / (np.linalg.norm(v) + 1e-8)

def test_normalized_vectors():
    print_header("Test 4 — Normalized vectors")

    a = np.random.randn(8)
    b = a + 0.5 * np.random.randn(8)

    a_n = normalize(a)
    b_n = normalize(b)

    cos_n = cosine_similarity(a_n, b_n)
    miz_n = mizan_similarity(a_n, b_n)

    print("normalized a:", a_n)
    print("normalized b:", b_n)

    print(f"\ncosine(normalized) = {cos_n:.6f}")
    print(f"mizan(normalized)  = {miz_n:.6f}")

    print("\nExplanation:")
    print("- Pretrained text models (MiniLM, BERT, etc.) normalize vectors.")
    print("- When norm ≈ 1 for all vectors → both Mizan and cosine behave similarly.")
    print("- This is EXACT reason your initial test returned similar values.")


# ---------------------------------------------------------------------
# Optional Visualization
# ---------------------------------------------------------------------
def plot_graph():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping plot.")
        return

    print_header("Plot — Cosine vs Mizan when scaling vector")

    v = np.array([1.0, 2.0, 3.0])
    ks = np.linspace(0.1, 5.0, 50)

    cosine_vals = [cosine_similarity(v, k*v) for k in ks]
    mizan_vals = [mizan_similarity(v, k*v) for k in ks]

    plt.figure(figsize=(8, 4))
    plt.plot(ks, cosine_vals, label="cosine")
    plt.plot(ks, mizan_vals, label="mizan")
    plt.xlabel("Scale factor k")
    plt.ylabel("Similarity")
    plt.title("Cosine vs Mizan as one vector is scaled")
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
if __name__ == "__main__":
    test_scaled_vectors()
    test_outlier()
    test_scale_sensitivity()
    test_normalized_vectors()
    plot_graph()
    