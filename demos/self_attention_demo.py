
"""
Self-Attention-Minibeispiel für das Token „von“ im Satz

„Paris ist die Hauptstadt von Frankreich“

Wir definieren:
- einfache Embeddings X
- Gewichtsmatrizen W_Q, W_K, W_V (hier Einheitsmatrizen)
- berechnen Q, K, V
- berechnen Attention-Scores und den neuen kontextualisierten Vektor für „von“.
"""

import numpy as np

def softmax(x: np.ndarray) -> np.ndarray:
    """
    Softmax-Funktion entlang der letzten Achse.
    """
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def main():
    tokens = ["Paris", "ist", "die", "Hauptstadt", "von"]
    n_tokens = len(tokens)
    d_model = 4  # kleine Vektordimension

    # Beispiel-Embeddings (x_i) für jedes Token (reproduzierbar mit Seed)
    np.random.seed(42)
    X = np.random.rand(n_tokens, d_model)

    # Gewichtsmatrizen W_Q, W_K, W_V als Einheitsmatrizen
    W_Q = np.eye(d_model)
    W_K = np.eye(d_model)
    W_V = np.eye(d_model)

    # Berechnung von Q, K, V
    Q = X @ W_Q.T  # (5,4)
    K = X @ W_K.T  # (5,4)
    V = X @ W_V.T  # (5,4)

    # Fokus auf Token „von“
    idx_von = tokens.index("von")
    q_von = Q[idx_von:idx_von+1, :]  # Form (1,4)

    # Attention-Scores: Skalarprodukte q_von • k_j
    scores = q_von @ K.T  # (1,5)
    weights = softmax(scores)  # (1,5)

    # Neuer kontextualisierter Vektor für „von“
    context_von = weights @ V  # (1,4)

    print("Tokens:", tokens)
    print("\nEmbeddings X:")
    print(X)
    print("\nQuery von 'von':")
    print(q_von)
    print("\nAttention-Scores (vor Softmax) für 'von' auf alle Tokens:")
    print(scores)
    print("\nAttention-Gewichte (nach Softmax) für 'von':")
    for t, w in zip(tokens, weights.flatten()):
        print(f"  Attention auf {t:10s}: {w:.3f}")
    print("\nNeuer kontextualisierter Vektor für 'von':")
    print(context_von)

if __name__ == "__main__":
    main()
