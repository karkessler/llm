
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
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def main():
    tokens = ["Paris", "ist", "die", "Hauptstadt", "von"]
    n_tokens = len(tokens)
    d_model = 4

    np.random.seed(42)
    X = np.random.rand(n_tokens, d_model)

    W_Q = np.eye(d_model)
    W_K = np.eye(d_model)
    W_V = np.eye(d_model)

    Q = X @ W_Q.T
    K = X @ W_K.T
    V = X @ W_V.T

    idx_von = tokens.index("von")
    q_von = Q[idx_von:idx_von+1, :]

    scores = q_von @ K.T
    weights = softmax(scores)
    context_von = weights @ V

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
