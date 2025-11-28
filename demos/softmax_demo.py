
"""
Softmax-Beispiel:

Gegeben sei ein Kontextvektor
    h = [0.304, 0.529, 0.603, 0.399]

und ein kleines Vokabular V = {Frankreich, Deutschland, Spanien} mit Zeilen von W_out.

Wir berechnen z = W_out * h und anschließend Softmax(z).
"""

import numpy as np

def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - np.max(z))
    return e / e.sum()

def main():
    h = np.array([0.304, 0.529, 0.603, 0.399])

    W_out = np.array([
        [1.0, 0.5, 1.0, 0.5],  # Frankreich
        [0.0, 1.0, 0.0, 1.0],  # Deutschland
        [1.0, 0.0, 1.0, 0.0],  # Spanien
    ])

    vocab = ["Frankreich", "Deutschland", "Spanien"]

    # Logits z = W_out * h
    z = W_out @ h

    print("Kontextvektor h:", h)
    print("\nLogits z für die Tokens:")
    for token, z_i in zip(vocab, z):
        print(f"  {token:11s}: {z_i:.3f}")

    # Softmax-Wahrscheinlichkeiten
    p = softmax(z)

    print("\nSoftmax-Wahrscheinlichkeiten p:")
    for token, p_i in zip(vocab, p):
        print(f"  P({token:11s}) = {p_i:.3f}")

    print("\nSumme der Wahrscheinlichkeiten:", p.sum())

if __name__ == "__main__":
    main()
