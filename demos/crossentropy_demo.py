
"""
Cross-Entropy-Beispiel:

Zielwort „Frankreich“ mit Wahrscheinlichkeit p_Frankreich.
Dann ist L = -log(p_Frankreich).

Wir berechnen L für zwei verschiedene p-Werte.
"""

import math

def cross_entropy_for_target(p_target: float) -> float:
    return -math.log(p_target)

def main():
    p1 = 0.6590
    L1 = cross_entropy_for_target(p1)

    p2 = 0.85
    L2 = cross_entropy_for_target(p2)

    print(f"p_Frankreich = {p1:.4f} -> L = -log(p) = {L1:.3f}")
    print(f"p_Frankreich = {p2:.4f} -> L = -log(p) = {L2:.3f}")
    print("\\nJe größer p_Frankreich, desto kleiner die Cross-Entropy L.")

if __name__ == "__main__":
    main()
