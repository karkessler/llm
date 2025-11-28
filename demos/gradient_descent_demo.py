
"""
Einfaches Beispiel für Gradientenabstieg:

Wir nehmen an, dass die Ableitung des Loss nach einem Gewicht w gegeben ist durch
    dL/dw = (p - y) * x

und führen mehrere Update-Schritte aus:

    w_neu = w_alt - eta * dL/dw
"""

def gradient_step(w, x, p, y, eta):
    """
    Führt einen einzelnen Gradientenabstiegsschritt durch.
    """
    grad = (p - y) * x
    w_new = w - eta * grad
    return w_new, grad

def main():
    w = 0.0      # Startgewicht
    x = 1.0      # Eingabe
    y = 1.0      # Ziel
    eta = 0.1    # Lernrate

    print("Gradientenabstieg für ein einzelnes Gewicht w")
    print("Start: w =", w)
    print("Ziel: y =", y)
    print("Eingabe x =", x)
    print("Lernrate eta =", eta)
    print()

    # Mehrere Schritte mit unterschiedlichen p-Werten
    p_values = [0.2, 0.4, 0.6, 0.8, 0.9]

    for step, p in enumerate(p_values, start=1):
        w, grad = gradient_step(w, x, p, y, eta)
        print(f"Schritt {step}: p = {p:.2f}, Grad = (p - y)*x = {grad:.3f}, neues w = {w:.3f}")

    print("\nInterpretation:")
    print("- Solange p < y ist, ist (p - y) negativ -> Grad negativ -> w wird größer.")
    print("- Wenn p sich y nähert, werden die Schritte kleiner.")
    print("- Ist p > y, dreht sich die Richtung um.")

if __name__ == "__main__":
    main()
