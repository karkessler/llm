"""
Gradientenberechnung – Demo für lineare Modelle
-----------------------------------------------
Passt zum Folienteil:
    z = W · x + b
    p = z (lineares Modell)
    L = (p - y)^2
    ∂L/∂W = 2 · (p - y) · x
"""

import numpy as np
import matplotlib.pyplot as plt

# Trainingsdaten (1 Feature, 1 Beispiel)
x = np.array([2.0])      # Eingabe
y = np.array([4.0])      # Zielwert

# Initiales Gewicht und Bias
W = np.array([0.5])
b = 0.0

# Lernrate
eta = 0.1

# Verlauf für Plot
weight_history = []
loss_history = []

def forward(x, W, b):
    """Lineares Modell: z = W * x + b"""
    return W * x + b

def loss(p, y):
    """Quadratischer Fehler"""
    return (p - y) ** 2

def gradient(x, y, W, b):
    """
    ∂L/∂W = 2·(p - y)·x
    ∂L/∂b = 2·(p - y)
    """
    p = forward(x, W, b)
    dL_dW = 2 * (p - y) * x
    dL_db = 2 * (p - y)
    return dL_dW, dL_db

# Trainingsschritte
for step in range(25):
    p = forward(x, W, b)
    L = loss(p, y)

    weight_history.append(W.copy())
    loss_history.append(L.item())

    dW, db = gradient(x, y, W, b)

    # Gradient Descent
    W = W - eta * dW
    b = b - eta * db

# ---- Plot ----
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(weight_history, label="W-Verlauf")
plt.xlabel("Schritt")
plt.ylabel("Gewicht W")
plt.title("Gradient Descent auf W")

plt.subplot(1, 2, 2)
plt.plot(loss_history, label="Loss", color="red")
plt.xlabel("Schritt")
plt.ylabel("Loss L")
plt.title("Fehlerverlauf")

plt.tight_layout()
plt.show()

print("Finales Gewicht W:", W)
print("Finaler Bias b:", b)
