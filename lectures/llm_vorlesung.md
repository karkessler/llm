
# Vom Token zur Antwort – Wie Large Language Models wirklich funktionieren

Dieses Markdown-Dokument ist die textbasierte Version der PowerPoint-Präsentation
„Angewandte Mathematik 3 – Anwendung: Large Language Models“.

Es ist so strukturiert, dass es gut mit Git (Diffs), GitHub und GitHub Pages benutzt
werden kann und mit den Python-Demos im Ordner `demos/` verzahnt ist.

---

## 1. Motivation: Vom Token zur Antwort

- Maschinelles Lernen (ML) als Teilgebiet der KI
- Deep Learning (DL) mit tiefen neuronalen Netzen
- GPT = Generative Pretrained Transformer
- Kernfrage: Wie kann ein Modell nur durch Statistik scheinbar „verstehen“?

**Beispielsatz:**

> „Paris ist die Hauptstadt von Frankreich.“

Aufgabe des Modells: Das nächste Token vorhersagen, z. B. das Wort „Frankreich“.


## 2. Transformer-Architektur & Self-Attention

- Pipeline: Text → Tokenisierung → Embedding → Attention → Output → Softmax → Loss
- Fokus auf Decoder-only-Modelle (GPT)
- Self-Attention: Jedes Token „schaut“ auf alle anderen Tokens im Kontext

Verknüpfte Demo:

- `python -m demos.tokenization_demo`
- `python -m demos.self_attention_demo`


## 3. Mathematische Grundlagen

- Vektoren und Matrizen als Grundbausteine
- Matrixmultiplikation als Misch- und Gewichtungsoperation
- Gradientenberechnung mittels Kettenregel
- Gradient Descent: \( W_{neu} = W_{alt} - \eta \cdot \nabla L \)

Verknüpfte Demo:

- `python -m demos.gradient_descent_demo`


## 4. Rechenbeispiel: Wie arbeitet das Modell?

Beispielsatz mit Fokus auf das Token **„von“**:

> „Paris ist die Hauptstadt von Frankreich“

Schritte:

1. Embeddings für alle Tokens
2. Berechnung von Q, K, V über Gewichtsmatrizen \(W_Q, W_K, W_V\)
3. Attention-Scores via Skalarprodukt
4. Softmax über die Scores → Attention-Gewichte
5. Neuer kontextualisierter Vektor für „von“

Verknüpfte Demo:

- `python -m demos.self_attention_demo`


## 5. Lernprozess & Backpropagation

- Ziel: Cross-Entropy-Loss minimieren
- Softmax liefert Wahrscheinlichkeitsverteilung über das Vokabular
- Cross-Entropy misst, wie „falsch“ das Modell liegt
- Backpropagation verteilt den Fehler zurück durch alle Schichten
- Gewichte \(W_Q, W_K, W_V, W_{out}\) werden aktualisiert

Demo:

- `python -m demos.crossentropy_demo`
- `python -m demos.gradient_descent_demo`


## 6. Wie entscheidet das Modell? (Decoder-Vorhersage)

1. Kontextvektor \(h\) aus der Attention
2. Lineare Projektion: \( z = W_{out} h + b \)
3. Softmax: \( p_i = \frac{e^{z_i}}{\sum_j e^{z_j}} \)
4. Auswahl des wahrscheinlichsten Tokens (Argmax)


**Mini-Beispiel:**

- Vokabular: {Frankreich, Deutschland, Spanien}
- Logits berechnet über einfache \(W_{out}\)-Zeilen
- Softmax ergibt z. B. \(P(\text{Frankreich}) = 0{,}44\)

Demo:

- `python -m demos.softmax_demo`


## 7. Modellarchitekturen im Überblick

- Embedding-Schicht
- Multi-Head Self-Attention
- Feedforward-MLP (mit Aktivierungen wie GeLU)
- Residual-Verbindungen und LayerNorm
- Viele Wiederholungen dieser Blöcke (z. B. 12, 24, 96 Layer)


## 8. Titans – Ein Ausblick auf neue Architekturen

- Weiterentwicklung klassischer Transformer
- Kombination von Attention, MLP und Gating
- Verbesserte Positionskodierung (z. B. Rotary, State-Space)
- Spezielle Memory-Architekturen (Short-Term, Long-Term, Persistent Memory)


## 9. Grenzen von LLMs

- Halluzinationen
- Wissensgrenze (Trainingszeitpunkt)
- Fehlende Quellenangaben
- Begrenzte Kontextlänge
- Kosten für lange Sequenzen
- Kein echtes „Verstehen“, nur Musterlernen


## 10. Fazit & Diskussion

- LLMs sind mächtige Wahrscheinlichkeitsmodelle
- Mathematik dahinter: Vektoren, Matrizen, Wahrscheinlichkeiten
- Softmax & Cross-Entropy ermöglichen Lernen
- Backpropagation und Gradient Descent passen Gewichte an

Diskussionsfragen:

- Was bedeutet „Verstehen“ bei rein statistischen Modellen?
- Wie gehen wir mit Halluzinationen und Fehlern um?
- Wo sind LLMs sinnvoll, wo nicht?


## 11. Glossar (Auszug)

- **Token**: Kleinstes Textelement im Modell (Wort, Subwort, Zeichen)
- **Embedding**: Zahlenvektor, der ein Token im Modellraum repräsentiert
- **Self-Attention**: Mechanismus, bei dem Tokens aufeinander „achten“
- **Logits**: Roh-Scores vor der Softmax-Funktion
- **Softmax**: Wandelt Logits in Wahrscheinlichkeiten um
- **Cross-Entropy**: Verlustfunktion für Klassifikation
- **Backpropagation**: Rückwärtsausbreitung des Fehlers
- **Gradient Descent**: Optimierungsverfahren zur Gewichtsaktualisierung

---

Die dazugehörigen Python-Demos befinden sich im Ordner `demos/` und können
direkt ausgeführt werden, um die im Skript beschriebenen Rechenschritte
nachzuvollziehen.
