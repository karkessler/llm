
# LLM-Vorlesung – Skript & Python-Demos

Dieses Repository ist eine Git-Version deiner Vorlesung

> „Vom Token zur Antwort – Wie Large Language Models wirklich funktionieren“

und enthält:

- eine Markdown-Fassung des Skripts (`lectures/llm_vorlesung.md`)
- automatisch extrahierte Bilder aus der PowerPoint (`assets/`)
- Python-Demos zu den Rechenbeispielen (`demos/`)
- eine einfache GitHub-Pages-Struktur (`docs/`)

## Struktur

```text
.
├─ README.md
├─ lectures/
│   └─ llm_vorlesung.md
├─ assets/
│   └─ (Bilder aus der PowerPoint)
├─ demos/
│   ├─ tokenization_demo.py
│   ├─ self_attention_demo.py
│   ├─ softmax_demo.py
│   ├─ crossentropy_demo.py
│   └─ gradient_descent_demo.py
├─ docs/
│   └─ index.md
├─ requirements.txt
└─ .gitignore
```

## Installation & Ausführung der Demos

Empfohlen: virtuelles Environment nutzen.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

Demos starten, z. B.:

```bash
python -m demos.tokenization_demo
python -m demos.self_attention_demo
python -m demos.softmax_demo
python -m demos.crossentropy_demo
python -m demos.gradient_descent_demo
```

## GitHub-Pages (Website) einrichten

1. Das Repository auf GitHub hochladen
2. In den Repository-Einstellungen „Pages“ aktivieren
3. Als Quelle `main`-Branch und Ordner `/docs` auswählen
4. Die Startseite ist dann `docs/index.md`

Damit entsteht eine Online-Version deiner Vorlesung mit Verlinkung der Demos.
