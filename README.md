
# LLM-Vorlesung – Skript, Folien & Python-Demos

Dieses Repository bündelt:

- das Textskript (`lectures/llm_vorlesung.md`)
- ein Folienskript mit eingebetteten PNGs und Links zu Demos (`lectures/llm_vorlesung_slides.md`)
- Python-Demos zu den Rechenbeispielen (`demos/`)
- eine einfache GitHub-Pages-Struktur (`docs/`)

## Demos ausführen

Empfohlen: virtuelles Environment nutzen.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

Dann z. B.:

```bash
python -m demos.tokenization_demo
python -m demos.self_attention_demo
python -m demos.softmax_demo
python -m demos.crossentropy_demo
python -m demos.gradient_descent_demo
```

## GitHub Pages

- `docs/index.md` ist als Startseite gedacht.
- In GitHub -> Settings -> Pages den Branch `main` und Folder `/docs` auswählen.
