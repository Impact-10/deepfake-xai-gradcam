Submission README
==================

This file explains what is inside the ZIP produced for LMS submission and how to run the code locally.

Files included by the packaging script:
- Python source files (*.py) at repository root and in `scripts/` (packaging script included)
- Web app files (Streamlit app), if present (e.g., `app.py`)
- Supporting modules: `gradcam.py`, `utils.py`, `train.py`, `train_anti_overfitting.py`, `evaluate.py`, `compare.py`
- `dataset/` directory (all images/frames that you included)
- `requirements.txt` (minimal dependencies list)
- `README.md` (paper and instructions)

What the packaging script does (quick):
- Exclude `checkpoints/`, `outputs/`, `.git/`, `__pycache__/` by default
- Skip model checkpoint files (.pth/.pt/.h5)
- Skip files larger than 200MB by default (adjustable in the script)

How to create the ZIP (from repo root):

```bash
# From repository root
python scripts/package_submission.py --output ../submission_code_dataset.zip
```

This will create `submission_code_dataset.zip` one level above the repo root (adjust `--output` if you prefer another location).

How to inspect the ZIP (verify contents):

```bash
# List zip contents
unzip -l ../submission_code_dataset.zip

# Or view specific files inside
unzip -l ../submission_code_dataset.zip | head -n 50
```

How to run the web app (if Streamlit):

1. Create a virtual environment and install requirements:

```bash
python -m venv venv
# activate venv (Windows example)
# bash (Git Bash or WSL):
source venv/Scripts/activate  # or `venv\Scripts\activate` on cmd/powershell
pip install -r requirements.txt
```

2. Start Streamlit app (example file `app.py`):

```bash
streamlit run app.py
```

Notes and tips:
- If your dataset is very large, the packaging script will still include it; check final ZIP size before upload.
- If the LMS has a file size limit, exclude the `dataset/` from the ZIP and instead upload dataset separately (as allowed). The script can be edited to skip `dataset` if needed.
- If you intentionally need to include checkpoints, move them to a separate small-archive and mention in the submission note.

Contact:
If you want, I can run the packaging script here and produce the final ZIP (I will not include large checkpoints by default). Tell me to proceed and I will create `submission_code_dataset.zip` and report its size and top-level contents.
