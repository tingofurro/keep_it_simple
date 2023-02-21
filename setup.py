import subprocess
import sys

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-Ur", "requirements.txt"]
)

import nltk

nltk.download("stopwords")

subprocess.check_call(["python", "-m", "spacy", "download", "en_core_web_sm"])
