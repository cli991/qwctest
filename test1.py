import nltk

nltk.data.path.append('/share/home/cli/code/memory/nltk_data')

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    print("NLTK resources found locally in utils.py")
except LookupError as e:
    print(f"Missing NLTK resource in utils.py: {e}")
    print("Offline mode: skip nltk.download()")