# fix_nltk.py
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download all required NLTK data
print("Downloading NLTK data... This may take a few minutes.")

# Download punkt (the main tokenizer)
nltk.download('punkt', quiet=False)

# Download the new punkt_tab if available
try:
    nltk.download('punkt_tab', quiet=False)
    print("✓ punkt_tab downloaded successfully")
except:
    print("ℹ punkt_tab not available, using standard punkt")

# Download other potentially needed data
nltk.download('wordnet', quiet=False)
nltk.download('omw-1.4', quiet=False)

print("✓ All NLTK downloads completed!")
print("You can now run your VQA system.")