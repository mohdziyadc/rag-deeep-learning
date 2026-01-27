import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import logging

logger = logging.getLogger(__name__)

# Mini Tokenizer
class Tokenizer():
    def __init__(self, remove_stopwords: bool = False,
    use_stemming: bool = False) -> None:
        
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.stemmer = PorterStemmer() if use_stemming else None

        self.stopwords = set(stopwords.words('english')) if remove_stopwords else set()

    
    def tokenize(self, text: str) -> list[str]:

        text = text.lower()

        # Remove special characters but keep spaces
        text = re.sub(r'[^a-z0-9\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        tokens = text.split()

        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopwords]
        
        if self.use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(t) for t in tokens]

        # Filter out very short tokens
        tokens = [t for t in tokens if len(t) > 1]

        return tokens

    def tokenize_query(self, query: str) -> list[str]:
        return self.tokenize(query)