import numpy as np
from collections import Counter
from src.preprocessing import preprocess_pipeline

class CustomCountVectorizer:
    def __init__(self, max_features=1000):
        self.max_features = max_features
        self.vocabulary_ = {}
        self.feature_names_ = []

    def fit(self, raw_documents):
        """
        Builds a vocabulary of up to max_features most frequent words from raw_documents.
        """
        word_counts = Counter()
        for doc in raw_documents:
            # Lowercase consistently so vocabulary deduplicates casing variants
            if isinstance(doc, str):
                tokens = doc.lower().split()
            else:
                tokens = [str(t).lower() for t in doc]
            word_counts.update(tokens)

        # Get top max_features words
        most_common = word_counts.most_common(self.max_features)
        
        # Build vocabulary dictionary mapping word -> index
        self.vocabulary_ = {word: idx for idx, (word, _) in enumerate(most_common)}
        self.feature_names_ = [word for word, _ in most_common]
        return self

    def transform(self, raw_documents):
        """
        Transforms raw_documents into a term-document count matrix.
        Returns a dense NumPy array.
        """
        num_docs = len(raw_documents)
        vocab_size = len(self.vocabulary_)
        X = np.zeros((num_docs, vocab_size), dtype=np.float32)

        for idx, doc in enumerate(raw_documents):
            if isinstance(doc, str):
                tokens = doc.lower().split()
            else:
                tokens = [str(t).lower() for t in doc]
            
            # Count words for this document
            doc_counts = Counter(tokens)
            for word, count in doc_counts.items():
                if word in self.vocabulary_:
                    X[idx, self.vocabulary_[word]] = count
        return X

    def fit_transform(self, raw_documents):
        return self.fit(raw_documents).transform(raw_documents)


class CustomTfidfVectorizer:
    def __init__(self, max_features=1000, norm='l2'):
        self.max_features = max_features
        self.norm = norm
        self.count_vectorizer = CustomCountVectorizer(max_features=max_features)
        self.idf_ = None

    def fit(self, raw_documents):
        """
        Fits CountVectorizer and computes IDF weights.
        """
        # Fit count vectorizer and transform to term counts
        X_counts = self.count_vectorizer.fit(raw_documents).transform(raw_documents)
        n_samples = X_counts.shape[0]

        # Document frequency (DF): number of documents containing each word
        # We sum the boolean mask (X_counts > 0) along columns
        df = np.sum(X_counts > 0, axis=0)

        # Compute IDF using scikit-learn standard formula (with smoothing)
        # idf(t) = log((1 + n_samples) / (1 + df(t))) + 1
        self.idf_ = np.log((1 + n_samples) / (1 + df)) + 1
        
        # Save vocabulary references
        self.vocabulary_ = self.count_vectorizer.vocabulary_
        self.feature_names_ = self.count_vectorizer.feature_names_
        return self

    def transform(self, raw_documents):
        """
        Transforms raw_documents into TF-IDF representation.
        """
        X_counts = self.count_vectorizer.transform(raw_documents)
        
        # TF-IDF = TF * IDF (using broadcasting)
        X_tfidf = X_counts * self.idf_

        # Apply normalization if specified
        if self.norm == 'l2':
            # Row-wise L2 norm: sqrt(sum(val^2))
            norms = np.linalg.norm(X_tfidf, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0.0] = 1.0
            X_tfidf = X_tfidf / norms

        return X_tfidf

    def fit_transform(self, raw_documents):
        return self.fit(raw_documents).transform(raw_documents)


class SequenceTokenizer:
    """
    Tokenizer and sequence padder for Recurrent Neural Networks (LSTM/GRU).
    Maps raw text documents into fixed-length integer token sequences.
    0: <PAD>, 1: <UNK>, 2..N: Word vocabulary indices.
    Fully serializable via pickle with zero external dependencies.
    """
    PAD_TOKEN = 0
    UNK_TOKEN = 1

    def __init__(self, max_vocab=5000, max_len=150):
        self.max_vocab = max_vocab
        self.max_len = max_len
        self.word_to_idx = {"<PAD>": self.PAD_TOKEN, "<UNK>": self.UNK_TOKEN}
        self.idx_to_word = {self.PAD_TOKEN: "<PAD>", self.UNK_TOKEN: "<UNK>"}
        self.vocab_size = 2

    def fit(self, raw_documents):
        """
        Builds the vocabulary from raw text documents based on frequency.
        """
        counts = Counter()
        for doc in raw_documents:
            if isinstance(doc, str):
                tokens = doc.lower().split()
            else:
                tokens = [str(t).lower() for t in doc]
            counts.update(tokens)

        # Reserve indices 0 (<PAD>) and 1 (<UNK>)
        top_words = counts.most_common(self.max_vocab - 2)
        for idx, (word, _) in enumerate(top_words, start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        self.vocab_size = len(self.word_to_idx)
        return self

    def transform(self, raw_documents):
        """
        Transforms raw text documents into a padded/truncated 2D NumPy array of shape (N, max_len).
        """
        num_docs = len(raw_documents)
        sequences = np.full((num_docs, self.max_len), self.PAD_TOKEN, dtype=np.int32)

        for i, doc in enumerate(raw_documents):
            if isinstance(doc, str):
                tokens = doc.lower().split()
            else:
                tokens = [str(t).lower() for t in doc]

            # Truncate tokens to max_len if necessary
            tokens = tokens[:self.max_len]

            for t, token in enumerate(tokens):
                sequences[i, t] = self.word_to_idx.get(token, self.UNK_TOKEN)

        return sequences

    def fit_transform(self, raw_documents):
        return self.fit(raw_documents).transform(raw_documents)
