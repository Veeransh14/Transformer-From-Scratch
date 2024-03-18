class Tokenizer:
    def __init__(self, oov_token='<OOV>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True):
        self.word_index = {oov_token: 1}  # Start with the OOV token
        self.index_word = {1: oov_token}  # Index 1 corresponds to the OOV token
        self.oov_token = oov_token  # Token to use for out-of-vocabulary words
        self.filters = set(filters)  # Use a set for faster character filtering
        self.lower = lower  # Convert text to lowercase

    def fit_on_texts(self, texts):
        # Iterate through the input texts and build the vocabulary
        word_counts = {}  # A dictionary to store word frequencies
        for text in texts:
            if self.lower:
                text = text.lower()
            # Use translate() for efficient character filtering
            text = text.translate(str.maketrans('', '', ''.join(self.filters)))
            words = text.split()
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort tokens by frequency and update word_index and index_word
        sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
        for idx, word in enumerate(sorted_words, start=len(self.word_index)):
            self.word_index[word] = idx
            self.index_word[idx] = word

    def texts_to_sequences(self, texts):
        # Convert input texts to sequences of token IDs
        sequences = []
        for text in texts:
            if self.lower:
                text = text.lower()
            text = text.translate(str.maketrans('', '', ''.join(self.filters)))
            words = text.split()
            sequence = [self.word_index.get(word, self.word_index[self.oov_token]) for word in words]
            sequences.append(sequence)
        return sequences

    def get_vocab_size(self):
        # Return the size of the vocabulary
        return len(self.word_index)

    def get_config(self):
        # Return the configuration for the tokenizer (useful for saving the tokenizer)
        return {
            'oov_token': self.oov_token,
            'filters': ''.join(self.filters),
            'lower': self.lower
        }
