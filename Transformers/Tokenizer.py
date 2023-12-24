
class Tokenizer:
    def __init__(self, oov_token='<OOV>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True):
        self.word_index = {}  # A dictionary to map words to token IDs
        self.index_word = {}  # A dictionary to map token IDs to words
        self.oov_token = oov_token  # Token to use for out-of-vocabulary words
        self.filters = filters  # Characters to filter out from the text
        self.lower = lower  # Convert text to lowercase

    def fit_on_texts(self, texts):
        # Iterate through the input texts and build the vocabulary
        tokens = []  # A list to store tokens from all texts
        for text in texts:
            if self.lower:
                text = text.lower()
            for char in self.filters:
                text = text.replace(char, ' ')
            words = text.split()
            tokens.extend(words)
        
        # Sort tokens by frequency
        word_counts = {}  # A dictionary to store word frequencies
        for word in tokens:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
        
        # Add tokens to the word_index and index_word dictionaries
        for idx, word in enumerate(sorted_words):
            self.word_index[word] = idx + 1  # Reserve 0 for the OOV token
            self.index_word[idx + 1] = word

    def texts_to_sequences(self, texts):
        # Convert input texts to sequences of token IDs
        sequences = []
        for text in texts:
            if self.lower:
                text = text.lower()
            for char in self.filters:
                text = text.replace(char, ' ')
            words = text.split()
            sequence = []
            for word in words:
                if word in self.word_index:
                    sequence.append(self.word_index[word])
                else:
                    sequence.append(self.word_index[self.oov_token])
            sequences.append(sequence)
        return sequences
    def get_vocab(self):
        return len(self.word_index)

    def get_config(self):
        # Return the configuration for the tokenizer (useful for saving the tokenizer)
        return {
            'oov_token': self.oov_token,
            'filters': self.filters,
            'lower': self.lower
        }

