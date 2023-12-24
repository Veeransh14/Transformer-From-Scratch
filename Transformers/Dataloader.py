from Tokenizer import Tokenizer  
from PadSequences import PadSequences  
class DataLoader:
    def __init__(self, max_sequence_length, padding_value):
        self.max_sequence_length = max_sequence_length
        self.padding_value = padding_value
        self.tokenizer = Tokenizer(oov_token='<OOV>', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True)
        self.pad_sequences = PadSequences(max_sequence_length=10, padding_value=0)

    def tokenize_text(self, text):
        # Tokenize the text using the Tokenizer class's fit_to_text and text_to_sequences methods
        tokenizer.fit_on_texts(text)
        sequences = tokenizer.texts_to_sequences(text)

        input_sequences = []
        output_sequences = []

        for sequence in sequences:
            for i in range(1, len(sequence)):
                input_seq = sequence[:i]
                output_word = sequence[i]

                input_sequences.append(input_seq)
                output_sequences.append(output_word)


        input_sequences = self.pad_sequences.pad_sequences(input_sequences)
       

        return input_sequences, output_sequences
