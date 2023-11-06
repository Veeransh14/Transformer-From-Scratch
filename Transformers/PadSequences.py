
class PadSequences:
    def __init__(self, max_sequence_length, padding_value=0):
        self.max_sequence_length = max_sequence_length
        self.padding_value = padding_value

    def pad_sequences(self, sequences):
        padded_sequences = []
        for sequence in sequences:
            if len(sequence) >= self.max_sequence_length:
                # If the sequence is longer than the specified length, truncate it
                padded_sequence = sequence[:self.max_sequence_length]
            else:
                # If the sequence is shorter, pad it with the padding value
                padded_sequence = sequence + [self.padding_value] * (self.max_sequence_length - len(sequence))
            padded_sequences.append(padded_sequence)
        return padded_sequences








