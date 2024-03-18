class PadSequences:
    def __init__(self, max_sequence_length, padding_value=0):
        if not isinstance(max_sequence_length, int) or max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be a positive integer")
        self.max_sequence_length = max_sequence_length
        self.padding_value = padding_value

    def pad_sequences(self, sequences):
        padded_sequences = []
        for sequence in sequences:
            if not isinstance(sequence, list):
                raise ValueError("Input sequences must be a list of lists")
            if len(sequence) >= self.max_sequence_length:
                padded_sequence = sequence[:self.max_sequence_length]
            else:
                padded_sequence = sequence + [self.padding_value] * (self.max_sequence_length - len(sequence))
            padded_sequences.append(padded_sequence)
        return padded_sequences

#  added input validation for checking max_sequence_length is a +ve number, to ensure that sequence is list of lists or else raise a ValueError 
