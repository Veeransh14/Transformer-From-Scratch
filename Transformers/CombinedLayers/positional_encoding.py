
import cupy as cp
class PositionalEncoding():
    def __init__(self, max_len, d_model, data_type=cp.float32):
        self.pe = cp.zeros((max_len, d_model), dtype=data_type)
        ev_cln = cp.arange(0, d_model, 2)
        diff = 1.0 / (10000) ** (ev_cln / d_model)
        pos = cp.arange(0, max_len)[:, cp.newaxis]
        self.pe[:, 0::2] = cp.sin(pos * diff)
        self.pe[:, 1::2] = cp.cos(pos * diff)
        self.pe = self.pe[cp.newaxis, :, :]
        del ev_cln, diff, pos

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]

    def backward(self, grad):
        return grad

