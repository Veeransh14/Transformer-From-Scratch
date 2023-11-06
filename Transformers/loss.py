import cupy as cp

class CrossEntropy():
    def __init__(self, padding_id, vocab_size):
        self.eps = 1e-6
        self.padding_id = padding_id
        self.vocab_size = vocab_size

    def one_hot(self, label):
        label_num = len(label)
        one_hot = cp.zeros((label_num, self.vocab_size))
        one_hot[cp.arange(label_num), label] = 1
        self.one_hot_label = one_hot
        return one_hot

    def forward(self, pred, label):
        one_hot_label = self.one_hot(label)
        loss = -(one_hot_label * cp.log(pred + self.eps)).sum(-1)
        return loss

    def grad(self, pred, label):
        grad = pred - self.one_hot_label
        return grad

class MSE():
    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

    def forward(self):
        loss = ((self.y_pred - self.y_true) ** 2).mean()
        return loss

    def grad(self):
        grad = 2 * (self.y_pred - self.y_true)
        return grad