import torch
from torch.nn.utils.rnn import pad_sequence


class CollateText:
    def __init__(self, padding_value: int, batch_first: bool):
        self.padding_value = padding_value
        self.batch_first = batch_first

    def __call__(self, batch):
        (text, summary) = zip(*batch)

        text_len = torch.LongTensor(list(map(len, text)))
        text_padded = pad_sequence(
            text, batch_first=self.batch_first, padding_value=self.padding_value)

        summary_len = torch.LongTensor(list(map(len, summary)))
        summary_padded = pad_sequence(
            summary, batch_first=self.batch_first, padding_value=self.padding_value)

        return (text_padded, text_len, summary_padded, summary_len)
