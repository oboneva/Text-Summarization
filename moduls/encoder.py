import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class EncoderBiLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, device):
        super(EncoderBiLSTM, self).__init__()

        self.device = device
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embed_size)

        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size, bidirectional=True, batch_first=True)

        self.fc = nn.Linear(in_features=self.hidden_size,
                            out_features=self.vocab_size)

    def forward(self, text, text_len):
        # embed text from [batch_size, seq_len] to [batch_size, seq_len, embedding_dim]
        # ex: [8, 16] -> [8, 16, 500]
        text_embed = self.embedding(text)

        # output: torch.Size([8, 1409, 512]) (batch, seq_len, vocab_size, bidirectional * hidden_size)
        # hidden_state_n, cell_state_n: torch.Size([2, 1409, 256])
        # output, (hidden_state_n, cell_state_n) = self.lstm(text_embed)

        # pack the padded, sorted and embedded text
        packed_captions = pack_padded_sequence(
            text_embed, text_len.cpu().numpy(), True)

        output, (hidden_state_n, cell_state_n) = self.lstm(packed_captions)

        return (hidden_state_n, cell_state_n)


def main():
    pass


if __name__ == "__main__":
    main()
