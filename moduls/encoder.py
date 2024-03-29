import torch
from torch import nn
from torch.nn.modules import dropout
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class EncoderBiLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, device):
        super(EncoderBiLSTM, self).__init__()

        self.device = device
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embed_size)

        self.dropout = nn.Dropout(0.3)

        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=2,
                            bidirectional=True,
                            batch_first=True,
                            dropout=0.3)

        self.fc = nn.Linear(in_features=self.hidden_size,
                            out_features=self.vocab_size)

    def forward(self, text, text_len):
        # embed text from [batch_size, seq_len] to [batch_size, seq_len, embedding_dim]
        # ex: [8, 16] -> [8, 16, 500]
        text_embed = self.embedding(text)
        text_embed = self.dropout(text_embed)

        # pack the padded, sorted and embedded text
        packed_captions = pack_padded_sequence(
            text_embed, text_len.cpu().numpy(), batch_first=True)

        # output: tensor of shape (L,N,D∗Hout) ex. (batch, seq, 2 * hidden_size)
        # h_n: tensor of shape (D∗num_layers,N,Hout) ex. (2 * 2, batch, hidden_size)
        # c_n: tensor of shape (D∗num_layers,N,Hcell) ex. (2 * 2, batch, hidden_size)

        output, (hidden_state_n, cell_state_n) = self.lstm(packed_captions)

        seq_unpacked, lens_unpacked = pad_packed_sequence(
            output, batch_first=True)

        # torch.Size([2, 844, 512]) torch.Size([4, 2, 256]) torch.Size([4, 2, 256])
        return seq_unpacked, lens_unpacked, (hidden_state_n, cell_state_n)


def main():
    pass


if __name__ == "__main__":
    main()
