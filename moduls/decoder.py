from moduls.attention import Attention
import torch
from torch import nn


class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, attention_size, vocab_size, device):
        super(DecoderLSTM, self).__init__()

        self.device = device
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embed_size)

        self.lstm_cell = nn.LSTMCell(input_size=embed_size + hidden_size,
                                     hidden_size=hidden_size)

        self.attention = Attention(hidden_size, hidden_size, attention_size)

        self.fc = nn.Linear(in_features=self.hidden_size,
                            out_features=self.vocab_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_outputs, encoded_text, summary, summary_len):
        batch_size, seq_len = summary.size()
        # remove <eos> from the summaries
        for i in range(len(summary_len)):
            summary[i][summary_len[i] - 1] = 0

        summary = summary[:, :-1]
        seq_len -= 1

        # embed summaries from [batch_size, seq_len] to [batch_size, seq_len, embedding_dim]
        # ex: [8, 16] -> [8, 16, 500]
        summary = self.embedding(summary)

        # init initial hidden and cell state with the last hidden and cell state of the encoder ([fwd, bwd] since it is bidirectional)
        hidden_state_t, cell_state_t = encoded_text

        # output container
        # ex: [8, 16, 10 000
        outputs_container = torch.zeros(batch_size, seq_len, self.vocab_size).to(
            self.device)

        for t in range(seq_len):
            _, context = self.attention(encoder_outputs, hidden_state_t)
            lstm_input = torch.cat((summary[:, t], context), dim=1)

            hidden_state_t, cell_state_t = self.lstm_cell(
                lstm_input, (hidden_state_t, cell_state_t))

            output = self.fc(hidden_state_t)

            outputs_container[:, t] = output

        return outputs_container

    def summarize(self, encoder_outputs, encoded_text, vocab, max_len=75):
        batch_size, _, _ = encoder_outputs.size()

        hidden_state_n, cell_state_n = encoded_text
        hidden_state_n = hidden_state_n.view(2, 2, batch_size, -1)
        cell_state_n = cell_state_n.view(2, 2, batch_size, -1)

        last_hidden = hidden_state_n[-1]
        last_cell = cell_state_n[-1]

        last_hidden_fwd = last_hidden[0]
        last_hidden_bwd = last_hidden[1]

        last_cell_fwd = last_cell[0]
        last_cell_bwd = last_cell[1]

        hidden_state_n = torch.cat((last_hidden_fwd, last_hidden_bwd), 1)
        cell_state_n = torch.cat((last_cell_fwd, last_cell_bwd), 1)

        sos = torch.tensor(vocab.stoi['<sos>']).view(1, -1).to(self.device)
        embed = self.embedding(sos)

        summaries = []

        for t in range(max_len):
            _, context = self.attention(encoder_outputs, hidden_state_n)
            lstm_input = torch.cat((embed[:, 0], context), dim=1)

            hidden_state_n, cell_state_n = self.lstm_cell(
                lstm_input, (hidden_state_n, cell_state_n))

            output = self.fc(cell_state_n)
            output = output.view(batch_size, -1)

            best_word_idx = output.argmax(dim=1)

            summaries.append(best_word_idx)

            if vocab.itos[best_word_idx] == "<eos>":
                break

            embed = self.embedding(best_word_idx.unsqueeze(0))

        return [vocab.itos[idx] for idx in summaries]


def main():
    pass


if __name__ == "__main__":
    main()
