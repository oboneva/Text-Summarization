import torch
from torch import nn


class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, device):
        super(DecoderLSTM, self).__init__()

        self.device = device
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=self.embed_size)

        self.lstm = nn.LSTMCell(input_size=embed_size,
                                hidden_size=hidden_size)

        self.fc = nn.Linear(in_features=self.hidden_size,
                            out_features=self.vocab_size)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoded_text, summary, summary_len):
        batch_size, seq_len = summary.size()
        # remove <eos> from the summaries
        for i in range(len(summary_len)):
            summary[i][summary_len[i] - 1] = 0

        summary = summary[:, :-1]

        # embed summaries from [batch_size, seq_len] to [batch_size, seq_len, embedding_dim]
        # ex: [8, 16] -> [8, 16, 500]
        summary = self.embedding(summary)

        hidden_state_t, cell_state_t = encoded_text
        hidden_state_t = hidden_state_t.view(1, batch_size, -1)
        cell_state_t = cell_state_t.view(1, batch_size, -1)

        # output: torch.Size([8, 74, 512])
        # output, (hidden_state_t, cell_state_t) = self.lstm(
        #     summary, (hidden_state_t, cell_state_t))

        # output = self.fc(output)

        # output container
        # ex: [8, 16, 10 000
        outputs_container = torch.zeros(batch_size, seq_len, self.vocab_size).to(
            self.device)

        for t in range(seq_len):
            output, (hidden_state_t, cell_state_t) = self.lstm(
                summary[:, t], (hidden_state_t, cell_state_t))
            output = self.fc(output)

            outputs_container[:, t] = output

        return outputs_container

    def summarize(self, encoded_text, vocab, batch_size, max_len=75):
        hidden_state_t, cell_state_t = encoded_text
        hidden_state_t = hidden_state_t.view(1, batch_size, -1)
        cell_state_t = cell_state_t.view(1, batch_size, -1)

        sos = torch.tensor(vocab.stoi['<sos>']).view(1, -1).to(self.device)
        embed = self.embedding(sos)

        summaries = []

        for t in range(max_len):
            hidden_state_t, cell_state_t = self.lstm(
                embed, (hidden_state_t, cell_state_t))

            output = self.fc(hidden_state_t)
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
