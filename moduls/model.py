from torch import nn
import torch
from moduls.decoder import DecoderLSTM
from moduls.encoder import EncoderBiLSTM
import configs


class EncoderDecoder(nn.Module):
    def __init__(self, model_config: configs.model_config, vocab_size: int, device):
        super(EncoderDecoder, self).__init__()

        self.encoder = EncoderBiLSTM(embed_size=model_config.embed_size,
                                     hidden_size=model_config.hidden_size,
                                     vocab_size=vocab_size,
                                     device=device)

        self.decoder = DecoderLSTM(vocab_size=vocab_size,
                                   hidden_size=model_config.hidden_size * 2,
                                   embed_size=model_config.embed_size,
                                   device=device)

        self.to(device)

    def forward(self, text, text_len, summary, summary_len):
        text_len, perm_idx = text_len.sort(0, descending=True)
        text = text[perm_idx]
        summary = summary[perm_idx]
        summary_len = summary_len[perm_idx]

        # torch.Size([2, 844, 512]) torch.Size([2]) torch.Size([4, 2, 256]) torch.Size([4, 2, 256])
        seq_unpacked, lens_unpacked, (hidden_state_n, cell_state_n) = self.encoder(
            text, text_len)

        batch, seq_len, hidden_size = seq_unpacked.size()

        hidden_state_n = hidden_state_n.view(2, 2, batch, -1)
        cell_state_n = cell_state_n.view(2, 2, batch, -1)

        last_hidden = hidden_state_n[-1]
        last_cell = cell_state_n[-1]

        last_hidden_fwd = last_hidden[0]
        last_hidden_bwd = last_hidden[1]

        last_cell_fwd = last_cell[0]
        last_cell_bwd = last_cell[1]

        new_hidden = torch.cat((last_hidden_fwd, last_hidden_bwd), 1)
        new_cell = torch.cat((last_cell_fwd, last_cell_bwd), 1)

        return self.decoder((new_hidden, new_cell), summary, summary_len)


def main():
    pass


if __name__ == "__main__":
    main()
