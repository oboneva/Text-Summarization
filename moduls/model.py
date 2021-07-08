from torch import nn
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
        # sort captions by sequence length in descending order
        text_len, perm_idx = text_len.sort(0, descending=True)
        text = text[perm_idx]
        summary = summary[perm_idx]
        summary_len = summary_len[perm_idx]

        encoded_text = self.encoder(text, text_len)
        return self.decoder(encoded_text, summary, summary_len)


def main():
    pass


if __name__ == "__main__":
    main()
