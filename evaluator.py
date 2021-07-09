import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from rouge import Rouge


class Evaluator:
    @torch.no_grad()
    def eval(self, model: Module, dl: DataLoader, verbose: bool, writer, writer_section: str, device, vocab):
        model.eval()

        rouge1f = 0
        rouge1p = 0
        rouge1r = 0
        rouge2f = 0
        rouge2p = 0
        rouge2r = 0
        rougelf = 0
        rougelp = 0
        rougelr = 0
        total_items = 0

        for step, (text_padded, text_lens, summary_padded, summary_lens) in enumerate(dl):

            text_padded = text_padded.to(device)
            text_lens = text_lens.to(device)
            summary_padded = summary_padded.to(device)
            summary_lens = summary_lens.to(device)

            batch_size = text_padded.size(0)
            total_items += batch_size

            # sort captions by sequence length in descending order
            text_lens, perm_idx = text_lens.sort(0, descending=True)
            text_padded = text_padded[perm_idx]
            summary_padded = summary_padded[perm_idx]
            summary_lens = summary_lens[perm_idx]

            for i in range(batch_size):
                text = text_padded[i: i + 1]
                text_len = text_lens[i: i + 1]
                summary = summary_padded[i]
                summary_len = summary_lens[i] - 1

                encoder_outputs, _, (hidden_state_n, cell_state_n) = model.encoder(
                    text.to(device), text_len)
                output = model.decoder.summarize(
                    encoder_outputs, (hidden_state_n, cell_state_n), vocab)

                predicted_sentence = ' '.join(output)

                # revert padded summaries to text
                summary_padded = summary[1:summary_len + 1]
                reference = ' '.join([vocab.itos[num]
                                      for num in summary_padded])

                scores = Rouge().get_scores(predicted_sentence, reference)

                rouge1f += scores["rouge-1"]["f"]
                rouge1p += scores["rouge-1"]["p"]
                rouge1r += scores["rouge-1"]["r"]
                rouge2f += scores["rouge-2"]["f"]
                rouge2p += scores["rouge-2"]["p"]
                rouge2r += scores["rouge-2"]["r"]
                rougelf += scores["rouge-l"]["f"]
                rougelp += scores["rouge-l"]["p"]
                rougelr += scores["rouge-l"]["r"]

            if step % 5 == 0:
                print("Reference: ", reference)
                print("Generated: ", predicted_sentence)

                print(scores)

        rouge1f /= total_items
        rouge1p /= total_items
        rouge1r /= total_items
        rouge2f /= total_items
        rouge2p /= total_items
        rouge2r /= total_items
        rougelf /= total_items
        rougelp /= total_items
        rougelr /= total_items

        writer.add_scalar(
            "{}/ROUGE-1-f".format(writer_section), rouge1f)
        writer.add_scalar(
            "{}/ROUGE-1-p".format(writer_section), rouge1p)
        writer.add_scalar(
            "{}/ROUGE-1-r".format(writer_section), rouge1r)

        writer.add_scalar(
            "{}/ROUGE-2-f".format(writer_section), rouge2f)
        writer.add_scalar(
            "{}/ROUGE-2-p".format(writer_section), rouge2p)
        writer.add_scalar(
            "{}/ROUGE-2-r".format(writer_section), rouge2r)

        writer.add_scalar(
            "{}/ROUGE-l-f".format(writer_section), rougelf)
        writer.add_scalar(
            "{}/ROUGE-l-p".format(writer_section), rougelp)
        writer.add_scalar(
            "{}/ROUGE-l-r".format(writer_section), rougelr)

        return [rouge1f, rouge1p, rouge1r, rouge2f, rouge2p, rouge2r, rougelf, rougelp, rougelr]


def main():
    pass


if __name__ == "__main__":
    main()
