import torch
from torch.utils.data import DataLoader
from torch.nn import Module
from rouge import Rouge


class Evaluator:
    @torch.no_grad()
    def eval(self, model: Module, dl: DataLoader, verbose: bool, writer, writer_section: str, device, vocab):
        model.eval()

        total_items = 0

        for step, (text_padded, text_len, summary_padded, summary_len) in enumerate(dl):

            text_padded = text_padded.to(device)
            text_len = text_len.to(device)
            summary_padded = summary_padded.to(device)
            summary_len = summary_len.to(device)

            batch_size = text_padded.size(0)
            total_items += batch_size

            # sort captions by sequence length in descending order
            text_len, perm_idx = text_len.sort(0, descending=True)
            text_padded = text_padded[perm_idx]
            summary_padded = summary_padded[perm_idx]
            summary_len = summary_len[perm_idx]

            # encode articles and generate summaries
            encoded_text = model.encoder(text_padded, text_len)
            generated_summaries = model.decoder(
                encoded_text, vocab, batch_size)

            generated_summaries = [' '.join(summary)
                                   for summary in generated_summaries]

            # revert padded summaries to text
            summary_padded = [summary_padded[i][1:summary_len[i] + 1]
                              for i in range(len(summary_padded))]
            references = [' '.join([vocab.itos[num] for num in summary])
                          for summary in summary_padded]

            scores = Rouge().get_scores(generated_summaries, references, avg=True)

            print(scores)

            if step % 5 == 0:
                print("Reference: ", references[0])
                print("Generated: ", generated_summaries[0])

            # writer.add_scalar(
            #     "{}/METEOR".format(writer_section), meteor)

            # if verbose:
            #     print(f"METEOR ", meteor)

            # return [meteor]


def main():
    pass


if __name__ == "__main__":
    main()
