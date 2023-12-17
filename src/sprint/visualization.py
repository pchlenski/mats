import torch
import pandas as pd
import matplotlib.pyplot as plt
from .feature_exploration import max_activating_examples
from .loading import load_model


def visualize_topk(feature_id, n_examples, model=None, pad=True, clip=None):
    if model is None:
        model = load_model()

    ex, val = max_activating_examples(feature_id, n_examples, return_feature_values=True)

    max_val_idx = val.argmax(dim=1)
    min_idx, max_idx = max_val_idx.min().item(), max_val_idx.max().item()

    if pad:
        ex_padded = []
        vals_padded = []
        for i in range(n_examples):
            row_max = max_val_idx[i].item()
            ex_padded.append(
                torch.cat(
                    [
                        torch.zeros(max_idx - row_max, dtype=torch.int),
                        ex[i].cpu(),
                        torch.zeros(row_max - min_idx, dtype=torch.int),
                    ]
                )
            )
            vals_padded.append(
                torch.cat(
                    [
                        torch.zeros(max_idx - row_max, dtype=torch.int),
                        val[i].cpu(),
                        torch.zeros(row_max - min_idx, dtype=torch.int),
                    ]
                )
            )

        ex = torch.stack(ex_padded)
        vals = torch.stack(vals_padded)

        if clip is not None:
            ex = ex[:, max_idx - clip : max_idx + clip]
            vals = vals[:, max_idx - clip : max_idx + clip]

    vals[ex < 2] = torch.nan

    # Plot
    df = pd.DataFrame([[model.tokenizer.decode(x) for x in row] for row in ex])

    fig = plt.figure(figsize=(df.shape[1], df.shape[0]))
    plt.imshow(vals, cmap="coolwarm", vmin=0)
    plt.colorbar()

    for i, row in df.iterrows():
        for j, token in enumerate(row):
            if token not in ["<|EOS|>", "<|PAD|>", "<|BOS|>"]:
                plt.text(j, i, token, ha="center", va="center")

    plt.xticks(range(df.shape[1]))
    plt.yticks(range(df.shape[0]))
    plt.tight_layout()
