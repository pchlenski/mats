import torch
import pandas as pd
import matplotlib.pyplot as plt

from transformer_lens import utils

from .feature_exploration import max_activating_examples
from .loading import load_model
from .sae_tutorial import process_tokens
from .attention import get_attn_head_contribs


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


def plot_head_token_contribs(contribs, tokens, dst, start=0, end=None, model=None):
    if model is None:
        model = load_model()
    if end is None:
        end = len(tokens)
    if end < 0:
        end = len(tokens) + end
    token_strs = list(
        map(lambda x: f"|{x}|", process_tokens(model.tokenizer.batch_decode(tokens)[start:end], model=model))
    )

    fig, ax = plt.subplots()
    matimg = ax.matshow(contribs[0, :, dst, start:end].detach().cpu().numpy())
    ax.set_xticks(range(end - start))
    ax.set_xticklabels(token_strs, rotation=90)
    fig.colorbar(matimg)
    plt.show()


def plot_head_token_contribs_for_prompt(model, prompt, dst, range_normal, layer=0, start=0, end=None, prepend_bos=True):
    tokens = model.tokenizer(prompt).input_ids
    _, cache = model.run_with_cache(
        prompt,
        stop_at_layer=1,
        names_filter=[
            utils.get_act_name("pattern", layer),
            utils.get_act_name("v", layer),
        ],
        prepend_bos=prepend_bos,
    )
    contribs = get_attn_head_contribs(model, layer, range_normal, cache=cache)
    plot_head_token_contribs(contribs, tokens, dst, start, end)
    return contribs
