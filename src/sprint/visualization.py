from sprint.vars import BATCH_SIZE
import torch
import pandas as pd
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import plotly.express as px

from transformer_lens import utils

from .feature_exploration import max_activating_examples
from .loading import load_model, load_data
from .sae_tutorial import process_tokens, process_token
from .attention import get_attn_head_contribs, get_attn_head_contribs_ov
from .vars import BATCH_SIZE


def get_topk(feature_id, n_examples, model=None, data=None, pad=True, clip=None, evenly_spaced=False):
    if model is None:
        model = load_model()
    if data is None:
        data = load_data(model=model)

    ex, val, r, c = max_activating_examples(feature_id, n_examples, model=model, evenly_spaced=evenly_spaced)

    max_val_idx = val.argmax(dim=1)
    min_idx, max_idx = max_val_idx.min().item(), max_val_idx.max().item()

    if pad:
        ex_padded = []
        val_padded = []
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
            val_padded.append(
                torch.cat(
                    [
                        torch.zeros(max_idx - row_max, dtype=torch.int),
                        val[i].cpu(),
                        torch.zeros(row_max - min_idx, dtype=torch.int),
                    ]
                )
            )

        ex = torch.stack(ex_padded)
        val = torch.stack(val_padded)

        if clip is not None:
            ex = ex[:, max_idx - clip : max_idx + clip]
            val = val[:, max_idx - clip : max_idx + clip]

    val[ex < 2] = torch.nan

    # Plot
    df = pd.DataFrame([[model.tokenizer.decode(x) for x in row] for row in ex], index=r.cpu().numpy())
    val = pd.DataFrame(val, index=r.cpu().numpy())

    return df, val, c


def visualize_topk(feature_id, n_examples, model=None, data=None, pad=True, clip=None, evenly_spaced=False):
    df, val, c = get_topk(
        feature_id, n_examples, model=model, data=data, pad=pad, clip=clip, evenly_spaced=evenly_spaced
    )
    fig = plt.figure(figsize=(df.shape[1] // 2, df.shape[0] // 2))
    ax = fig.add_subplot(111)
    ax.matshow(val, cmap="coolwarm", vmin=0)

    # for i, row in df.iterrows():
    for i, (_, row) in enumerate(df.iterrows()):  # Hacky but I need indices
        for j, token in enumerate(row):
            if token not in ["<|EOS|>", "<|PAD|>", "<|BOS|>"]:
                plt.text(j, i, token, ha="center", va="center", fontsize=8)

    ax.set_xticks(range(len(df.columns)), df.columns)
    ax.set_yticks(range(len(df.index)), [f"R={dfi}, C={ci}" for dfi, ci in zip(df.index, c)])

    return ax


def visualize_topk_plotly(feature_id, n_examples, model=None, data=None, pad=True, clip=None):
    df, vals = get_topk(feature_id, n_examples, model=model, data=data, pad=pad, clip=clip)

    # Create Plotly figure
    fig = go.Figure(
        data=go.Heatmap(z=vals.numpy(), x=list(range(df.shape[1])), y=list(range(df.shape[0])), colorscale="rdbu")
    )

    # Add text annotations
    for i, row in df.iterrows():
        for j, token in enumerate(row):
            if token not in ["", "", ""]:
                fig.add_annotation(x=j, y=i, text=token, showarrow=False, xanchor="center", yanchor="middle")

    # Update layout
    fig.update_layout(
        xaxis=dict(tickmode="array", tickvals=list(range(df.shape[1]))),
        yaxis=dict(tickmode="array", tickvals=list(range(df.shape[0]))),
        height=df.shape[0] * 40,  # Adjust height based on number of rows
        width=df.shape[1] * 80,  # Adjust width based on number of columns
    )

    return fig


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
    # fig = plt.figure(figsize=(contribs.shape))
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


def plot_attn_contribs_for_example(
    model, data, example_idx, token_idx, feature_mid, start_token_idx=0, ov_only=False, layer=0, batch_size=BATCH_SIZE
):
    with torch.no_grad():
        tokens = data[example_idx]
        _, cache = model.run_with_cache(
            tokens,
            stop_at_layer=1,
            names_filter=[utils.get_act_name("pattern", 0), utils.get_act_name("v", 0)],
        )
        if not ov_only:
            attn_contribs = get_attn_head_contribs(
                model=model, layer=layer, cache=cache, data=data, batch_size=batch_size, range_normal=feature_mid
            )
            attn_contribs_window = attn_contribs[0, :, token_idx, start_token_idx : token_idx + 1]
        else:
            attn_contribs = get_attn_head_contribs_ov(
                model=model, layer=layer, cache=cache, data=data, batch_size=batch_size, range_normal=feature_mid
            )
            attn_contribs_window = attn_contribs[0, :, start_token_idx : token_idx + 1]

        # Matplotlib code
        fig = plt.figure(figsize=(attn_contribs_window.shape[1] // 2, attn_contribs_window.shape[0] // 2))
        plt.imshow(attn_contribs_window.detach().cpu().numpy())
        plt.xticks(
            range(token_idx - start_token_idx + 1),
            [model.to_string(x) for x in tokens[start_token_idx : token_idx + 1]],
            rotation=90,
        )
        # print(attn_contribs_window.sum().item())
        # fig = px.imshow(
        #     utils.to_numpy(attn_contribs_window),
        #     x=list(
        #         map(
        #             lambda x, i: f"|{process_token(x, model=model)}| pos {str(i)}",
        #             model.tokenizer.batch_decode(tokens[start_token_idx : token_idx + 1]),
        #             range(start_token_idx, token_idx + 1),
        #         )
        #     ),
        #     color_continuous_midpoint=0,
        # )
        # fig.update_xaxes(tickangle=90)
        # fig = go.Figure(
        #     data=go.Heatmap(
        #         z=utils.to_numpy(attn_contribs_window),
        #         x=list(
        #             map(
        #                 lambda x, i: f"|{process_token(x, model=model)}| pos {str(i)}",
        #                 model.tokenizer.batch_decode(tokens[start_token_idx : token_idx + 1]),
        #                 range(start_token_idx, token_idx + 1),
        #             )
        #         ),
        #         colorscale="rdbu",
        #     )
        # )
        # fig.update_layout(
        #     xaxis=dict(tickmode="array", tickvals=list(range(token_idx - start_token_idx + 1))),
        #     height=300,
        #     width=token_idx - start_token_idx + 1,
        # )

        # return fig
