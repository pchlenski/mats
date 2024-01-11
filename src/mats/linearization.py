from tqdm import tqdm
import torch
from transformer_lens import utils
from .loading import load_model, load_sae, load_data
from .vars import SAE_CFG, BATCH_SIZE


def cossim(a, b):
    """Take the cosine similarity between two vectors - usually for reverse-engineered features"""
    return (torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b))).item()


def get_tangent_plane_at_point(x_0_new, f, range_normal):
    """Linear approximation of f at x_0_new"""
    x_0_new.requires_grad_(True)
    g = lambda x: f(x) @ range_normal
    grad = torch.autograd.grad(g(x_0_new), x_0_new)
    return grad[0]


def ln2_mlp_until_post(x, ln, mlp, use_ln=True):
    """Get MLP activations for x"""
    if use_ln:
        x = ln(x)
    x = x @ mlp.W_in + mlp.b_in
    x = mlp.act_fn(x)
    return x


def ln2_mlp_until_out(x, ln, mlp, use_ln=True):
    """Get MLP outputs for x"""
    if use_ln:
        x = ln(x)
    return mlp(x)


def ln_ov(x, model, layer, head):
    """Run the OV circuit on x"""
    return model.blocks[layer].ln1(x) @ model.OV[layer, head]


def get_top_tokens(tokenizer, vector, k=5, reverse=False):
    """Get the top k tokens for a vector"""
    topk = torch.topk(vector, k=k, largest=(not reverse))
    return topk.values, tokenizer.batch_decode([[x] for x in topk.indices])


def get_feature_activations(
    model, feature_idx, token_idx, data, n_batches, batch_size, encoder, mlp_out=False, use_ln=False, layer=0
):
    with torch.no_grad():
        tokens = data[: batch_size * n_batches]
        hidden_acts = []
        for batch in tqdm(tokens.split(batch_size)):
            _, cache = model.run_with_cache(
                batch,
                stop_at_layer=1,
                names_filter=[
                    utils.get_act_name("post", 0),
                    utils.get_act_name("resid_mid", 0),
                    utils.get_act_name("attn_scores", 0),
                ],
            )
            mlp_acts = cache[utils.get_act_name("post", 0)]
            mlp_acts_flattened = mlp_acts.reshape(-1, SAE_CFG["d_mlp"])
            loss, x_reconstruct, hidden_acts_batch, l2_loss, l1_loss = encoder(mlp_acts_flattened)
            hidden_acts.append(hidden_acts_batch)
            # This is equivalent to:
            # hidden_acts = F.relu((mlp_acts_flattened - encoder.b_dec) @ encoder.W_enc + encoder.b_enc)

        hidden_acts = torch.cat(hidden_acts, dim=0)[feature_idx]
        feature = encoder.W_enc[:, feature_idx]

        mid_acts = cache[utils.get_act_name("resid_mid", layer)]
        x_mid = mid_acts[0, token_idx][None, None, :]
        my_fun = ln2_mlp_until_post if not mlp_out else ln2_mlp_until_out
        feature_mid = get_tangent_plane_at_point(
            x_mid, lambda x: my_fun(x, model.blocks[layer].ln2, model.blocks[layer].mlp, use_ln=use_ln), feature
        )[0, 0]

        mid_acts_feature_scores = mid_acts.reshape(-1, model.cfg.d_model) @ feature_mid

        return {"hidden acts": hidden_acts, "mid acts feature scores": mid_acts_feature_scores}


def analyze_linearized_feature(
    feature_idx,
    sample_idx,
    token_idx,
    layer=0,
    head=0,
    model=None,
    encoder=None,
    data=None,
    # batch_size=BATCH_SIZE,
    n_tokens=10,
    mlp_out=False,
    use_ln=True,
    feature=None,
):
    """
    Analyzes a whole feature example using linearizations.
    """
    # Get cache
    if model is None:
        model = load_model()
    if data is None:
        data = load_data(model=model)
    if encoder is None:
        encoder = load_sae()

    # Get cache
    _, cache = model.run_with_cache(
        # data[batch_start:batch_end],
        data[sample_idx],
        # stop_at_layer=1,
        names_filter=[
            utils.get_act_name("post", layer),
            utils.get_act_name("resid_mid", layer),
            utils.get_act_name("attn_scores", layer),
        ],
    )
    mlp_acts_flattened = cache[utils.get_act_name("post", layer)].reshape(-1, SAE_CFG["d_mlp"])
    mlp_out_flattened = cache[utils.get_act_name("resid_mid", layer)].reshape(-1, SAE_CFG["d_mlp"] // 4)
    _, _, hidden_acts, _, _ = encoder(mlp_out_flattened) if mlp_out else encoder(mlp_acts_flattened)
    # _, _, hidden_acts, _, _ = encoder(mlp_acts_flattened) if mlp_out else encoder(mlp_out_flattened)

    # Tweaks to feature vectors
    if feature is None:
        feature = encoder.W_enc[:, feature_idx]
    if mlp_out:
        feature = feature @ model.blocks[layer].mlp.W_out

    # Linearization component
    mid_acts = cache[utils.get_act_name("resid_mid", layer)]
    x_mid = mid_acts[0, token_idx][None, None, :]
    my_fun = ln2_mlp_until_post if not mlp_out else ln2_mlp_until_out
    feature_mid = get_tangent_plane_at_point(
        x_mid, lambda x: my_fun(x, model.blocks[layer].ln2, model.blocks[layer].mlp, use_ln=use_ln), feature
    )[0, 0]

    mid_acts_feature_scores = mid_acts.reshape(-1, model.cfg.d_model) @ feature_mid

    # Unembed
    feature_unembed = model.W_E @ feature_mid
    token_scores, token_strs = get_top_tokens(model.tokenizer, feature_unembed, k=n_tokens)

    # OV circuit
    ov_feature_head_unembed = model.W_E @ model.OV[layer, head] @ feature_mid
    ov_token_scores, ov_token_strs = get_top_tokens(model.tokenizer, ov_feature_head_unembed, k=n_tokens)

    # ln_OV
    # We use the token to index into the embedding matrix!
    ln_ov_feature_head_unembed = get_tangent_plane_at_point(
        model.W_E[data[sample_idx, token_idx].item()], lambda x: ln_ov(x, model, layer, head), feature_mid
    )
    ln_ov_token_scores, ln_ov_token_strs = get_top_tokens(model.tokenizer, ln_ov_feature_head_unembed, k=n_tokens)

    # QK
    qk_feature_head_unembed = model.W_E @ model.QK[layer, head] @ feature_mid
    qk_token_scores, qk_token_strs = get_top_tokens(model.tokenizer, qk_feature_head_unembed, k=n_tokens)

    return {
        "feature": feature,
        "sae activations": hidden_acts,
        "mid": feature_mid,
        "activation scores": mid_acts_feature_scores,
        "tokens": feature_unembed,
        "token scores": token_scores,
        "token strings": token_strs,
        "OV unembed": ov_feature_head_unembed,
        "OV token scores": ov_token_scores,
        "OV token strings": ov_token_strs,
        "ln+OV unembed": ln_ov_feature_head_unembed,
        "ln+OV token scores": ln_ov_token_scores,
        "ln+OV token strings": ln_ov_token_strs,
        "QK unembed": qk_feature_head_unembed,
        "QK token scores": qk_token_scores,
        "QK token strings": qk_token_strs,
    }


def get_feature_mid(
    data,
    sample_idx,
    feature_token_idx,
    feature_idx,
    use_ln=True,
    layer=0,
    mlp_out=True,
    model=None,
    encoder=None,
    feature=None,
):
    """Convenience function since we pull out this activation a lot"""
    return analyze_linearized_feature(
        feature_idx,
        sample_idx,
        feature_token_idx,
        layer=layer,
        model=model,
        data=data,
        mlp_out=mlp_out,
        use_ln=use_ln,
        encoder=encoder,
        feature=feature,
    )["mid"]


# def get_feature_mid_jacob(all_tokens, feature_example_idx, feature_token_idx, feature_post, use_ln=True, layer=0, mlp_out=True, model=None, encoder=None):
#     with torch.no_grad():
#         _, cache = model.run_with_cache(all_tokens[feature_example_idx], names_filter=[
#           utils.get_act_name("resid_mid", layer)
#         ])
#     mid_acts = cache[utils.get_act_name("resid_mid", layer)]
#     x_mid = mid_acts[0, feature_token_idx][None, None, :]

#     my_fun = (ln2_mlp_until_post if not mlp_out else ln2_mlp_until_out)
#     feature_mid = get_tangent_plane_at_point(x_mid,
#         lambda x: my_fun(x, model.blocks[layer].ln2, model.blocks[layer].mlp, use_ln=use_ln),
#         feature_post
#     )[0,0]
#     return feature_mid


# with torch.no_grad():
#     _, cache = model.run_with_cache(data, names_filter=[utils.get_act_name("resid_mid", layer)])
# mid_acts = cache[utils.get_act_name("resid_mid", layer)]
# x_mid = mid_acts[0, feature_token_idx][None, None, :]

# my_fun = ln2_mlp_until_post if not mlp_out else ln2_mlp_until_out
# feature_mid = get_tangent_plane_at_point(
#     x_mid, lambda x: my_fun(x, model.blocks[layer].ln2, model.blocks[layer].mlp, use_ln=use_ln), feature_post
# )[0, 0]
# return feature_mid
# result = analyze_linearized_feature(
