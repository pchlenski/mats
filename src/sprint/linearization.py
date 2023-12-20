import torch
from transformer_lens import utils
from .loading import load_model, load_sae, load_data
from .vars import SAE_CFG, BATCH_SIZE


def cossim(a, b):
    return (torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b))).item()


def get_tangent_plane_at_point(x_0_new, f, range_normal):
    # now, find the tangent hyperplane at x_0_new
    x_0_new.requires_grad_(True)
    g = lambda x: f(x) @ range_normal
    grad = torch.autograd.grad(g(x_0_new), x_0_new)
    return grad[0]


def ln2_mlp_until_post(x, ln, mlp):
    x = ln(x)
    x = x @ mlp.W_in + mlp.b_in
    x = mlp.act_fn(x)
    return x


def ln_ov(x, model, layer, head):
    return model.blocks[layer].ln1(x) @ model.OV[layer, head]


def get_top_tokens(tokenizer, vector, k=5, reverse=False):
    topk = torch.topk(vector, k=k, largest=(not reverse))
    return topk.values, tokenizer.batch_decode([[x] for x in topk.indices])


def analyze_linearized_feature(
    feature_idx,
    sample_idx,
    token_idx,
    layer=0,
    head=0,
    model=None,
    encoder=None,
    data=None,
    batch_size=BATCH_SIZE,
    n_tokens=10,
    encode="mid",  # "mlp_out"
):
    # Get cache
    if model is None:
        model = load_model()
    if data is None:
        data = load_data(model=model)
    if encoder is None:
        encoder = load_sae()

    # Get batch start and end
    idx_in_batch = sample_idx % batch_size
    batch_start = sample_idx - (idx_in_batch)
    batch_end = batch_start + batch_size

    # Get cache
    _, cache = model.run_with_cache(
        data[batch_start:batch_end],
        # stop_at_layer=1,
        names_filter=[
            utils.get_act_name("post", layer),
            utils.get_act_name("resid_mid", layer),
            utils.get_act_name("attn_scores", layer),
        ],
    )
    mlp_acts = cache[utils.get_act_name("post", layer)]
    mlp_acts_flattened = mlp_acts.reshape(-1, SAE_CFG["d_mlp"])
    _, _, hidden_acts, _, _ = encoder(mlp_acts_flattened)

    # Linearization component
    feature = encoder.W_enc[:, feature_idx]
    feature_domain = feature @ model.blocks[layer].mlp.W_out

    mid_acts = cache[utils.get_act_name("resid_mid", layer)]
    x_mid = mid_acts[idx_in_batch, token_idx][None, None, :]
    feature_mid = get_tangent_plane_at_point(
        x_mid, lambda x: ln2_mlp_until_post(x, model.blocks[layer].ln2, model.blocks[layer].mlp), feature
    )[0, 0]

    mid_acts_feature_scores = mid_acts.reshape(-1, model.cfg.d_model) @ feature_mid

    # Unembed
    feature_unembed = model.W_E @ feature_mid
    token_scores, token_strs = get_top_tokens(model.tokenizer, feature_unembed, k=n_tokens)

    # OV circuit
    ov_feature_head_unembed = model.W_E @ model.OV[layer, head] @ feature_mid
    ov_token_scores, ov_token_strs = get_top_tokens(model.tokenizer, ov_feature_head_unembed, k=n_tokens)

    # ln_OV
    ln_ov_feature_head_unembed = get_tangent_plane_at_point(
        model.W_E[model.to_single_token("('")], lambda x: ln_ov(x, model, layer, head), feature_mid
    )
    ln_ov_token_scores, ln_ov_token_strs = get_top_tokens(model.tokenizer, ln_ov_feature_head_unembed, k=n_tokens)

    # QK
    qk_feature_head_unembed = model.W_E @ model.QK[layer, head] @ feature_mid
    qk_token_scores, qk_token_strs = get_top_tokens(model.tokenizer, qk_feature_head_unembed, k=n_tokens)

    return {
        "feature": feature,
        "sae activations": hidden_acts,
        "domain": feature_domain,
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
