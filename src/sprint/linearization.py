import torch
from transformer_lens import utils
from .loading import load_model, load_sae, load_data
from .vars import SAE_CFG


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


def get_top_tokens(tokenizer, vector, k=5, reverse=False):
    topk = torch.topk(vector, k=k, largest=(not reverse))
    return topk.values, tokenizer.batch_decode([[x] for x in topk.indices])


def get_activations(model, data, encoder, batch_size):
    if data is None:
        data = load_data(model=model)

    # Running model
    _, cache = model.run_with_cache(
        data[:batch_size],
        stop_at_layer=1,
        names_filter=[
            utils.get_act_name("post", 0),
            utils.get_act_name("resid_mid", 0),
            utils.get_act_name("attn_scores", 0),
        ],
    )
    mlp_acts = cache[utils.get_act_name("post", 0)]
    mlp_acts_flattened = mlp_acts.reshape(-1, SAE_CFG["d_mlp"])
    _, _, hidden_acts, _, _ = encoder(mlp_acts_flattened)

    return cache, hidden_acts


def analyze_linearized_feature(
    feature_idx, sample_idx, token_idx, model=None, encoder=None, data=None, batch_size=128, n_tokens=10
):
    # Get cache
    if model is None:
        model = load_model()
    if data is None:
        data = load_data(model=model)
    if encoder is None:
        encoder = load_sae()
    cache, hidden_acts = get_activations(model, data, encoder, batch_size)

    # Linearization component
    feature = encoder.W_enc[:, feature_idx]
    feature_domain = feature @ model.blocks[0].mlp.W_out

    mid_acts = cache[utils.get_act_name("resid_mid", 0)]
    x_mid = mid_acts[sample_idx, token_idx][None, None, :]
    feature_mid = get_tangent_plane_at_point(
        x_mid, lambda x: ln2_mlp_until_post(x, model.blocks[0].ln2, model.blocks[0].mlp), feature
    )[0, 0]

    mid_acts_feature_scores = mid_acts.reshape(-1, model.cfg.d_model) @ feature_mid

    # Unembed
    feature_unembed = model.W_E @ feature_mid
    token_scores, token_strs = get_top_tokens(model.tokenizer, feature_unembed, k=n_tokens)

    return {
        "feature": feature,
        "sae activations": hidden_acts,
        "domain": feature_domain,
        "mid": feature_mid,
        "activation scores": mid_acts_feature_scores,
        "tokens": feature_unembed,
        "token scores": token_scores,
        "token strings": token_strs,
    }
