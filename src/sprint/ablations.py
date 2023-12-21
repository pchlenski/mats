import torch
from transformer_lens import utils
from .linearization import ln2_mlp_until_out


def attention_resampling_ablation(prompt_clean, prompt_corrupted, feature_idx, token_idx, model, sae):
    # Check that the tokens line up
    tokens_clean = model.to_tokens(prompt_clean).detach().cpu().numpy().flatten()
    tokens_corrupted = model.to_tokens(prompt_corrupted).detach().cpu().numpy().flatten()
    print("\t".join([model.to_string(t) for t in tokens_clean]))
    print("\t".join([model.to_string(t) for t in tokens_corrupted]))

    # Names
    pattern = utils.get_act_name("attn_out", 0)
    post = utils.get_act_name("post", 0)

    # Get the attention pattern
    _, cache_corrupt = model.run_with_cache(prompt_corrupted, names_filter=[pattern, post])
    _, cache_clean = model.run_with_cache(prompt_clean, names_filter=[pattern, post])

    # Definte the ablation hook
    def ablate_attn(activation, hook):
        print(hook.name)
        return cache_corrupt[pattern]

    def ablate_attn2(activation, hook):
        print(hook.name)
        return cache_clean[pattern]

    # Now patch in the corrupt attention pattern and get outputs
    with model.hooks(fwd_hooks=[(pattern, ablate_attn)]):
        _, cache_resampled = model.run_with_cache(prompt_clean, names_filter=[post])
    with model.hooks(fwd_hooks=[(pattern, ablate_attn2)]):
        _, cache_resampled2 = model.run_with_cache(prompt_corrupted, names_filter=[post])

    return [
        sae(cache[post][0, token_idx])[2].detach().cpu().numpy().flatten()[feature_idx]
        for cache in [cache_corrupt, cache_clean, cache_resampled, cache_resampled2]
    ]


@torch.no_grad()
def activation_steering(model, prompt, steering_info, measure_info, encoder=None):
    # steering_info: {'layer', 'act_name', 'token', 'vec'}
    # if use_sae: measure_info is like {'layer', 'act_name', 'token', 'feature_idx'}
    # otherwise, measure_info is like {'layer', 'act_name', 'token', 'vec'}
    def steer_hook_fn(activation, hook):
        new_activation = activation.clone()
        new_activation[0, steering_info["token"]] = activation[0, steering_info["token"]] + steering_info["vec"]
        return new_activation

    measure_act_str = utils.get_act_name(measure_info["act_name"], measure_info["layer"])
    steering_act_str = utils.get_act_name(steering_info["act_name"], measure_info["layer"])

    _, cache_unsteered = model.run_with_cache(prompt, names_filter=[steering_act_str, measure_act_str])
    with model.hooks(fwd_hooks=[(steering_act_str, steer_hook_fn)]):
        _, cache_steered = model.run_with_cache(prompt, names_filter=[steering_act_str, measure_act_str])

    if encoder is not None:
        _, _, unsteered_hidden_acts, _, _ = encoder(cache_unsteered[measure_act_str])
        _, _, steered_hidden_acts, _, _ = encoder(cache_steered[measure_act_str])
        unsteered_score = unsteered_hidden_acts[0, measure_info["token"], measure_info["feature_idx"]].item()
        steered_score = steered_hidden_acts[0, measure_info["token"], measure_info["feature_idx"]].item()
    else:
        unsteered_score = (cache_unsteered[measure_act_str][0, measure_info["token"]] @ measure_info["vec"]).item()
        steered_score = (cache_steered[measure_act_str][0, measure_info["token"]] @ measure_info["vec"]).item()
    return unsteered_score, steered_score


def quick_direct_patch(token, feature_idx, model, sae, layer=0):
    _, cache = model.run_with_cache("{'name': '")
    clean_act = cache[utils.get_act_name("resid_mid", layer)]
    dirty_act = clean_act.clone()
    dirty_act[0, -1, :] = (
        dirty_act[0, -1, :] - model.W_E[model.to_single_token(" '")] + model.W_E[model.to_single_token(token)]
    )
    _, _, hidden_clean, _, _ = sae(
        ln2_mlp_until_out(clean_act, model.blocks[layer].ln2, model.blocks[layer].mlp)[0, -1]
    )
    _, _, hidden_dirty, _, _ = sae(
        ln2_mlp_until_out(dirty_act, model.blocks[layer].ln2, model.blocks[layer].mlp)[0, -1]
    )
    return hidden_dirty[feature_idx] - hidden_clean[feature_idx]


@torch.no_grad()
def quick_direct_patch_interp(token, t, model, feature_idx, sae, layer=0):
    _, cache = model.run_with_cache("{'name': '")
    clean_act = cache[utils.get_act_name("resid_mid", layer)]
    dirty_act = clean_act.clone()
    dirty_act[0, -1, :] = (
        dirty_act[0, -1, :] - t * model.W_E[model.to_single_token(" '")] + t * model.W_E[model.to_single_token(token)]
    )
    _, _, hidden_clean, _, _ = sae(
        ln2_mlp_until_out(clean_act, model.blocks[layer].ln2, model.blocks[layer].mlp)[0, -1]
    )
    _, _, hidden_dirty, _, _ = sae(
        ln2_mlp_until_out(dirty_act, model.blocks[layer].ln2, model.blocks[layer].mlp)[0, -1]
    )
    return (hidden_dirty[feature_idx] - hidden_clean[feature_idx]).item()


@torch.no_grad()
def dla(model, input, layer, range_normal, dst_token=-1, mid=False):
    _, cache = model.run_with_cache(input)
    dlas = []
    print(cache[utils.get_act_name("attn_out", 0)].shape)
    dlas.append(((cache[utils.get_act_name("resid_pre", 0)][:, dst_token] @ range_normal).item(), ("resid_pre", 0)))
    for cur_layer in range(layer):
        dlas.append(
            (
                (cache[utils.get_act_name("attn_out", cur_layer)][:, dst_token] @ range_normal).item(),
                ("attn_out", cur_layer),
            )
        )
        dlas.append(
            (
                (cache[utils.get_act_name("mlp_out", cur_layer)][:, dst_token] @ range_normal).item(),
                ("mlp_out", cur_layer),
            )
        )

    if mid:
        dlas.append(
            ((cache[utils.get_act_name("attn_out", layer)][:, dst_token] @ range_normal).item(), ("attn_out", layer))
        )
    return dlas
