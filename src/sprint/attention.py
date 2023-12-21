import torch

from transformer_lens import utils

from .vars import BATCH_SIZE


def _validate_cache(cache, data, model, batch_size=BATCH_SIZE):
    # Get cache
    if cache is None:
        if data is not None:
            _, cache = model.run_with_cache(
                data[:batch_size],
                stop_at_layer=1,
                names_filter=[
                    utils.get_act_name("pattern", 0),
                    utils.get_act_name("v", 0),
                ],
            )
        else:
            raise ValueError("Either cache or data must be provided.")
    return cache


def get_attn_head_contribs(model, layer, range_normal, cache=None, data=None, batch_size=BATCH_SIZE, use_half=True):
    cache = _validate_cache(cache, data, model, batch_size=batch_size)
    split_vals = cache[utils.get_act_name("v", layer)]
    split_vals = split_vals.to(torch.float16) if use_half else split_vals
    attn_pattern = cache[utils.get_act_name("pattern", layer)]
    attn_pattern = attn_pattern.to(torch.float16) if use_half else attn_pattern

    #'batch head dst src, batch src head d_head -> batch head dst src d_head'
    weighted_vals = torch.einsum("b h d s, b s h f -> b h d s f", attn_pattern, split_vals)

    # 'batch head dst src d_head, head d_head d_model -> batch head dst src d_model'
    weighted_outs = torch.einsum("b h d s f, h f m -> b h d s m", weighted_vals, model.W_O[layer])

    # 'batch head dst src d_model, d_model -> batch head dst src'
    contribs = torch.einsum("b h d s m, m -> b h d s", weighted_outs, range_normal)

    return contribs


def get_attn_head_contribs_ov(model, layer, range_normal, cache=None, data=None, batch_size=BATCH_SIZE, use_half=True):
    cache = _validate_cache(cache, data, model, batch_size=batch_size)
    split_vals = cache[utils.get_act_name("v", layer)]
    split_vals = split_vals.to(torch.float16) if use_half else split_vals
    # print(split_vals.shape)

    # 'batch src head d_head, head d_head d_model -> batch head src d_model'
    weighted_outs = torch.einsum("b s h f, h f m -> b h s m", split_vals, model.W_O[layer])

    # 'batch head src d_model, d_model -> batch head src'
    contribs = torch.einsum("b h s m, m -> b h s", weighted_outs, range_normal)

    return contribs


def do_single_token_qk(dst_token_str, src_token_str, head, dst_pos=40, src_pos=40, model=None):
    max_pos_list = []
    query = model.W_E[model.to_single_token(dst_token_str)] + model.W_pos[dst_pos]
    query = model.blocks[0].ln1(query)
    query = query @ model.blocks[0].attn.W_Q[head]
    query = query + model.blocks[0].attn.b_Q[head]

    key = model.W_pos[src_pos] + model.W_E[model.to_single_token(src_token_str)]
    key = model.blocks[0].ln1(key)
    key = key @ model.blocks[0].attn.W_K[head]
    key = key + model.blocks[0].attn.b_K[head]

    test_qk = query @ key.T
    return test_qk.item()
