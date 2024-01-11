import torch

from datasets import load_dataset
from transformer_lens import HookedTransformer, utils
from transformer_lens.utils import tokenize_and_concatenate

from .sae_tutorial import AutoEncoder
from .vars import MODEL, RUN, DATASET


def load_model(
    model_name: str = MODEL,
    # fold_ln=False,
    fold_ln: bool = True,
    use_cuda: bool = True,
    half_precision: bool = True,
    verbose: bool = True,
    **kwargs,
) -> HookedTransformer:
    model = HookedTransformer.from_pretrained(model_name, fold_ln=fold_ln)
    model = model.cuda() if use_cuda else model
    model = model.to(torch.float16) if half_precision else model
    print(f"Model device: {next(model.parameters()).device}") if verbose else None
    return model


def load_data(
    model: HookedTransformer,
    dataset_name: str = DATASET,
    split: str = "train",
    max_length: int = 128,
    seed: int = 42,
    use_cuda: bool = True,
    verbose: bool = True,
    **kwargs,
) -> torch.Tensor:
    data = load_dataset(dataset_name, split=split)
    # If dataset name does not have "tokenized" in it, tokenize it:
    if "tokenized" not in dataset_name:
        tokenized_data = tokenize_and_concatenate(data, model.tokenizer, max_length=max_length).shuffle(seed)
        tokens = tokenized_data["tokens"]
    else:
        tokens = data.shuffle(seed)
    tokens = tokens.cuda() if use_cuda else tokens
    print(f"Tokens shape: {tokens.shape}, dtype: {tokens.dtype}, device: {tokens.device}") if verbose else None
    return tokens


def load_sae(
    run_id: str = RUN, use_cuda: bool = True, half_precision: bool = True, verbose: bool = True, **kwargs
) -> AutoEncoder:
    encoder = AutoEncoder.load_from_hf(run_id)
    encoder = encoder.cuda() if use_cuda else encoder
    encoder = encoder.to(torch.float16) if half_precision else encoder
    print(f"Encoder device: {next(encoder.parameters()).device}") if verbose else None
    return encoder


def load_all(**kwargs) -> (HookedTransformer, torch.Tensor, AutoEncoder):
    model = load_model(**kwargs)
    tokens = load_data(model=model, **kwargs)
    sae = load_sae(**kwargs)
    return model, tokens, sae
