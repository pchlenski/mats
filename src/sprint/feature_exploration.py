import os
import torch

from .vars import DD
from .loading import load_model, load_data


def max_activating_examples(feature_id, n_examples, model=None, data=None, evenly_spaced=False):
    if model is None:
        model = load_model()
    if data is None:
        data = load_data(model=model)

    # Load feature vector
    features = torch.load(os.path.join(DD, "feature_vectors_merged", f"{feature_id}_full.pt")).to_dense()

    # Find the top n_examples examples that activate the feature the most
    if evenly_spaced:
        idx = torch.argsort(features.flatten(), descending=True)
        # Filter down to nonzero
        idx = idx[features.flatten()[idx] > 0]
        # Uniformly space out indices
        spacing = max(1, len(idx) // n_examples)
        idx = idx[::spacing][:n_examples]
    else:
        idx = torch.topk(features.flatten(), n_examples, dim=0).indices

    n = features.shape[1]

    r = idx // n
    c = idx % n

    out = data[r]

    return out, features[r], r, c
