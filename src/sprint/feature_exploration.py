import os
import torch

from .vars import DD
from .loading import load_model, load_data


def max_activating_examples(feature_id, n_examples, model=None, data=None, return_feature_values=False):
    if model is None:
        model = load_model()
    if data is None:
        data = load_data(model=model)

    # Load feature vector
    features = torch.load(os.path.join(DD, "feature_vectors_merged", f"{feature_id}_full.pt")).to_dense()

    # Find the top n_examples examples that activate the feature the most
    top_examples = torch.topk(features.flatten(), n_examples, dim=0)
    n = features.shape[1]

    r = top_examples.indices // n
    c = top_examples.indices % n

    if return_feature_values:
        return data[r], features[r]
    else:
        return data[r]
