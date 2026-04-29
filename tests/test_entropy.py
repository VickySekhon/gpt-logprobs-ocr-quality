"""
Tests per-token entropy and entropy for all tokens of output in entropy.py by comparing it to results from scan2latex_entropy.py.
"""

import os
import numpy as np

from src.utils import (
    load_cache_json,
    load_text_pair,
    get_page_id_from_image,
    get_cache_key,
)
from src.entropy import token_entropies_from_logprobs
from src.scan2latex_entropy import calculate_entropy


def test_page():
    image_folder = os.path.join(os.getcwd(), "data/images")
    page_id = np.array(
        [get_page_id_from_image(image) for image in os.listdir(image_folder)]
    )[0]
    
    if not page_id:
        return

    cache = load_cache_json()
    cache_key = get_cache_key(page_id, "gpt_4o", 10, 1)
    
    try:
        token_logprobs = cache[cache_key]["token_logprobs"]
    except KeyError as e:
        print(f"Key doesn't exist in Cache, run 'make run-all'")
        raise e
     
    x1 = token_entropies_from_logprobs(token_logprobs)
    x1 = (sum(x1), sum(x1) / len(x1))
    # Positional entropies are not needed remove them
    x2 = calculate_entropy(token_logprobs, len(token_logprobs), 10)[:-1]

    assert (
        x1 == x2
    ), f"Entropy does not match. entropy.py produced: {x1} whereas scan2latex_entropy.py produced {x2}"


if __name__ == "__main__":
    test_page()
