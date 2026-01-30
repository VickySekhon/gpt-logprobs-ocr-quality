from Levenshtein import distance
from math import log
from torchmetrics.text import CharErrorRate
import torch
import numpy as np

def calculate_cer(ground_truth, prediction):
    """Calculate Character Error Rate using Levenshtein distance."""
    edit_distance = distance(ground_truth, prediction)
    cer = edit_distance / len("".join(ground_truth.split(" ")))
    return f"{cer*100:.2f}%"

def _calculate_shannon_entropy(probability):
    """Calculate Shannon entropy for a single probability."""
    return -(probability * log(probability, 2))

def calculate_average_shannon_entropy(probabilities: list[float], top_k=None):
    """Calculate average Shannon entropy across probabilities."""
    probabilities = sorted(probabilities, reverse=True)
    if top_k is not None:
        probabilities = probabilities[:top_k]
    return sum(_calculate_shannon_entropy(p) for p in probabilities)

def max_entropy(probabilities):
    """Calculate maximum entropy for given number of outcomes."""
    n = len(probabilities)
    return log(n, 2)

def calculate_normalized_entropy(probabilities: list[float], top_k=None):
    """Calculate normalized entropy (0-1 scale)."""
    actual_e = calculate_average_shannon_entropy(probabilities, top_k)
    max_e = max_entropy(probabilities)
    return actual_e / max_e if max_e > 0 else 0

if __name__ == "__main__":
     uncertain_probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
     certain_probabilities = [0.1, 0.9]
     shannon_entropy_uc = calculate_average_shannon_entropy(uncertain_probabilities)
     shannon_entropy_c = calculate_average_shannon_entropy(certain_probabilities)
     entropy_m = max_entropy(uncertain_probabilities)
     entropy_n = calculate_normalized_entropy(uncertain_probabilities)
     print(f"Shannon entropy: uncertain is {shannon_entropy_uc:.2f} certain is {shannon_entropy_c:.2f}")
     print(f"Max entropy: {max_entropy}")
     print(f"Normalized entropy: {entropy_n}")


     # Example usage
     # ground_truth = "hello there my name is vicky sekhon and I like to code sometimes and make cool programs of software on my computer that allow me to think like an engineer and build robust while yet elegant and very reliable software systems that companies can trust and use."
     # prediction = "hello there my name is vic sekhon and I lke to code sometime and make cool programs of software on my computer that allow me to think like an engineyr and build robust while yet elegant and very reliable software systems that compynies can trust and use"
     ground_truth = "machine learning"
     prediction = "machin lerning"
     cer = calculate_cer(ground_truth, prediction)
     print(f"CER: {cer*100:.2f}%")  # Output: CER: 0.133


     # Initialize metric
     cer_metric = CharErrorRate()

     # Calculate CER for batches of predictions
     references = ["machine learning"]
     hypotheses = ["machin lerning"]

     # Convert to tensor format
     cer = cer_metric(hypotheses, references)
     print(f"Batch CER: {cer.item():.3f}")   