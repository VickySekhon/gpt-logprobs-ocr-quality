from rapidfuzz.distance import Levenshtein

""" 
Calculates CER between OCR and GT.
"""

def _levenshtein_distance(ocr, ground_truth):
     return Levenshtein.distance(ocr, ground_truth)

def cer(ocr, ground_truth):
     assert len(ground_truth) != 0, "Cannot compute CER for empty GT text."
     edits_needed = _levenshtein_distance(ocr, ground_truth)
     return edits_needed / len(ground_truth)