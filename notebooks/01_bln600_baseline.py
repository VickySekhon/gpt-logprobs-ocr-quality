import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.loader import *
from src.normalization import normalize_text_all
from src.metrics import cer

PAGES_TO_LOAD = 20

df = load_bln600_metadata()
example_printed = False

y_cer = []
x_page_id = []
for i in range(PAGES_TO_LOAD):
     page_id = df.iloc[i]["page-id"]
     ocr_s, gt_s = load_text_pair(page_id)
     
     before_ocr, before_gt = ocr_s, gt_s
     ocr, gt = normalize_text_all(before_ocr, before_gt)
     
     if not example_printed:
          print(f"Before normalization ocr: {before_ocr}\nBefore normalization gt: {before_gt}")
          print(f"After normalization ocr: {ocr}\nAfter normalization gt: {gt}")
          example_printed = True
          
     _cer = cer(ocr, gt)
     x_page_id.append(page_id)
     y_cer.append(_cer)
     
     print(f"Page ID: {page_id} has a CER of {_cer} between ocr and gt texts.")

plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_cer)), y_cer, alpha=0.8)
plt.plot(range(len(y_cer)), y_cer, alpha=0.4)
plt.xticks(range(len(x_page_id)), x_page_id, rotation=45, ha="right")
plt.ylabel("CER")
plt.xlabel("Page ID")
plt.ylim(0, max(y_cer) * 1.1 if y_cer else 1)
plt.title("Character Error Rate (CER) for 20 Page Excerpts")
plt.tight_layout()
plt.show()