import os
from formula import calculate_cer

OCR_DIR = './input/ocr-text'
GT_DIR = './input/ground-truth'

ocr_files = sorted(os.listdir(OCR_DIR))
gt_files = sorted(os.listdir(GT_DIR))


for ocr_filename, gt_filename in zip(ocr_files, gt_files):
     if ocr_filename == "3200797029.txt" and gt_filename == "3200797029.txt":
          # Read content from OCR file
          with open(os.path.join(OCR_DIR, ocr_filename), "r", encoding="utf-8") as ocr_file:
               ocr_content = ocr_file.read()
          
          # Read content from ground-truth file
          with open(os.path.join(GT_DIR, gt_filename), "r", encoding="utf-8") as gt_file:
               gt_content = gt_file.read()
          
          
          cer = calculate_cer(gt_content, ocr_content)
          print(cer)
          
          # Compare characters
          # for ocr_char, gt_char in zip(ocr_content, gt_content):
          #      print(ocr_char, gt_char)