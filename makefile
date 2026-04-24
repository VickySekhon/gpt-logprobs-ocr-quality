PYTHON = python3
PIP = pip

TOP_K = 10
MAX_PAGES = 100
OUTPUT = results

.PHONY: install run-all figs clean

install:
	$(PIP) install -r requirements.txt

run-all:
	$(PYTHON) src/predict_quality.py --top-k $(TOP_K) --max-pages $(MAX_PAGES) --output $(OUTPUT)

figs:
	$(PYTHON) scripts/entropy_vs_cer.py --top-k $(TOP_K)
	$(PYTHON) scripts/roc_thresholds.py --top-k $(TOP_K)
	$(PYTHON) scripts/stratified_analysis.py --top-k $(TOP_K)

clean:
	rm -rf $(OUTPUT)/figures/*.png 
	rm -rf $(OUTPUT)/csv/*.csv