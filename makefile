ifeq ($(OS),Windows_NT)
    PYTHON = py -3
    PIP    = py -3 -m pip
else
    PYTHON = python3
    PIP    = python3 -m pip
endif

TOP_K = 10
MAX_PAGES = 100
THREADS = 20
OUTPUT = results

.PHONY: install run-all figs clean

install:
	$(PIP) install -r requirements.txt

run-all:
	$(PYTHON) -m src.predict_quality --top-k $(TOP_K) --max-pages $(MAX_PAGES) --output $(OUTPUT) --threads $(THREADS)

figs:
	$(PYTHON) -m scripts.entropy_vs_cer --top-k $(TOP_K)
	$(PYTHON) -m scripts.roc_thresholds --top-k $(TOP_K)
	$(PYTHON) -m scripts.stratified_analysis --top-k $(TOP_K)

clean:
	rm -rf $(OUTPUT)/figures/*.png 
	rm -rf $(OUTPUT)/csv/*.csv
	rm -rf $(OUTPUT)/tables/*.png