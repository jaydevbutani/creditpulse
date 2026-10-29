# CreditPulse — reproducible pipeline (report §5). `make all` rebuilds everything from data/raw/*.csv.
PY ?= python3
export PYTHONWARNINGS=ignore

.PHONY: rebuild-data all data labels panel features baselines deep deep-post sensitivity evaluate explain batch report test serve dashboard docker clean

all: data labels panel features baselines deep deep-post sensitivity evaluate explain batch report test

# use the shipped trained models: rebuild only the deterministic data stages (~1 minute, byte-identical)
rebuild-data: data labels panel features

data:        ; $(PY) -m creditpulse.data.load
labels:      ; $(PY) -m creditpulse.labels.events
panel:       ; $(PY) -m creditpulse.features.panel
features:    ; $(PY) -m creditpulse.features.tokens
baselines:   ; $(PY) -m creditpulse.train.baselines
# two single-threaded workers share a 2-core CPU; set CP_THREADS/run sequentially on bigger machines
deep:
	CP_THREADS=1 $(PY) -m creditpulse.train.deep M6_multitask M3_gru & \
	CP_THREADS=1 $(PY) -m creditpulse.train.deep M5_transformer M4_lstm & wait
deep-post:   ; $(PY) -m creditpulse.train.deep --post
sensitivity: ; $(PY) -m creditpulse.train.sensitivity
evaluate:    ; $(PY) -m creditpulse.eval.evaluate
explain:     ; $(PY) -m creditpulse.explain.attrib
batch:       ; $(PY) -m creditpulse.serve.scoring
report:      ; $(PY) scripts/make_report.py
test:        ; $(PY) -m pytest -q

serve:       ; uvicorn creditpulse.serve.api:app --host 0.0.0.0 --port 8000
dashboard:   ; streamlit run app/dashboard.py --server.port 8501
docker:      ; docker build -t creditpulse . && docker run --rm -p 8000:8000 -p 8501:8501 creditpulse

clean:
	rm -rf data/interim/* data/processed/* artifacts/* reports/tables/* reports/figures/* reports/*.jsonl
