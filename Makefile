PY ?= python
DATA ?= lerobot/svla_so101_pickplace
RUN ?= outputs/main

.PHONY: install install-robot lint test train eval bench plot

install:
	$(PY) -m pip install -e ".[plots,dev]"

install-robot:
	$(PY) -m pip install -e ".[robot,plots,dev]"

lint:
	ruff check so101 tests

test:
	pytest

train:
	$(PY) -m so101.train --output-dir $(RUN) --data.repo-id $(DATA)

eval:
	$(PY) -m so101.eval_offline lerobot/smolvla_base $(RUN)/final --repo-id $(DATA) --out results/eval_offline.json

bench:
	$(PY) -m so101.bench --checkpoint $(RUN)/final --repo-id $(DATA) --out results/bench.json

plot:
	$(PY) -m so101.plots --run-dir $(RUN) --eval-json results/eval_offline.json --traj-json results/trajectory.json
