.PHONY: reproduce

# Reruns every cell. Took 12 minutes on an Apple M3 Pro (CPU only).
# PIP_CONSTRAINT keeps the notebook's own pip installs on the pinned versions.
reproduce:
	grep '==' requirements.txt > .constraints.txt
	PIP_CONSTRAINT=.constraints.txt jupyter nbconvert --to notebook --execute \
		--ExecutePreprocessor.timeout=-1 --output notebook.executed.ipynb notebook.ipynb
	python scripts/results_table.py notebook.executed.ipynb
