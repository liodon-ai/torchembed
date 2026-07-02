.PHONY: docs docs-serve test lint format

MODULES = torchembed torchembed.positional torchembed.fourier \
          torchembed.categorical torchembed.patch torchembed.temporal \
          torchembed._triton

docs:
	pdoc -o docs/api -t docs/dracula --docformat google $(MODULES)

docs-serve:
	pdoc -t docs/dracula --docformat google $(MODULES)

test:
	python -m pytest tests/ -v

lint:
	ruff check torchembed/ tests/

format:
	ruff format torchembed/ tests/
