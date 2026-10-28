PYTHON ?= python

.PHONY: build clean
build:
	$(PYTHON) -m build --no-isolation --sdist --wheel

clean:
	rm -rf build dist src/*.egg-info
