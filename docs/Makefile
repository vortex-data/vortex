# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?= --fail-on-warning --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: Makefile help clean rust-html

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

python-and-rust-html: html rust-html
	true  # need a non-empty rule to prevent matching the %: rule.

serve:
	mkdir -p _build/vortex
	-ln -s ../html _build/vortex/docs  # makes absolute links like /vortex/docs/rust/html work correctly
	echo The docs are served at http://localhost:8000/vortex/docs/
	(cd _build/ && python3 -m http.server)

watch:
	fswatch -o -e '#[^#]*#' -e '__pycache__' -e '\.#.*' -e target -e .git -e .venv -e pyvortex/python/vortex/_lib.abi3.so -e docs/_build -e docs/a.vortex ../ | xargs -L1 /bin/bash -c 'make python-and-rust-html'

rust-html:
	RUSTDOCFLAGS="--enable-index-page -Z unstable-options" cargo doc \
		--no-deps \
		--workspace \
		--exclude bench-vortex \
		--exclude xtask \
		--all-features \
		--target-dir \
		$(BUILDDIR)/html/rust
