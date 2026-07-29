SPHINXOPTS    ?= --fail-on-warning --keep-going
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

.PHONY: Makefile help

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

serve:
	@rm -rf "$(BUILDDIR)"
	uv run --all-packages --with sphinx-autobuild sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" --re-ignore '$(BUILDDIR)/.*' --watch _theme --watch _ext
