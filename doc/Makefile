# Makefile for the dc3 documentation, structured after PypeIt's.
#
# Read the Docs builds from the committed files and runs none of the generation
# steps below, so their products -- the API pages and the characterization
# figures -- are committed, and must be regenerated and committed when they
# change.
#
#   make html          regenerate the API pages, then build the HTML
#   make htmlonly      build the HTML from the files as they stand
#   make picky         as html, but treat every warning as an error, as Read the
#                      Docs does, and report every broken cross-reference
#   make apirst        regenerate the API pages only
#   make figures       regenerate every characterization figure (several minutes)
#   make resampling    regenerate one of them
#   make clean         remove the build and the generated API pages
#
#   make figures FORMAT=svg    write another format than the default

BUILDDIR      = _build
SPHINXBUILD   = sphinx-build
SPHINXAPI     = sphinx-apidoc
SPHINXAPIOPT  = members,private-members,undoc-members,show-inheritance
SPHINXOPTS    = -b html -d $(BUILDDIR)/doctrees -aE -w ./sphinx_warnings.out
LOCALFILES    = $(BUILDDIR)/* api/*.rst sphinx_warnings.out

PYTHON ?= python
FORMAT ?= png
FIGURES = figures/characterization
CHARACTERIZATIONS = pixelization matching resampling velscale preparation splices correction

.PHONY: help clean apirst html htmlonly picky figures $(CHARACTERIZATIONS)

help:
	@echo "Options are 'make <target>' where <target> is one of"
	@echo "  html       regenerate the API pages, then build the HTML"
	@echo "  htmlonly   build the HTML from the files as they stand"
	@echo "  picky      as html, but fail on any warning and report broken cross-references"
	@echo "  apirst     regenerate the API pages only"
	@echo "  figures    regenerate every characterization figure (several minutes)"
	@echo "  clean      remove the build and the generated API pages"

clean:
	rm -rf $(LOCALFILES)

apirst:
	SPHINX_APIDOC_OPTIONS=$(SPHINXAPIOPT) $(SPHINXAPI) --separate --force --no-toc -o ./api \
		../dc3 ../dc3/tests ../dc3/pkg/version.py

html:
	$(MAKE) apirst
	$(SPHINXBUILD) $(SPHINXOPTS) . $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

htmlonly:
	$(SPHINXBUILD) $(SPHINXOPTS) . $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

picky:
	$(MAKE) apirst
	$(SPHINXBUILD) $(SPHINXOPTS) -n -W --keep-going . $(BUILDDIR)/html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

figures: $(CHARACTERIZATIONS)

$(CHARACTERIZATIONS):
	MPLBACKEND=agg $(PYTHON) scripts/characterize_$@.py --outdir $(FIGURES) --format $(FORMAT)
