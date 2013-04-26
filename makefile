all:
	git submodule update --init
	cd gco ; make ; make library
	cd matsci ; python setup.py build_ext --inplace