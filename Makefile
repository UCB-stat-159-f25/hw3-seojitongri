# Makefile for HW3 - MyST and LIGO Project

env:
	conda env update --file environment.yml --prune || conda env create --file environment.yml

html:
	myst build --html

clean:
	rm -rf figures/* audio/* _build/*
