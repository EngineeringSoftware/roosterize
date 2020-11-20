INSTALL_DIR = $(HOME)/opt

PY_SRCS = preprocess.py train.py translate.py $(wildcard roosterize/*.py) $(wildcard onmt/*.py)


all: package

.PHONY: package
package: dist/roosterize/roosterize dist/roosterize.tgz

dist/roosterize/roosterize: roosterize.spec $(PY_SRCS)
	pyinstaller roosterize.spec -y --log-level WARN

dist/roosterize.tgz: dist/roosterize/roosterize
	cd dist && tar czf roosterize.tgz roosterize/

.PHONY: install
install: package
	rm -rf $(INSTALL_DIR)/roosterize
	mkdir -p $(INSTALL_DIR)/roosterize
	cp -r dist/roosterize $(INSTALL_DIR)/roosterize/bin

.PHONY: clean
clean:
	-rm -rf dist/ build/
