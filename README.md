# Roosterize

Roosterize is a tool for suggesting lemma names in verification
projects that use the [Coq proof assistant](https://coq.inria.fr).

## Requirements

- [OCaml 4.07.1](https://ocaml.org)
- [SerAPI 0.7.1](https://github.com/ejgallego/coq-serapi)
- [Coq 8.10.2](https://coq.inria.fr/download)
- [Python 3.7.7+](https://www.python.org)
- [PyTorch 1.1.0](https://pytorch.org/get-started/previous-versions/#v110)

## Installation and usage

### Installation of OCaml, Coq, and SerAPI

We strongly recommend installing the required versions of OCaml, Coq,
and SerAPI via the [OPAM package manager](https://opam.ocaml.org),
version 2.0.6 or later.

To set up the OPAM-based OCaml environment, use:
```
opam switch create 4.07.1
opam switch 4.07.1
eval $(opam env)
```
Then, install Coq and SerAPI, pinning them to avoid unintended upgrades:
```
opam update
opam pin add coq 8.10.2
opam pin add coq-serapi 8.10.0+0.7.1
```

### Installation of Python and PyTorch

We recommend installing the required versions of Python and PyTorch
using [Conda](https://docs.conda.io/en/latest/miniconda.html).

Then, install PyTorch by following the instructions
[here](https://pytorch.org/get-started/previous-versions/#v110), using
the installation command suitable for your operating system, Python
package manager, and whether you want to use it on a CPU or GPU.

Next, clone the Roosterize repository and enter the directory:
```
git clone https://github.com/EngineeringSoftware/roosterize.git
cd roosterize
```

To install other required Python libraries (`pip` is included with
Python installation):
```
pip install -r requirements.txt
```

### Download pre-trained models

Roosterize's pre-trained models are available via this Google Drive
[link](https://drive.google.com/file/d/1L0-BMOrP0WYX7L1bAhKRkJPLm7VPeMsE/view?usp=sharing).
Download the file `models.tgz` from the link and put it under this
directory, then uncompress the file:
```
tar xzf models.tgz
```

### Usage

To use Roosterize on a Coq verification project, you need to first
build and install the project with the command provided by the project
(usually `make` followed by `make install`). Then, at the root directory of
the project repository, run the command
```
python -m roosterize.main suggest_lemmas --project=$PATH_TO_PROJECT --serapi_options=$SERAPI_OPTIONS --model-dir=./models/roosterize-ta --output=./output
```
where `$PATH_TO_PROJECT` should be replaced with the path to the
project, and `$SERAPI_OPTIONS` should be replaced with the SerAPI
command line options for mapping logical paths to directories (see [SerAPI's
documentation](https://github.com/ejgallego/coq-serapi/blob/v8.11/FAQ.md#does-serapi-support-coqs-command-line-flags)).
For example, if the logical path (inside Coq) for the project is `Verified`,
you should set `SERAPI_OPTIONS="-Q $PATH_TO_PROJECT,Verified"`.

The command extracts all lemmas from the project, uses Roosterize's
pre-trained model (at `./models/roosterize-ta`) to predict a lemma name
for each lemma, and finally prints the lemma name update suggestions,
i.e., the predicted lemma names that are different from to the existing ones.
Below is an example of printed suggestions:
```
>>>>> Suggestions:
infotheo/ecc_classic/bch.v: infotheo.ecc_classic.bch.BCH.BCH_PCM_altP1 -> inde_F2
infotheo/ecc_classic/bch.v: infotheo.ecc_classic.bch.BCH.BCH_PCM_altP2 -> inde_mul
infotheo/ecc_classic/bch.v: infotheo.ecc_classic.bch.BCH.PCM_altP -> F2_eq0
infotheo/ecc_classic/bch.v: infotheo.ecc_classic.bch.BCH.PCM_alt_GRS -> P
infotheo/ecc_classic/bch.v: infotheo.ecc_classic.bch.BCH_codebook -> map_P
...
```

For other usages and command line interfaces of Roosterize, please
check the help:
```
python -m roosterize.main help
```

## Technique

Roosterize learns and suggests lemma names using neural networks
that take serialized Coq lemma statements and elaborated terms as input.
For example, the Coq lemma sentence
```coq
Lemma mg_eq_proof L1 L2 (N1 : mgClassifier L1) : L1 =i L2 -> nerode L2 N1.
```
is serialized into the following tokens:
```lisp
(Sentence((IDENT Lemma)(IDENT mg_eq_proof)(IDENT L1)(IDENT L2)
  (KEYWORD"(")(IDENT N1)(KEYWORD :)(IDENT mgClassifier)
  (IDENT L1)(KEYWORD")")(KEYWORD :)(IDENT L1)(KEYWORD =i)(IDENT L2)
  (KEYWORD ->)(IDENT nerode)(IDENT L2)(IDENT N1)(KEYWORD .)))
```
and the corresponding elaborated term (simplified):
```lisp
(Prod (Name (Id char)) ... (Prod (Name (Id L1)) ...
 (Prod (Name (Id L2)) ... (Prod (Name (Id N1)) ...
  (Prod Anonymous (App (Ref (DirPath ((Id ssrbool) (Id ssr) (Id Coq))) (Id eq_mem)) ...
   (Var (Id L1)) ... (Var (Id L2)))
  (App (Ref (DirPath ((Id myhill_nerode) (Id RegLang))) (Id nerode)) ...
   (Var (Id L2)) ... (Var (Id N1))))))))
```

The diagram below illustrates Roosterize's neural network
architecture, as applied to this example:

<img src="seqtoseq-arch.svg" width="700" title="Roosterize architecture">

Our [research paper][arxiv-paper] outlines the design of Roosterize,
and describes an evaluation on a [corpus][math-comp-corpus]
of serialized Coq code derived from the [Mathematical Components][math-comp-website]
family of projects.

If you have used Roosterize in a research project, please cite
the research paper in any related publication:
```bibtex
@inproceedings{NieETAL20Roosterize,
  author = {Nie, Pengyu and Palmskog, Karl and Li, Junyi Jessy and Gligoric, Milos},
  title = {Deep Generation of {Coq} Lemma Names Using Elaborated Terms},
  booktitle = {International Joint Conference on Automated Reasoning},
  pages = {To appear},
  year = {2020},
}
```

[arxiv-paper]: https://arxiv.org/abs/2004.07761
[math-comp-corpus]: https://github.com/EngineeringSoftware/math-comp-corpus
[math-comp-website]: https://math-comp.github.io

## Authors

- [Pengyu Nie](https://cozy.ece.utexas.edu/~pynie/)
- [Karl Palmskog](https://setoid.com)
- [Emilio Jesús Gallego Arias](https://www.irif.fr/~gallego/)
- [Junyi Jessy Li](http://jessyli.com)
- [Milos Gligoric](http://users.ece.utexas.edu/~gligoric/)
