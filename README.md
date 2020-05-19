# Roosterize

Roosterize is a tool for suggesting lemma names in verification
projects that use the [Coq proof assistant](https://coq.inria.fr).

<b>We are actively updating this repository and will make it ready by
the end of May. Stay tuned!</b>

## Requirements

- [OCaml 4.07.1](https://ocaml.org)
- [SerAPI 0.7.1](https://github.com/ejgallego/coq-serapi)
- [Coq 8.10.2](https://coq.inria.fr/download)
- [Python 3.7+](https://www.python.org)
- [PyTorch 1.1.0](https://pytorch.org/get-started/previous-versions/#v110)

## Installation and usage

We strongly recommend installing the required versions of OCaml, Coq,
and SerAPI via the [OPAM package manager](https://opam.ocaml.org),
version 2.0.6 or later.  We recommend installing the required versions
of Python and PyTorch using [Conda](https://docs.conda.io/en/latest/).

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

Then, install PyTorch following the instructions
[here](https://pytorch.org/get-started/previous-versions/#v110), using
the correct installation command depending on your operating system,
Python package manager, and whether you want to use it on CPU or GPU.

Next, clone the Roosterize repository and enter the directory:
```
git clone https://github.com/EngineeringSoftware/roosterize.git
cd roosterize
```

To install other required Python libraries (`pip3` is included with
Python installation):
```
pip3 install -r requirements.txt
```

To use Roosterize for suggesting lemma names in a Coq verification
project using the pre-trained model (provided in this repository),
where $PROJECT_PATH is the path to the project:
```
python3 -m roosterize.main suggest_lemma_names --project_path=$PROJECT_PATH
```

For other usages and command line interfaces of Roosterize, please
check the manual page:
```
python3 -m roosterize.main help
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
and the corresponding elaborated term:
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
