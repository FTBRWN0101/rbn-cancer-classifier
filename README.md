# Classifying ovarian cancer from serum proteomics with random Boolean network attractors

Each patient's serum mass spectrum sets the starting state of a random Boolean network, and a classifier learns from how the network settles: the attractor it reaches and the path it takes there.
The code asks whether those dynamics separate ovarian cancer from normal samples as well as a random forest trained on the measurements directly.

Built for an MSc module at the University of Sussex (2026).

## What it found

On the 77 held-out test samples, the RBN classifier reached an accuracy of 0.935 and a ROC AUC of 0.985.
It flagged 46 of the 49 cancer samples as cancer, and 2 of the 28 normal samples.
A random forest trained on the same 30 binary features reached 0.961 and 0.999, flagging 47 of the 49 cancer samples and 1 of the 28 normal samples.
The network's 12 dynamical features came close to the baseline without matching it.
The RBN classifier did not hold up under measurement noise: flipping 5%, 10% and 20% of its binary inputs cut its accuracy to 0.883, 0.766 and 0.688.

## Run it

Install [uv](https://docs.astral.sh/uv/), then run:

```bash
uv sync
uv run python scripts/download_data.py
uv run python src
```

The download script fetches the dataset (36 MB), checks it against pinned hashes and writes `data/raw_data/Ovarian.arff`.
The pipeline then runs in under a minute.
It prepares the data, selects 30 features, trains and scores the RBN classifier and the random forest, and tests noise robustness.
It finishes with two standalone demos, of a random Boolean network and of an NK fitness landscape.
It writes processed arrays to `data/processed/`, and metrics and figures to `results/`, which git ignores.

The code labels Cancer as class 0 and Normal as class 1.
The F1 score it prints, and the decision threshold it tunes, therefore refer to the Normal class: the RBN classifier's F1 is 0.912 for Normal and 0.948 for Cancer.
The printed classification report lists both classes.

Expect small differences from the numbers above.
Feature selection ranks binary features, and thousands of them tie exactly, so numpy's tie order picks the last few of the 30, and that order can vary between machines.
On a Windows PC, 27 of the 30 features matched the original run, and the RBN classifier reached a ROC AUC of 0.998.

## How the code is laid out

- `src/data/`: loads the ARFF, scales each feature, makes a stratified 70/30 split and binarises each feature at its training median.
- `src/feature_selection/`: ranks features by mutual information, Fisher score and the difference in class means, and keeps the best 30.
- `src/rbn/`: the random Boolean network, its simulation, attractor detection and the 12 trajectory features.
- `src/evaluation/`: the attractor classifier, the random forest baseline and the noise test.
- `src/nk_landscape/`: an NK fitness landscape with simulated annealing, a demo that is separate from the classifier.
- `src/utils/`: plots, saved metrics and project paths.
- `scripts/download_data.py`: fetches and checks the dataset.
- [METHODS.md](METHODS.md): each stage in detail, its parameters and its caveats.

## Test it

```bash
uv run pytest
```

The tests run the whole pipeline on a small made-up dataset, so they need no download.
They also check the loader's label coding, and that the download script refuses a file whose hash does not match.

Tested on Linux and Windows with Python 3.12.

## Data

The data are the NCI's 6-19-02 ovarian cancer serum proteomics dataset, which the Kent Ridge repository cites to Petricoin et al. (2002).
It holds 253 serum samples, 162 from ovarian cancer patients and 91 from controls.
Each sample is a SELDI-TOF mass spectrum of 15,154 mass-to-charge (M/Z) intensities.
The code calls each M/Z value a gene, after the gene-regulatory framing of random Boolean networks, but the measurements are protein mass spectra, not gene expression.

`scripts/download_data.py` fetches the copy held by the Kent Ridge Bio-medical Data Set Repository, from the Elvira project at leo.ugr.es.
If that site is down, it uses the Internet Archive's copy of the same file.
It rebuilds the ARFF the original run used, byte for byte, checks it against a pinned SHA-256 hash, and keeps nothing that fails.
Kent Ridge scaled every intensity to [0, 1] over all 253 samples, so the code reads normalised values.
The NCI no longer publishes the raw 6-19-02 spectra.
The data's own terms of use apply.

## Write-up

The full write-up isn't published here.
If you'd like to read it, email me at ftbrwn01@gmail.com.

## Licence

The code is under the Apache License 2.0; see [LICENSE](LICENSE).
Copyright 2026 Findlay Brown.

## Citation

To cite this code, use **Cite this repository** on the GitHub page, which reads [CITATION.cff](CITATION.cff).
The work builds on:

- Kauffman, S. A. (1969). Metabolic stability and epigenesis in randomly constructed genetic nets. *Journal of Theoretical Biology* 22(3): 437-467.
- Kauffman, S. A. and Weinberger, E. D. (1989). The NK model of rugged fitness landscapes and its application to maturation of the immune response. *Journal of Theoretical Biology* 141(2): 211-245.
- Kirkpatrick, S., Gelatt, C. D. and Vecchi, M. P. (1983). Optimization by simulated annealing. *Science* 220(4598): 671-680.
- Petricoin, E. F. et al. (2002). Use of proteomic patterns in serum to identify ovarian cancer. *The Lancet* 359(9306): 572-577. https://doi.org/10.1016/S0140-6736(02)07746-2
