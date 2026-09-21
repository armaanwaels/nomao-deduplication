# nomao-deduplication

Place deduplication on the Nomao dataset: given two records describing businesses or places, decide whether they are the same place. I compared seven models on it, from logistic regression to a pretrained tabular foundation model. This was my final project for COMP 432 (Machine Learning) at Concordia University.

## Task and data

The [Nomao dataset](https://archive.ics.uci.edu/dataset/227/nomao) (UCI #227, also OpenML 1486) has 34,465 pairs of place records. Each pair is described by 118 similarity features: name, address, phone and GPS comparisons from several sources. The label says whether the pair is the same place. 71.4% of pairs are duplicates, so the positive class is the majority.

The notebook uses a stratified 80/20 train/test split with seed 42 (27,572 train, 6,893 test). Hyperparameters were tuned with 5-fold cross-validation on the training set, optimizing F1. Every number below is on the held-out test set.

## Results

Produced by `make reproduce`, which reruns every cell of `notebook.ipynb` and prints this table from the executed copy (12 minutes on an Apple M3 Pro, CPU only):

| Model | F1 | ROC AUC | Precision | Recall |
|---|---|---|---|---|
| Random Forest | 0.9772 | 0.9948 | 0.9774 | 0.9771 |
| KNN | 0.9728 | 0.9848 | 0.9711 | 0.9746 |
| SVM (RBF) | 0.9724 | 0.9915 | 0.9854 | 0.9598 |
| Feedforward NN (PyTorch) | 0.9672 | 0.9911 | 0.9859 | 0.9492 |
| TabNet | 0.9642 | 0.9907 | 0.9841 | 0.9450 |
| TabPFN v1 | 0.9594 | 0.9846 | 0.9564 | 0.9624 |
| Logistic Regression | 0.9564 | 0.9868 | 0.9779 | 0.9358 |

The outputs saved in the committed notebook came from a Colab run. They match this rerun exactly for KNN, SVM and TabPFN, and are within 0.0007 F1 for the other four. The small drift comes from different library versions and hardware.

TabPFN v1 can only take 1,000 training rows and 100 features, so it was trained on a stratified 1,000-row sample with the top 100 features by mutual information. A random forest trained on that same sample scored 0.9588 F1, against 0.9594 for TabPFN.

## Run it

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make reproduce
```

`requirements.txt` pins the versions the rerun used. The notebook installs TabPFN v1 itself partway through, and `make reproduce` sets `PIP_CONSTRAINT` so that install stays on the pinned versions. TabPFN v1 breaks on scikit-learn 1.8 and later.

To make the notebook run outside Colab I removed one import: `tabpfn` was imported at the top, before the torch compatibility patch that TabPFN v1 needs. It is now imported only in the TabPFN section, after the patch. No other code changed.

## What I learned

On well-engineered pairwise similarity features, a tuned random forest beat every neural model I tried, including TabNet. TabPFN with no tuning matched a random forest trained on the same 1,000 rows, which makes it a strong baseline when labels are scarce but not a substitute for the full training set.
