# Independent-Study-2026

#  DNA Disease Predictor

A bioinformatics pipeline that reads a DNA sequence, translates it into a protein, and uses a neural network built from scratch to predict the pathogenicity of known mutations linked to that gene.

---

## Overview

This project bridges molecular biology and machine learning. Given a DNA sequence in FASTA format, the pipeline:

1. Translates the DNA into a protein sequence using the standard codon table
2. Queries the ClinVar database for known disease-linked mutations in that gene
3. Extracts physicochemical features from each mutation
4. Runs those features through a neural network trained from scratch to predict disease risk

Everything is built low-level — no scikit-learn, no TensorFlow. Just Python and NumPy.

---

## Background

### The Biology

DNA encodes proteins through a two-step process:

```
DNA → (transcription) → mRNA → (translation) → Protein
```

The translation step works by reading the DNA in groups of three nucleotides called **codons**. Each codon maps to a specific amino acid. A sequence of amino acids folds into a protein. Mutations — single nucleotide changes — can alter which amino acid is produced, sometimes disrupting the protein's function and causing disease.

### The Machine Learning

The model is trained on data from **ClinVar**, a public database maintained by NCBI that catalogs hundreds of thousands of genetic variants and their clinical significance (Pathogenic, Benign, Uncertain). Each mutation is converted into a numeric feature vector based on the physicochemical properties of the amino acid change, then fed into a two-layer neural network trained with backpropagation.

---

## Data Sources

### DNA Sequences — NCBI GenBank

Download FASTA files from [ncbi.nlm.nih.gov/nuccore](https://www.ncbi.nlm.nih.gov/nuccore):

1. Search for a gene (e.g. `insulin human`, `BRCA1`, `TP53`)
2. Click a result → **Send to** → **File** → Format: **FASTA** → **Create File**

A FASTA file looks like:
```
>NM_000207.3 Homo sapiens insulin (INS), mRNA
ATGGCCCTGTGGATGCGCCTCCTGCCCCTGCTGGCGCTGCTGGCCCTCTGGGGACCT...
```

### Mutation Database — ClinVar

Download `variant_summary.txt.gz` from the [ClinVar FTP server](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/) and unzip it:

```bash
gunzip variant_summary.txt.gz
```

This file contains ~500,000 variants with their clinical significance, associated gene, and linked diseases.

---

## Project Structure

```
Final/
├── main.py              # Entry point — runs the full pipeline
├── translator.py        # DNA → protein translation
├── load_clinvar.py      # Parses ClinVar database into a lookup dict
├── disease_checker.py   # Rule-based disease lookup (non-ML)
├── features.py          # Converts mutations into numeric feature vectors
├── neural_net.py        # Neural network built from scratch with NumPy
├── train.py             # Trains the model on ClinVar data
├── predict.py           # Runs ML prediction on a new FASTA file
├── model_weights.npz    # Saved model weights (generated after training)
├── norm_params.npz      # Normalization parameters (generated after training)
├── variant_summary.txt  # ClinVar database (download separately)
└── sequence.fasta       # Your input DNA sequence
```

---

## Installation

No external ML libraries needed. Only NumPy is required beyond the Python standard library.

```bash
pip install numpy
```

---

## Usage

### Step 1 — Train the model

This reads ClinVar, builds a labeled dataset, and trains the neural network. Run once — takes about 1 minute.

```bash
python train.py
```

Output:
```
 Building dataset from ClinVar...
 Dataset: 84,312 samples | 61,489 pathogenic | 22,823 benign

  Training...

  Epoch   0 | Loss: 0.6891 | Test Accuracy: 61.2%
  Epoch  20 | Loss: 0.5203 | Test Accuracy: 74.8%
  Epoch  40 | Loss: 0.4471 | Test Accuracy: 79.3%
  ...
  Epoch 180 | Loss: 0.3102 | Test Accuracy: 86.1%

 Model saved to model_weights.npz
 Training complete!
```

### Step 2 — Predict on a DNA sequence

```bash
python predict.py sequence.fasta
```

Output:
```
 Gene: INS | Protein length: 110 amino acids

 Running ML prediction on known variants for INS...

  Risk Score   Mutation                                      Disease
  ----------   -------------------------------------------   --------------------
   HIGH 0.97   NM_000207.3(INS):c.71C>A (p.Ala24Asp)      Permanent neonatal diabetes
   HIGH 0.94   NM_000207.3(INS):c.147C>G (p.Phe49Leu)     Hyperproinsulinemia
   MED  0.61   NM_000207.3(INS):c.266G>A (p.Arg89His)     Neonatal diabetes / MODY
   LOW  0.23   NM_000207.3(INS):c.188G>A (p.Arg63His)     MODY type 10
```

### Alternative — Rule-based lookup (no ML)

```bash
python main.py sequence.fasta
```

This skips the ML model and directly queries ClinVar for known pathogenic variants linked to the gene.

---

## How It Works

### 1. DNA Translation (`translator.py`)

The script scans all three reading frames of the DNA sequence looking for Open Reading Frames (ORFs) — sequences that begin with a start codon (`ATG`) and end with a stop codon (`TAA`, `TAG`, or `TGA`). Each codon in between is looked up in the standard genetic code table and converted to its amino acid.

### 2. Feature Extraction (`features.py`)

Each ClinVar mutation string (e.g. `p.Gly32Ser`) is parsed to extract:

- The original amino acid and its physicochemical properties (hydrophobicity, charge, polarity, size)
- The new amino acid and its properties
- The position of the mutation in the protein (normalized)
- The magnitude of the property change between original and new amino acid

This produces a 13-dimensional feature vector for each mutation.

```
[position, orig_hydrophobicity, orig_charge, orig_polarity, orig_size,
           new_hydrophobicity,  new_charge,  new_polarity,  new_size,
           Δhydrophobicity,     Δcharge,     Δpolarity,     Δsize]
```

### 3. Neural Network (`neural_net.py`)

A fully connected neural network with two hidden layers:

```
Input (13) → ReLU → Hidden (16) → ReLU → Hidden (8) → Sigmoid → Output (1)
```

- **Activation:** ReLU in hidden layers, Sigmoid at output (produces a 0–1 probability)
- **Loss:** Binary cross-entropy
- **Optimization:** Gradient descent with backpropagation, implemented from scratch
- **Regularization:** Dropout (20%) and reduced learning rate to prevent overconfidence
- **Weight init:** Xavier initialization for stable gradients

### 4. Training (`train.py`)

ClinVar variants labeled **Pathogenic** (label = 1) and **Benign** (label = 0) are extracted, deduplicated, and converted to feature vectors. The dataset is normalized (zero mean, unit variance) and split 80/20 into training and test sets. The model trains for 200 epochs.

---

## Example Genes to Try

| Gene | Disease Link | NCBI Search |
|------|-------------|-------------|
| `INS` | Neonatal diabetes, Hyperproinsulinemia | `insulin human mRNA` |
| `BRCA1` | Breast and ovarian cancer | `BRCA1 human mRNA` |
| `TP53` | Li-Fraumeni syndrome, many cancers | `TP53 human mRNA` |
| `CFTR` | Cystic fibrosis | `CFTR human mRNA` |
| `HBB` | Sickle cell disease, Beta-thalassemia | `HBB human mRNA` |

---

## Limitations

- The model predicts pathogenicity based on physicochemical properties alone. Real clinical variant interpretation also considers evolutionary conservation, population frequency, structural context, and functional assays.
- ClinVar labels reflect current scientific consensus and may change as new evidence emerges.
- The pipeline identifies mutations that are *known* in ClinVar. Novel mutations not in the database cannot be scored.
- This tool is for educational purposes only and should not be used for clinical decision-making.

---

## Possible Extensions

- Add evolutionary conservation scores as additional features (via phyloP or GERP scores)
- Visualize training loss and accuracy curves with matplotlib
- Accept any FASTA file and auto-detect the gene via NCBI BLAST
- Build a Flask web interface for drag-and-drop FASTA upload
- Train on novel mutations using sequence embeddings

---

## References

- [NCBI ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) — variant database
- [NCBI GenBank](https://www.ncbi.nlm.nih.gov/genbank/) — DNA sequences
- [The Genetic Code](https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi) — standard codon table
- Kyte & Doolittle (1982) — amino acid hydrophobicity scale used in feature extraction
