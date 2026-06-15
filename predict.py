# predict.py
import numpy as np
import os
import re
import sys
from translator import read_fasta, find_proteins
from features import mutation_to_vector
from neural_net import NeuralNet
from load_clinvar import load_clinvar

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def predict(fasta_path: str):
    # Load model
    model = NeuralNet()
    model.load(os.path.join(BASE_DIR, "model_weights.npz"))
    norm = np.load(os.path.join(BASE_DIR, "norm_params.npz"))
    mean, std = norm["mean"], norm["std"]

    # Translate DNA
    header, dna = read_fasta(fasta_path)
    gene_match = re.search(r'\((\w+)\)', header)
    gene = gene_match.group(1) if gene_match else "UNKNOWN"
    proteins = find_proteins(dna)

    if not proteins:
        print(" No proteins found.")
        return

    print(f"\n Gene: {gene} | Protein length: {len(proteins[0])} amino acids")

    # Load known variants for this gene and score each one
    print(f"\n Running ML prediction on known variants for {gene}...\n")
    db = load_clinvar()

    if gene not in db:
        print(f"No variants found for {gene}")
        return

    results = []
    for variant in db[gene]:
        vec = mutation_to_vector(variant["mutation"], len(proteins[0]))
        if vec is None:
            continue
        vec_norm = (vec - mean) / (std + 1e-8)
        score = model.forward(vec_norm.reshape(1, -1))[0][0]
        results.append((score, variant))

    results.sort(reverse=True)

    print(f"  {'Risk Score':<12} {'Mutation':<45} {'Disease'}")
    print(f"  {'-'*10:<12} {'-'*43:<45} {'-'*20}")
    for score, v in results[:10]:
        risk = " HIGH" if score > 0.7 else " MED" if score > 0.4 else " LOW"
        mutation_short = v['mutation'][:42] + "..." if len(v['mutation']) > 42 else v['mutation']
        disease_short  = v['disease'].split('|')[0][:35]
        print(f"  {risk} {score:.2f}   {mutation_short:<45} {disease_short}")

if __name__ == "__main__":
    fasta = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE_DIR, "sequence.fasta")
    predict(fasta)