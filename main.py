# main.py

import sys
from translator import read_fasta, find_proteins
from load_clinvar import load_clinvar
from disease_checker import check_for_disease

def extract_gene_name(header: str) -> str:
    """Try to pull gene symbol from FASTA header."""
    # FASTA headers often contain gene symbols in parentheses e.g. (INS) or (BRCA1)
    import re
    match = re.search(r'\((\w+)\)', header)
    return match.group(1) if match else "UNKNOWN"

def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else "sequence.fasta"

    # Step 1: Read + translate DNA
    print(f"Reading: {filepath}\n")
    header, dna = read_fasta(filepath)
    gene_name = extract_gene_name(header)

    print(f" Gene:   {gene_name}")
    print(f" Length: {len(dna)} nucleotides\n")

    proteins = find_proteins(dna)

    if not proteins:
        print("No proteins found.")
        return

    print(f"Found {len(proteins)} protein(s):\n")
    for i, protein in enumerate(proteins, 1):
        print(f"  Protein {i} ({len(protein)} amino acids): {' → '.join(protein[:8])}...")

    # Step 2: Load disease database
    print("\nLoading ClinVar disease database...")
    clinvar_db = load_clinvar("variant_summary.txt")

    # Step 3: Check for disease links
    check_for_disease(proteins[0], gene_name, clinvar_db)

if __name__ == "__main__":
    main()