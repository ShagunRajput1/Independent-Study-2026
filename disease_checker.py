# disease_checker.py

def check_for_disease(protein: list[str], gene_name: str, clinvar_db: dict) -> None:
    """
    Given a protein (list of amino acids) and a gene name,
    check ClinVar for known disease-linked mutations.
    """
    protein_str = "-".join(protein)  # e.g. "Met-Ala-Leu-Trp-..."

    print(f"\n Checking gene '{gene_name}' against ClinVar...\n")

    if gene_name not in clinvar_db:
        print(f"'{gene_name}' not found in ClinVar database.")
        print("  Try checking the gene symbol (e.g. 'BRCA1', 'INS', 'TP53')")
        return

    variants = clinvar_db[gene_name]
    print(f"  Found {len(variants)} known pathogenic variant(s) for {gene_name}:\n")

    for v in variants[:10]:  # show first 10 to avoid flooding output
        print(f"  Mutation:     {v['mutation']}")
        print(f"  Disease:      {v['disease']}")
        print(f"  Significance: {v['significance']}")
        print()

    if len(variants) > 10:
        print(f"  ... and {len(variants) - 10} more variants.")