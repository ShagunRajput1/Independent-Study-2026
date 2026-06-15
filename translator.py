# translator.py

CODON_TABLE = {
    'TTT': 'Phe', 'TTC': 'Phe', 'TTA': 'Leu', 'TTG': 'Leu',
    'CTT': 'Leu', 'CTC': 'Leu', 'CTA': 'Leu', 'CTG': 'Leu',
    'ATT': 'Ile', 'ATC': 'Ile', 'ATA': 'Ile', 'ATG': 'Met (START)',
    'GTT': 'Val', 'GTC': 'Val', 'GTA': 'Val', 'GTG': 'Val',
    'TCT': 'Ser', 'TCC': 'Ser', 'TCA': 'Ser', 'TCG': 'Ser',
    'CCT': 'Pro', 'CCC': 'Pro', 'CCA': 'Pro', 'CCG': 'Pro',
    'ACT': 'Thr', 'ACC': 'Thr', 'ACA': 'Thr', 'ACG': 'Thr',
    'GCT': 'Ala', 'GCC': 'Ala', 'GCA': 'Ala', 'GCG': 'Ala',
    'TAT': 'Tyr', 'TAC': 'Tyr', 'TAA': 'STOP', 'TAG': 'STOP',
    'CAT': 'His', 'CAC': 'His', 'CAA': 'Gln', 'CAG': 'Gln',
    'AAT': 'Asn', 'AAC': 'Asn', 'AAA': 'Lys', 'AAG': 'Lys',
    'GAT': 'Asp', 'GAC': 'Asp', 'GAA': 'Glu', 'GAG': 'Glu',
    'TGT': 'Cys', 'TGC': 'Cys', 'TGA': 'STOP', 'TGG': 'Trp',
    'CGT': 'Arg', 'CGC': 'Arg', 'CGA': 'Arg', 'CGG': 'Arg',
    'AGT': 'Ser', 'AGC': 'Ser', 'AGA': 'Arg', 'AGG': 'Arg',
    'GGT': 'Gly', 'GGC': 'Gly', 'GGA': 'Gly', 'GGG': 'Gly',
}

def read_fasta(filepath: str) -> tuple[str, str]:
    """Parse a FASTA file and return (header, sequence)."""
    header, sequence = "", []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                header = line[1:]
            else:
                sequence.append(line.upper())
    return header, "".join(sequence)

def find_proteins(dna: str) -> list[list[str]]:
    """
    Scan all 3 reading frames for ORFs (Open Reading Frames).
    Returns a list of proteins, each as a list of amino acid names.
    """
    proteins = []

    for frame in range(3):
        i = frame
        in_protein = False
        current_protein = []

        while i + 3 <= len(dna):
            codon = dna[i:i+3]
            amino_acid = CODON_TABLE.get(codon, "Unknown")

            if amino_acid == "Met (START)" and not in_protein:
                in_protein = True
                current_protein = ["Met"]

            elif in_protein:
                if amino_acid == "STOP":
                    if len(current_protein) > 5:
                        proteins.append(current_protein)
                    in_protein = False
                    current_protein = []
                else:
                    current_protein.append(amino_acid.split()[0])

            i += 3

    return proteins