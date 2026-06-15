# features.py
import re
import numpy as np

# Physicochemical properties for each amino acid
# [hydrophobicity, charge, polarity, size]
AA_PROPERTIES = {
    'Ala': [1.8,  0,  0, 0.3], 'Arg': [-4.5, 1,  1, 0.9],
    'Asn': [-3.5, 0,  1, 0.5], 'Asp': [-3.5, -1, 1, 0.5],
    'Cys': [2.5,  0,  0, 0.4], 'Gln': [-3.5, 0,  1, 0.6],
    'Glu': [-3.5, -1, 1, 0.6], 'Gly': [-0.4, 0,  0, 0.1],
    'His': [-3.2, 0.5,1, 0.7], 'Ile': [4.5,  0,  0, 0.7],
    'Leu': [3.8,  0,  0, 0.7], 'Lys': [-3.9, 1,  1, 0.7],
    'Met': [1.9,  0,  0, 0.7], 'Phe': [2.8,  0,  0, 0.8],
    'Pro': [-1.6, 0,  0, 0.5], 'Ser': [-0.8, 0,  1, 0.3],
    'Thr': [-0.7, 0,  1, 0.5], 'Trp': [-0.9, 0,  1, 1.0],
    'Tyr': [-1.3, 0,  1, 0.9], 'Val': [4.2,  0,  0, 0.5],
    'Ter': [0,    0,  0, 0.0], # stop codon
}

def parse_mutation(mutation_str: str) -> dict | None:
    """Extract position, original AA, and new AA from a mutation string."""
    # Match patterns like (p.Gly32Ser) or (p.Arg89His)
    match = re.search(r'p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|Ter)', mutation_str)
    if not match:
        return None
    return {
        "original": match.group(1),  # e.g. Gly
        "position": int(match.group(2)),  # e.g. 32
        "new": match.group(3),  # e.g. Ser
    }

def mutation_to_vector(mutation_str: str, protein_length: int = 110) -> np.ndarray | None:
    """
    Convert a mutation string into a feature vector of numbers.
    Output: 1D numpy array of length 10
    """
    parsed = parse_mutation(mutation_str)
    if not parsed:
        return None

    orig = AA_PROPERTIES.get(parsed["original"])
    new  = AA_PROPERTIES.get(parsed["new"])
    if not orig or not new:
        return None

    orig = np.array(orig)
    new  = np.array(new)

    position_normalized = parsed["position"] / protein_length
    property_delta = new - orig          # how much the AA changed
    property_magnitude = np.abs(property_delta)  # size of change

    # Final vector: [position, orig_props x4, new_props x4, delta_magnitude x4] = 13 features
    vector = np.concatenate([
        [position_normalized],
        orig,
        new,
        property_magnitude
    ])
    return vector  # shape: (13,)