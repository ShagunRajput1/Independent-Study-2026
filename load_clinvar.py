# load_clinvar.py
import csv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_clinvar(filepath=None) -> dict:
    if filepath is None:
        filepath = os.path.join(BASE_DIR, "variant_summary.txt")
    db = {}
    seen = set()

    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            gene = row.get("GeneSymbol", "").strip()
            disease = row.get("PhenotypeList", "").strip()
            significance = row.get("ClinicalSignificance", "").strip()
            mutation = row.get("Name", "").strip()

            if gene and disease and "Pathogenic" in significance:
                key = (gene, mutation, disease)
                if key in seen:
                    continue
                seen.add(key)

                if gene not in db:
                    db[gene] = []
                db[gene].append({
                    "mutation": mutation,
                    "disease": disease,
                    "significance": significance
                })

    print(f" Loaded {sum(len(v) for v in db.values())} pathogenic variants across {len(db)} genes.")
    return db