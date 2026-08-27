import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse(p):
    text = p.read_text().lower()
    return re.findall(r'"([a-z]{5})"', text)


sol = parse(ROOT / "data" / "wordlists" / "solutions_classic_2315.txt")
non = parse(ROOT / "data" / "wordlists" / "nonsolutions_classic_10657.txt")
print("solutions:", len(sol), "unique:", len(set(sol)))
print("nonsolutions:", len(non), "unique:", len(set(non)))
print("total allowed:", len(sol) + len(non))
print("overlap:", len(set(sol) & set(non)))
