import sys
sys.path.insert(0, 'src')
sys.path.insert(0, '.')
from memory_trp.loading import load_contacts
import numpy as np

datasets = [
    ("Primary", "primaryschool", 242, 11),
    ("High School 1", "hs2011", 126, 4),
    ("High School 2", "hs2012", 180, 5),
    ("High School 3", "hs2013", 327, 9),
    ("Conference", "conference", 405, None),
    ("Office", "workplace", 232, 12),
    ("Hospital", "hospital", 75, 4),
    ("Malawi", "malawi", 86, None),
]

print(f"{'Dataset':<15} | {'n (code)':<8} {'n (Table 1)':<12} | {'k (code)':<8} {'k (Table 1)':<12} | Status")
print("-" * 75)
for label, name, exp_n, exp_k in datasets:
    c, l = load_contacts(name)
    act_n = int(max(c['i'].max(), c['j'].max())) + 1
    act_k = int(np.unique(l).size) if l is not None else None
    
    n_match = (act_n == exp_n)
    k_match = (act_k == exp_k)
    status = "EXACT MATCH" if (n_match and k_match) else "NOTE"
    print(f"{label:<15} | {act_n:<8} {exp_n:<12} | {str(act_k):<8} {str(exp_k):<12} | {status}")
