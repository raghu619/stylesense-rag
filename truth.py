"""Prints what the correct answer actually was. The reveal, for the demo."""

import glob
import re

LIMIT = 1500
matches = []

for path in glob.glob("knowledge-base/products/*.md"):
    text = open(path, encoding="utf-8").read()
    price = int(re.search(r"Price: Rs\s*([\d,]+)", text).group(1).replace(",", ""))
    if price <= LIMIT:
        matches.append((price, text.splitlines()[0].lstrip("# ").strip()))

print(f"\nProducts under Rs {LIMIT}: {len(matches)}\n")
for price, name in sorted(matches):
    print(f"   Rs {price:<6} {name}")
print()
