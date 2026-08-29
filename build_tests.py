"""
Build evaluation/tests.jsonl with ground truth derived from the corpus itself.

Why derived and not hand written: a keyword that does not exist anywhere in your
documents scores zero forever, and you would spend an evening tuning retrieval to
chase a typo. Every keyword below is verified against the corpus before it is written.

Run:  python build_tests.py
"""

import glob
import json
import re
from pathlib import Path

PRODUCTS = []
for path in sorted(glob.glob("knowledge-base/products/*.md")):
    text = open(path, encoding="utf-8").read()

    def field(name, t=text):
        match = re.search(rf"- {name}:\s*(.+)", t)
        return match.group(1).strip() if match else ""

    price = re.search(r"Price: Rs\s*([\d,]+)", text)
    PRODUCTS.append({
        "name": text.splitlines()[0].lstrip("# ").strip(),
        "category": field("Category").lower(),
        "gender": field("Gender").lower(),
        "price": int(price.group(1).replace(",", "")) if price else 0,
        "text": text.lower(),
    })

CORPUS = "\n".join(open(f, encoding="utf-8").read() for f in glob.glob("knowledge-base/*/*.md"))


def match(**rules):
    """Return product names matching the rules. This is the ground truth."""
    hits = []
    for p in PRODUCTS:
        if "mentions" in rules and rules["mentions"] not in p["text"]:
            continue
        if "category" in rules and p["category"] != rules["category"]:
            continue
        if "genders" in rules and p["gender"] not in rules["genders"]:
            continue
        if "max_price" in rules and p["price"] > rules["max_price"]:
            continue
        hits.append(p)
    return hits


def products_answer(prefix, hits):
    listed = ", ".join(f"{h['name']} (Rs {h['price']})" for h in hits)
    return f"{prefix} There are {len(hits)}: {listed}."


TESTS = []


def add(question, category, keywords, answer):
    TESTS.append({"question": question, "keywords": keywords,
                  "reference_answer": answer, "category": category})


def add_products(question, group, prefix, **rules):
    """group is the test category. rules are the product filters, which may also
    contain a key called category. Different things, so different names."""
    hits = match(**rules)
    add(question, group, [h["name"] for h in hits], products_answer(prefix, hits))


# ---------------- direct_fact: one document holds the answer ----------------
add("How long do I have to return an item and what condition must it be in?",
    "direct_fact", ["15 calendar days", "unworn", "tags attached"],
    "You may return most items within 15 calendar days of receiving your order. "
    "The item must be unworn, unwashed, in original condition with all tags attached "
    "and the original packaging intact. Sale items and gift cards are non-returnable.")

add("How much does express shipping cost and how fast is it?",
    "direct_fact", ["Express Shipping", "150", "2 to 3 business days"],
    "Express shipping costs Rs 150 per order and delivers in 2 to 3 business days, "
    "to metro cities including Mumbai, Delhi, Bangalore, Chennai, Hyderabad, Kolkata and Pune.")

add("Who founded StyleSense and when did the online platform go live?",
    "direct_fact", ["Riya Kapoor", "Arjun Mehta", "June 2018"],
    "StyleSense was founded by fashion enthusiast Riya Kapoor and tech entrepreneur "
    "Arjun Mehta. It launched in 2018 and the online platform went live in June 2018.")

add("What is the minimum order value and what does standard shipping cost?",
    "direct_fact", ["Minimum order value", "299", "50 per order"],
    "The minimum order value for delivery is Rs 299. Standard shipping costs Rs 50 per "
    "order and takes 5 to 7 business days.")

add("How many days do I have to return something, and who pays return shipping?",
    "direct_fact", ["15 calendar days", "30-day return window"],
    "The knowledge base contradicts itself. The returns policy says 15 calendar days "
    "with return shipping paid by the customer, while the About page claims a 30-day "
    "return window with free return shipping. A good answer surfaces the conflict and "
    "prefers the returns policy.")

# ---------------- occasion: the answer spans many product pages ----------------
add_products("What should I wear to a beach wedding?", "occasion",
             "Products suitable for a wedding.", mentions="wedding")
add_products("What should I wear for the Mumbai monsoon?", "occasion",
             "Products suitable for the monsoon.", mentions="monsoon")
add_products("What should I wear to a festival?", "occasion",
             "Products suitable for festivals.", mentions="festival")
add_products("I need something for the gym.", "occasion",
             "Products suitable for the gym.", mentions="gym")

# ---------------- constraint: a number or an attribute filters the answer -------
add_products("Show me something under 1500 rupees.", "constraint",
             "Products priced at Rs 1500 or under.", max_price=1500)
add_products("Do you have anything under 1000 rupees?", "constraint",
             "Products priced at Rs 1000 or under.", max_price=1000)
add_products("Which kurtas do you have for men?", "constraint",
             "Kurtas for men, including unisex.", category="kurtas", genders=("men", "unisex"))
add_products("Women's dresses under 3000 rupees?", "constraint",
             "Dresses for women priced at Rs 3000 or under.",
             category="dresses", genders=("women", "unisex"), max_price=3000)

# ---------------- enumeration: completeness is the whole task ------------------
add_products("List all the footwear you sell.", "enumeration",
             "All footwear in the catalogue.", category="footwear")
add_products("What jackets do you have?", "enumeration",
             "All jackets in the catalogue.", category="jackets")
add_products("Which of your products are unisex?", "enumeration",
             "All unisex products.", genders=("unisex",))

# ---------------- validate, then write ----------------------------------------
missing = []
for test in TESTS:
    for keyword in test["keywords"]:
        if keyword.lower() not in CORPUS.lower():
            missing.append((test["question"], keyword))

if missing:
    print("KEYWORDS NOT FOUND IN THE CORPUS, fix these before trusting any score:")
    for question, keyword in missing:
        print(f"  {keyword!r}  in  {question}")
    raise SystemExit(1)

out = Path("evaluation/tests.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for test in TESTS:
        f.write(json.dumps(test) + "\n")

print(f"Wrote {len(TESTS)} tests to {out}, every keyword verified against the corpus.\n")
for test in TESTS:
    print(f"  {test['category']:<12} {len(test['keywords']):>2} keywords   {test['question']}")
