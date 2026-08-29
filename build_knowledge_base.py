from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv(override=True)
client = OpenAI()
MODEL = "gpt-4.1-mini"

CATEGORIES = {
    "shirts": 7, "t-shirts": 6, "kurtas": 6, "dresses": 7,
    "jeans": 6, "trousers": 6, "jackets": 7, "footwear": 7,
}

GUIDES = [
    "How to dress for the Mumbai monsoon",
    "What to wear to an Indian wedding as a guest",
    "Building a ten piece summer capsule wardrobe",
    "Fabric guide: linen, cotton, denim and blends",
    "Fit guide: relaxed, regular and slim",
]

POLICIES = [
    "About StyleSense, our story and what we stand for",
    "Sizing chart and fit policy",
    "Returns and exchanges policy",
    "Shipping and delivery policy",
]


class Product(BaseModel):
    filename: str
    markdown: str


class ProductList(BaseModel):
    products: list[Product]


PRODUCT_PROMPT = """Write {count} different product pages for a fictional Indian online
fashion store called StyleSense. Category: {category}.

Each page is markdown and follows exactly this shape:

# <Product name>

<Two or three sentences describing how it looks and feels.>

## Details
- Brand: <fictional brand name, never a real one>
- Category: {category}
- Gender: <men, women or unisex>
- Colour: <plain words>
- Fabric: <e.g. cotton linen blend>
- Fit: <e.g. relaxed>
- Sizes: <list>
- Price: Rs <number between 499 and 9999>

## Best for
<One or two sentences naming real occasions and seasons: office, beach, monsoon,
wedding, festival, travel, gym, date night.>

## Care
<One sentence.>

Spread prices across cheap to premium. Vary gender, colour, fabric and fit.
filename is lowercase with hyphens and ends in .md, for example linen-resort-shirt.md
"""


def write(folder: str, name: str, text: str) -> None:
    Path(f"knowledge-base/{folder}/{name}").write_text(text, encoding="utf-8")


def build_products() -> int:
    total = 0
    for category, count in CATEGORIES.items():
        result = client.chat.completions.parse(
            model=MODEL,
            messages=[{"role": "user",
                       "content": PRODUCT_PROMPT.format(count=count, category=category)}],
            response_format=ProductList,
        ).choices[0].message.parsed
        for product in result.products:
            name = product.filename.strip().lower().replace(" ", "-").split("/")[-1]
            name = name if name.endswith(".md") else name + ".md"
            write("products", name, product.markdown)
            total += 1
        print(f"  {category}: {len(result.products)} pages")
    return total


def build_docs(folder: str, titles: list[str], style: str) -> None:
    for title in titles:
        text = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content":
                       f"Write a markdown document titled '{title}' for a fictional Indian "
                       f"online fashion store called StyleSense. {style} "
                       f"300 to 500 words, with a '# ' title and two or three '## ' sections. "
                       f"Be specific and concrete, invent details where needed."}],
        ).choices[0].message.content
        name = title.lower().split(",")[0].split(":")[0].replace(" ", "-") + ".md"
        write(folder, name, text)
        print(f"  {name}")


print("Products")
count = build_products()
print("Guides")
build_docs("guides", GUIDES, "Practical styling advice, no fluff.")
print("Policies")
build_docs("policies", POLICIES, "Clear operational detail: numbers, days, conditions.")
print(f"\nDone. {count} products plus 9 other documents.")
