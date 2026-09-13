"""
Price parsing, in one place.

ingest.py needs to read a price OUT of a product document to store it as chunk
metadata. The evaluation needs to read a price ceiling OUT of a question to
check violations, and step 3 needs the same function to build the Chroma where
clause. Three callers, one definition.

The rule this follows is the one already written at the top of rag.py: if the
evaluation builds its own copy, you are measuring a system you do not ship.
The moment a second file needs the same parser, it moves here rather than
being pasted.
"""

import glob
import re

# "Price: Rs 1,499" inside a product markdown file.
PRICE_IN_DOC = re.compile(r"Price:\s*Rs\s*([\d,]+)", re.IGNORECASE)

# "under 1500 rupees", "below Rs 2000", "less than 3000" inside a user question.
PRICE_LIMIT = re.compile(
    r"(?:under|below|less than|within|up ?to)\s*(?:rs\.?\s*)?([\d,]+)", re.IGNORECASE
)


def parse_price(text: str) -> int | None:
    """The price stated in a product document, as an int, or None if it states none."""
    match = PRICE_IN_DOC.search(text)
    return int(match.group(1).replace(",", "")) if match else None


def parse_price_limit(question: str) -> int | None:
    """The price ceiling a question asks for, or None if it asks for no ceiling."""
    match = PRICE_LIMIT.search(question)
    return int(match.group(1).replace(",", "")) if match else None


def load_catalogue(pattern: str = "knowledge-base/products/*.md") -> dict[str, dict]:
    """
    source path -> {title, price}, read from disk.

    Keyed by path because that is what chunk metadata carries, and read from the
    FILE rather than from chunk text: at small chunk sizes a chunk can hold a
    product name without its Price line, because the splitter cut between them.
    Metadata survives chunking, text does not.
    """
    catalogue = {}
    for path in glob.glob(pattern):
        text = open(path, encoding="utf-8").read()
        price = parse_price(text)
        if price is not None:
            catalogue[path] = {"title": text.splitlines()[0].lstrip("# ").strip(), "price": price}
    return catalogue
