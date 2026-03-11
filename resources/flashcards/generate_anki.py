"""Generate Anki .apkg decks from flashcard CSVs."""
import csv
import os
import sys

try:
    import genanki
except ImportError:
    print("Install genanki: pip install genanki")
    sys.exit(1)

# Stable deck IDs (random-looking numbers that Anki uses to track decks across imports)
DECK_IDS = {
    "foundations": 2059400110,
    "alignment": 2059400111,
    "systems": 2059400112,
    "complete": 2059400113,
}

# Card template: plain question/answer with a horizontal rule separator
MODEL = genanki.Model(
    1607392319,
    "no-magic Algorithm Card",
    fields=[
        {"name": "Question"},
        {"name": "Answer"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": "{{Question}}",
            "afmt": '{{FrontSide}}<hr id="answer">{{Answer}}',
        },
    ],
)


def load_csv(filepath: str) -> list[dict[str, str | list[str]]]:
    """Load flashcards from a tab-separated CSV file."""
    cards: list[dict[str, str | list[str]]] = []
    with open(filepath, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header row
        for row in reader:
            if len(row) >= 2:
                cards.append({
                    "question": row[0],
                    "answer": row[1],
                    "tags": row[2].split() if len(row) > 2 else [],
                })
    return cards


def create_deck(
    name: str, deck_id: int, cards: list[dict[str, str | list[str]]]
) -> genanki.Deck:
    """Create an Anki deck from card data."""
    deck = genanki.Deck(deck_id, name)
    for card in cards:
        note = genanki.Note(
            model=MODEL,
            fields=[card["question"], card["answer"]],
            tags=card["tags"],
        )
        deck.add_note(note)
    return deck


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tiers = ["foundations", "alignment", "systems"]
    all_cards: list[dict[str, str | list[str]]] = []

    for tier in tiers:
        csv_path = os.path.join(script_dir, f"{tier}.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping")
            continue

        cards = load_csv(csv_path)
        all_cards.extend(cards)

        deck = create_deck(f"no-magic::{tier.title()}", DECK_IDS[tier], cards)
        output_path = os.path.join(script_dir, f"no-magic-{tier}.apkg")
        genanki.Package(deck).write_to_file(output_path)
        print(f"Generated {output_path} ({len(cards)} cards)")

    # Combined deck with all cards
    complete_deck = create_deck("no-magic::Complete", DECK_IDS["complete"], all_cards)
    complete_path = os.path.join(script_dir, "no-magic-complete.apkg")
    genanki.Package(complete_deck).write_to_file(complete_path)
    print(f"Generated {complete_path} ({len(all_cards)} cards)")


if __name__ == "__main__":
    main()
