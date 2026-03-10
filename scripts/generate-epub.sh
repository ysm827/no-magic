#!/usr/bin/env bash
# Generate an EPUB book from all no-magic algorithm scripts.
# Requires: pandoc (https://pandoc.org)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$REPO_DIR/build"
CSS_FILE="$SCRIPT_DIR/epub.css"
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Check for pandoc
if ! command -v pandoc &>/dev/null; then
    echo "Error: pandoc is required. Install: brew install pandoc (macOS) or apt install pandoc (Ubuntu)"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

BOOK="$TEMP_DIR/book.md"

# --- Extract thesis docstring (first """...""" block) from a Python file ---
extract_thesis() {
    local file="$1"
    python3 -c "
import re, sys
text = open(sys.argv[1]).read()
m = re.search(r'\"\"\"(.*?)\"\"\"', text, re.DOTALL)
if m:
    print(m.group(1).strip())
" "$file"
}

# --- Extract TRADEOFFS section from a Python file ---
extract_tradeoffs() {
    local file="$1"
    python3 -c "
import sys
lines = open(sys.argv[1]).readlines()
in_section = False
result = []
for line in lines:
    stripped = line.strip()
    if stripped == '# === TRADEOFFS ===':
        in_section = True
        continue
    if in_section:
        if stripped.startswith('# ===') and 'TRADEOFFS' not in stripped:
            break
        if stripped == '' or (not stripped.startswith('#') and stripped != ''):
            break
        # Strip leading '# ' from comment lines
        if stripped.startswith('# '):
            result.append(stripped[2:])
        elif stripped == '#':
            result.append('')
if result:
    print('\n'.join(result))
" "$file"
}

# --- Extract README sections by header name ---
extract_readme_section() {
    local file="$1"
    local header="$2"
    python3 -c "
import re, sys
text = open(sys.argv[1]).read()
header = sys.argv[2]
# Match ## Header through the next ## or end of file
pattern = r'## ' + re.escape(header) + r'\s*\n(.*?)(?=\n## |\Z)'
m = re.search(pattern, text, re.DOTALL)
if m:
    print(m.group(1).strip())
" "$file" "$header"
}

# --- Build the Markdown document ---

# Title page
cat > "$BOOK" <<'TITLEEOF'
---
title: "no-magic"
subtitle: "AI/ML Algorithms from First Principles"
author: "Mathews-Tom"
---

TITLEEOF

# Introduction
echo "# Introduction" >> "$BOOK"
echo "" >> "$BOOK"

what_section=$(extract_readme_section "$REPO_DIR/README.md" "What This Is")
if [ -n "$what_section" ]; then
    echo "## What This Is" >> "$BOOK"
    echo "" >> "$BOOK"
    echo "$what_section" >> "$BOOK"
    echo "" >> "$BOOK"
fi

philosophy_section=$(extract_readme_section "$REPO_DIR/README.md" "Philosophy")
if [ -n "$philosophy_section" ]; then
    echo "## Philosophy" >> "$BOOK"
    echo "" >> "$BOOK"
    echo "$philosophy_section" >> "$BOOK"
    echo "" >> "$BOOK"
fi

# --- Helper: add a script as a chapter ---
add_chapter() {
    local file="$1"
    local basename
    basename=$(basename "$file" .py)

    echo "  Processing: $file"

    echo "## $basename" >> "$BOOK"
    echo "" >> "$BOOK"

    # Thesis
    local thesis
    thesis=$(extract_thesis "$file")
    if [ -n "$thesis" ]; then
        echo "> $thesis" >> "$BOOK"
        echo "" >> "$BOOK"
    fi

    # Tradeoffs
    local tradeoffs
    tradeoffs=$(extract_tradeoffs "$file")
    if [ -n "$tradeoffs" ]; then
        echo "### Tradeoffs" >> "$BOOK"
        echo "" >> "$BOOK"
        echo '```' >> "$BOOK"
        echo "$tradeoffs" >> "$BOOK"
        echo '```' >> "$BOOK"
        echo "" >> "$BOOK"
    fi

    # Full source
    echo "### Source" >> "$BOOK"
    echo "" >> "$BOOK"
    echo '```python' >> "$BOOK"
    cat "$file" >> "$BOOK"
    echo "" >> "$BOOK"
    echo '```' >> "$BOOK"
    echo "" >> "$BOOK"
}

# --- Collect scripts per tier ---

# Tier directories and their part titles
declare -a TIERS=("01-foundations" "02-alignment" "03-systems")
declare -a PARTS=("Part I: Foundations" "Part II: Alignment & Training Techniques" "Part III: Systems & Inference")

for i in "${!TIERS[@]}"; do
    tier="${TIERS[$i]}"
    part="${PARTS[$i]}"
    tier_dir="$REPO_DIR/$tier"

    echo "# $part" >> "$BOOK"
    echo "" >> "$BOOK"

    # Collect micro*.py files sorted alphabetically
    while IFS= read -r script; do
        add_chapter "$script"
    done < <(find "$tier_dir" -maxdepth 1 -name 'micro*.py' | sort)

    # Collect comparison scripts (non-micro *.py files)
    while IFS= read -r script; do
        add_chapter "$script"
    done < <(find "$tier_dir" -maxdepth 1 -name '*.py' ! -name 'micro*.py' | sort)
done

# --- Generate EPUB ---
echo ""
echo "Generating EPUB..."

EPUB_OUT="$OUTPUT_DIR/no-magic.epub"

pandoc "$BOOK" \
    -o "$EPUB_OUT" \
    --toc \
    --toc-depth=2 \
    --metadata title="no-magic" \
    --metadata author="Mathews-Tom" \
    --css "$CSS_FILE" \
    --split-level=1

echo "Done: $EPUB_OUT"
