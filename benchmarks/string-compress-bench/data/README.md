# String-Compression Benchmark Corpora

Real-world string corpora used to stress-test the string compression backends
(FSST-8, FSST-12, OnPair, OnPair16, OnPair-cpp) beyond the synthetic datasets
in the crate.

Every file is plain UTF-8, one record per line, with no blank lines and no
embedded control characters. The first line of every file is a single
`# `-prefixed comment giving the source URL and license; loaders should skip
any leading lines that start with `# `.

Lines longer than 512 bytes are filtered out during preparation, and each
file is capped at roughly 500 KB so the repo stays small. The loader is
expected to take the first N records as requested by the benchmark.

## Files

| File | Records | File size | Avg line length | Source | License | Stresses |
|------|--------:|----------:|----------------:|--------|---------|----------|
| `pride_and_prejudice.txt` |  7,776 | 488 KiB | 63.3 B | Project Gutenberg eBook #1342, Jane Austen, via GITenberg mirror | Project Gutenberg License (public domain in US) | Natural English text with rich word/n-gram statistics. Heavy reuse of common bigrams/trigrams; should be the sweet spot for FSST-8. |
| `words_alpha.txt`         | 20,000 | 205 KiB |  9.5 B | `dwyl/english-words` (`words_alpha.txt`) | Unlicense (public domain) | High-cardinality short identifiers (mean ~10 B, no spaces). Exercises FSST-12's larger symbol table and rejects dictionaries that can only exploit very long shared prefixes. |
| `gov_hostnames.txt`       |  6,695 | 148 KiB | 21.6 B | `cisagov/dotgov-data` `current-federal.csv`, plus synthesized `www.` / `https://.../about` / `https://.../contact` variants to hit the row count target | CC0 1.0 (also US Government work) | Real US `.gov` hostnames and URLs. Mix of short domains and longer URL templates with shared `https://www.` / `.gov` prefixes/suffixes — FSST-friendly with some OnPair shared-template content. |
| `airport_records.txt`     |  5,859 | 488 KiB | 84.3 B | `datasets/airport-codes` (`airport-codes.csv`) — columns `type|name|iso_country|iso_region|municipality|iata_code|coordinates` joined with `\|` | Open Data Commons Public Domain Dedication & License (ODC-PDDL 1.0) | Structured records with strong shared templates: `small_airport\|...\|US\|US-XX\|...`. Designed to be OnPair-friendly (long shared prefixes and recurring substrings between consecutive rows). |
| `world_cities.txt`        | 15,707 | 488 KiB | 30.8 B | `datasets/world-cities` (`world-cities.csv`) — `name, subcountry, country` | CC-BY 3.0 (derived from GeoNames, CC-BY) | UTF-8 mixed-script city names (Latin, Cyrillic, Arabic, CJK, accented characters) with repeating country/subcountry suffixes. Tests UTF-8 handling and a mix of FSST symbol reuse and OnPair-like row-to-row redundancy. |

## What each corpus is meant to expose

- **Pride and Prejudice** — natural prose. Wide vocabulary, frequent function
  words, English n-gram statistics. The classic FSST-friendly case.
- **words_alpha** — dictionary of distinct short English words. Each line is
  effectively a unique short identifier. There are no inter-line shared
  templates, so OnPair pair-encoding has little to exploit; FSST-12's wider
  symbol table should pull ahead of FSST-8 here.
- **gov_hostnames** — domain names and synthesized URL paths. Tests how the
  encoders deal with very small alphabets (`[a-z0-9.-/]`) and shared
  suffixes (`.gov`) and prefixes (`https://www.`). Sits between FSST and
  OnPair in profile.
- **airport_records** — structured `|`-delimited records with a fixed schema
  and many repeated tokens (`US-PA`, `small_airport`, etc.). Should be the
  best case for OnPair / OnPair16, because consecutive rows share long
  literal sub-strings.
- **world_cities** — UTF-8 records with international scripts and a
  repeating tail (`..., <subcountry>, <country>`). Exercises both the
  byte-oriented FSST symbol selection on multi-byte UTF-8 sequences and the
  OnPair pair-encoding on the redundant suffix.

## Reproducing the data

The files in this directory were produced from public mirrors using only
`curl -sL --max-time 30` plus a short Python cleaner that:

1. fetches the upstream file,
2. strips Project Gutenberg boilerplate when applicable,
3. drops blank lines and lines with control characters,
4. drops lines longer than 512 bytes,
5. caps the file at ~500 KB while keeping at least 2,000 records, and
6. prepends a single `# `-prefixed license/source attribution.

Upstream sources:

- Pride and Prejudice — <https://github.com/GITenberg/Pride-and-Prejudice_1342>
  (Gutenberg eBook #1342, raw plain text)
- words_alpha — <https://github.com/dwyl/english-words>
- .gov hostnames — <https://github.com/cisagov/dotgov-data>
  (`current-federal.csv`)
- airport-codes — <https://github.com/datasets/airport-codes>
- world-cities — <https://github.com/datasets/world-cities>
