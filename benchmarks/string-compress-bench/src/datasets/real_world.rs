// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Real-world string corpora vendored under `data/`.
//!
//! Each loader reads the file lazily at corpus-construction time, skips the
//! single `# `-prefixed source/license header, drops blank lines, and trims
//! to the caller's row budget. If the file is missing the corpus comes back
//! empty (which the `all_datasets` aggregator filters out) so the bench
//! continues to work for users who only want the synthetic data.
//!
//! Each entry's source + license attribution lives in `data/README.md`.

use std::path::PathBuf;

use crate::datasets::Corpus;

/// Resolves to the crate's `data/` directory at compile time, so the loader
/// finds the files no matter where `cargo` is invoked from.
fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("data")
}

/// Reads a vendored corpus, skips the license header, drops empties, takes
/// `scale` rows. Returns an empty vec if the file is missing — letting the
/// aggregator quietly skip absent datasets.
fn load_lines(file_name: &str, scale: usize) -> Vec<Vec<u8>> {
    let path = data_dir().join(file_name);
    let Ok(content) = std::fs::read(&path) else {
        return Vec::new();
    };
    content
        .split(|&b| b == b'\n')
        .filter(|line| !line.is_empty() && !line.starts_with(b"# "))
        .take(scale)
        .map(<[u8]>::to_vec)
        .collect()
}

/// Project Gutenberg's *Pride and Prejudice* (PG #1342). Natural English
/// prose, one physical line per row including chapter headings, dialogue,
/// and narration. Hits the FSST sweet spot: many short recurring 2-8 byte
/// fragments ("the ", " and ", "ing ", "Mr. ", etc.) with a long tail of
/// less-frequent words.
pub fn pride_and_prejudice(scale: usize) -> Corpus {
    Corpus {
        name: "pride_and_prejudice",
        strings: load_lines("pride_and_prejudice.txt", scale),
        needles: vec![
            b"Elizabeth".to_vec(),
            b" the ".to_vec(),
            b"Mr. Darcy".to_vec(),
        ],
    }
}

/// `dwyl/english-words` — one English word per line. Short, alphabetic,
/// high-cardinality (370 k+ entries upstream; we take the first 20 k).
/// Stresses FSST-12's larger symbol table: FSST-8 cannot fit a useful
/// fraction of word stems in its 255-symbol table; FSST-12's 4096 slots
/// have room for many common stems.
pub fn english_words(scale: usize) -> Corpus {
    Corpus {
        name: "english_words",
        strings: load_lines("words_alpha.txt", scale),
        needles: vec![b"ing".to_vec(), b"tion".to_vec(), b"a".to_vec()],
    }
}

/// `cisagov/dotgov-data` — registered US federal `.gov` hostnames. Short
/// (mean ~22 B), heavy `.gov` suffix, frequent agency-name fragments.
/// Roughly URL-shaped; rewards prefix/substring backends.
pub fn gov_hostnames(scale: usize) -> Corpus {
    Corpus {
        name: "gov_hostnames",
        strings: load_lines("gov_hostnames.txt", scale),
        needles: vec![b".gov".to_vec(), b"data".to_vec(), b"nasa".to_vec()],
    }
}

/// `datasets/airport-codes` — pipe-delimited airport records. Each row is a
/// long record with a recurring template:
/// `type|name|iso_country|iso_region|municipality|iata_code|coordinates`.
/// The repeated `|` separators, repeated country codes, and similar
/// `iso_region` prefixes make this an OnPair-friendly long-template shape.
pub fn airport_records(scale: usize) -> Corpus {
    Corpus {
        name: "airport_records",
        strings: load_lines("airport_records.txt", scale),
        needles: vec![b"|US|".to_vec(), b"small_airport".to_vec(), b"heliport".to_vec()],
    }
}

/// `datasets/world-cities` — `name, subcountry, country` triples covering
/// ~15 k cities in many scripts. Comma-separated with a long tail of
/// recurring country names ("United States", "China", "India"). Mixed
/// UTF-8 scripts stress encoders that assume mostly-ASCII input.
pub fn world_cities(scale: usize) -> Corpus {
    Corpus {
        name: "world_cities",
        strings: load_lines("world_cities.txt", scale),
        needles: vec![
            b", United States".to_vec(),
            b", China".to_vec(),
            b", India".to_vec(),
        ],
    }
}
