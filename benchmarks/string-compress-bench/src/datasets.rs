// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Synthetic string-corpus generators.
//!
//! We can not download real-world datasets here, so each generator produces a
//! deterministic seeded corpus that exercises a different property a string
//! compressor cares about (skewed dictionaries, long shared prefixes, random
//! noise, URL-shaped strings, fragmented bag-of-words, etc.).

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;

/// A synthetic string corpus shared by every backend benchmark.
#[derive(Clone)]
pub struct Corpus {
    /// Short identifier (used in report rows and bench arg labels).
    pub name: &'static str,
    /// The strings themselves. Empty strings are allowed.
    pub strings: Vec<Vec<u8>>,
    /// A few well-known needles for pushdown / LIKE evaluation. These are
    /// chosen to hit a non-trivial fraction of `strings` so the predicate
    /// produces a measurable result.
    pub needles: Vec<Vec<u8>>,
}

// `Debug` is implemented manually so divan's bench-arg formatter shows the
// dataset name instead of dumping every row.
impl std::fmt::Debug for Corpus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name)
    }
}

impl Corpus {
    pub fn total_bytes(&self) -> usize {
        self.strings.iter().map(|s| s.len()).sum()
    }

    pub fn len(&self) -> usize {
        self.strings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

/// Build the suite of synthetic datasets used by every bench / report run.
pub fn all_datasets(scale: usize) -> Vec<Corpus> {
    vec![
        skewed_dictionary(scale),
        url_like(scale),
        random_bytes(scale),
        long_shared_prefix(scale),
        natural_words(scale),
        json_like(scale),
        short_codes(scale),
        adversarial_mix(scale),
    ]
}

/// 32-word vocabulary; each row is `1-6` words drawn from a Zipf-ish
/// distribution. Hits the FSST sweet spot of a small, high-frequency
/// dictionary.
pub fn skewed_dictionary(scale: usize) -> Corpus {
    let mut rng = StdRng::seed_from_u64(0xA1A1);
    let vocab: [&str; 32] = [
        "the", "of", "and", "to", "in", "that", "for", "is", "on", "with", "as", "this",
        "by", "be", "an", "or", "are", "from", "at", "we", "but", "not", "you", "they",
        "have", "has", "had", "will", "would", "could", "should", "may",
    ];

    let mut strings = Vec::with_capacity(scale);
    for _ in 0..scale {
        let word_count = rng.random_range(1..=6);
        let mut buf = Vec::with_capacity(32);
        for w in 0..word_count {
            if w > 0 {
                buf.push(b' ');
            }
            // Skew so early-vocab words dominate.
            let idx = ((rng.random::<f64>().powi(3)) * vocab.len() as f64) as usize;
            buf.extend_from_slice(vocab[idx.min(vocab.len() - 1)].as_bytes());
        }
        strings.push(buf);
    }

    Corpus {
        name: "skewed_dict",
        strings,
        needles: vec![b"the".to_vec(), b"and".to_vec(), b" of ".to_vec()],
    }
}

/// URL-shaped strings with a small set of schemes/hosts and random paths.
/// Exercises FSST's ability to learn fixed prefixes (`https://`) and
/// recurring infixes (`/v1/`, `?id=`).
pub fn url_like(scale: usize) -> Corpus {
    let mut rng = StdRng::seed_from_u64(0xB2B2);
    let schemes = ["https://", "http://"];
    let hosts = [
        "example.com",
        "api.example.com",
        "cdn.example.com",
        "vortex.dev",
        "github.com/spiraldb/vortex",
    ];
    let paths = ["/v1/users/", "/v1/items/", "/v2/orders/", "/static/", "/index"];
    let query_keys = ["id=", "ref=", "src=", "tag="];

    let mut strings = Vec::with_capacity(scale);
    for _ in 0..scale {
        let mut buf = Vec::with_capacity(80);
        buf.extend_from_slice(schemes.choose(&mut rng).unwrap().as_bytes());
        buf.extend_from_slice(hosts.choose(&mut rng).unwrap().as_bytes());
        buf.extend_from_slice(paths.choose(&mut rng).unwrap().as_bytes());
        for _ in 0..rng.random_range(0..8) {
            buf.push(rng.random_range(b'a'..=b'z'));
        }
        if rng.random_bool(0.6) {
            buf.push(b'?');
            buf.extend_from_slice(query_keys.choose(&mut rng).unwrap().as_bytes());
            let n: u32 = rng.random();
            buf.extend_from_slice(n.to_string().as_bytes());
        }
        strings.push(buf);
    }

    Corpus {
        name: "urls",
        strings,
        needles: vec![
            b"https://".to_vec(),
            b"example.com".to_vec(),
            b"/v1/".to_vec(),
        ],
    }
}

/// High-entropy random bytes from a 64-character alphabet. Worst case for
/// dictionary-based compressors.
pub fn random_bytes(scale: usize) -> Corpus {
    let mut rng = StdRng::seed_from_u64(0xC3C3);
    let strings: Vec<Vec<u8>> = (0..scale)
        .map(|_| {
            let len = rng.random_range(4..=64);
            (0..len)
                .map(|_| {
                    let r = rng.random_range(0..64u8);
                    if r < 26 {
                        b'a' + r
                    } else if r < 52 {
                        b'A' + (r - 26)
                    } else {
                        b'0' + (r - 52)
                    }
                })
                .collect()
        })
        .collect();
    Corpus {
        name: "random_alnum",
        strings,
        needles: vec![b"aA".to_vec(), b"a0".to_vec()],
    }
}

/// All strings share a long prefix (`product://catalog/2026/`), then drift.
/// Stress-tests long-symbol coverage.
pub fn long_shared_prefix(scale: usize) -> Corpus {
    let mut rng = StdRng::seed_from_u64(0xD4D4);
    let prefix = b"product://catalog/2026/q4/region-na/category-electronics/sku-";
    let strings: Vec<Vec<u8>> = (0..scale)
        .map(|_| {
            let mut buf = Vec::with_capacity(prefix.len() + 12);
            buf.extend_from_slice(prefix);
            for _ in 0..rng.random_range(6..=12) {
                buf.push(rng.random_range(b'0'..=b'9'));
            }
            buf
        })
        .collect();
    Corpus {
        name: "long_prefix",
        strings,
        needles: vec![
            prefix.to_vec(),
            b"region-na".to_vec(),
            b"category-electronics".to_vec(),
        ],
    }
}

/// Bag of natural-English-looking words drawn with replacement; each row is
/// `1-12` of them. Different sparsity profile than `skewed_dict`.
pub fn natural_words(scale: usize) -> Corpus {
    let mut rng = StdRng::seed_from_u64(0xE5E5);
    let vocab = [
        "data", "vortex", "compression", "string", "benchmark", "query", "table",
        "column", "scan", "encoding", "symbol", "dictionary", "fast", "static",
        "byte", "pair", "match", "longest", "prefix", "decode", "encode", "lookup",
        "system", "memory", "throughput", "ratio", "speed", "bench", "size", "level",
        "tier", "node", "shard",
    ];
    let strings: Vec<Vec<u8>> = (0..scale)
        .map(|_| {
            let n = rng.random_range(1..=12);
            let mut buf = Vec::with_capacity(8 * n);
            for i in 0..n {
                if i > 0 {
                    buf.push(b' ');
                }
                buf.extend_from_slice(vocab.choose(&mut rng).unwrap().as_bytes());
            }
            buf
        })
        .collect();
    Corpus {
        name: "natural_words",
        strings,
        needles: vec![
            b"vortex".to_vec(),
            b"compression".to_vec(),
            b"dictionary".to_vec(),
        ],
    }
}

/// Mini JSON snippets - exercises punctuation-heavy substrings and quoted
/// keys, a near-pathological case for naive prefix matchers.
pub fn json_like(scale: usize) -> Corpus {
    let mut rng = StdRng::seed_from_u64(0xF6F6);
    let keys = ["id", "name", "kind", "status", "ts", "score"];
    let statuses = ["ok", "pending", "failed", "queued"];
    let strings: Vec<Vec<u8>> = (0..scale)
        .map(|_| {
            let id: u32 = rng.random();
            let key = keys.choose(&mut rng).unwrap();
            let status = statuses.choose(&mut rng).unwrap();
            format!(
                "{{\"{key}\":\"{status}\",\"id\":{id},\"score\":{score:.2}}}",
                score = rng.random::<f64>()
            )
            .into_bytes()
        })
        .collect();
    Corpus {
        name: "json_like",
        strings,
        needles: vec![
            b"\"status\":".to_vec(),
            b"\"pending\"".to_vec(),
            b"\"score\":".to_vec(),
        ],
    }
}

/// Very short fixed-format codes like `US-12345`, `JP-00042`. These barely
/// give the dictionary trainer enough material to do anything interesting.
pub fn short_codes(scale: usize) -> Corpus {
    let mut rng = StdRng::seed_from_u64(0x1717);
    let cc = ["US", "JP", "GB", "DE", "FR", "BR", "IN", "CN", "AU", "MX"];
    let strings: Vec<Vec<u8>> = (0..scale)
        .map(|_| {
            let mut buf = Vec::with_capacity(8);
            buf.extend_from_slice(cc.choose(&mut rng).unwrap().as_bytes());
            buf.push(b'-');
            let n: u32 = rng.random_range(0..100_000);
            buf.extend_from_slice(format!("{n:05}").as_bytes());
            buf
        })
        .collect();
    Corpus {
        name: "short_codes",
        strings,
        needles: vec![b"US-".to_vec(), b"JP-".to_vec()],
    }
}

/// Stress dataset: every row is drawn from one of four sub-patterns that
/// individually defeat a *different* part of each algorithm. Even with all
/// backends doing their best, the dictionary can not converge on any one
/// pattern, so ratios collapse toward 1.0 (or worse, for backends that
/// spend bytes on a dictionary header).
///
/// Sub-patterns (each ≈25 % of rows, interleaved deterministically):
///
/// 1. **`session`** — 22-character base64-shaped session IDs. High Shannon
///    entropy, no recurrence across rows. FSST's symbol-table training
///    finds nothing better than 1-byte symbols, paying full table overhead
///    for ~0 % savings. OnPair's pair-frequency counter never hits the
///    merge threshold, so it stays at 16 bits/token ≈ 2× input.
/// 2. **`period9`** — a randomly chosen 9-byte motif repeated 3-7 times.
///    FSST's symbol table caps individual symbols at 8 bytes, so it can
///    capture *part* of the motif but always needs an escape or seam at
///    byte 9. OnPair16 is similarly bounded by `MAX_TOKEN_SIZE = 16`, so
///    it can swallow one motif but not stitch two together cheaply. The
///    LPM trainer's randomness also means the "winning" alignment differs
///    across runs.
/// 3. **`hex`** — a 40-character random hex blob (think SHA-1). Distinct
///    alphabet from the base64 rows. FSST learns 1-byte hex digits but no
///    pair fires often enough to beat 1:1. OnPair often merges `[0-9a-f]`
///    pairs and beats FSST, but never recovers training cost on the
///    dictionary header.
/// 4. **`ascii`** — random printable ASCII drawn uniformly from the
///    95-character set, 8-24 chars long. The widest alphabet of the four;
///    no two rows share any 3-byte substring with high probability.
///
/// Because the four sub-patterns share no symbols, the trained dictionary
/// is forced to spend slots on each population, leaving none with high
/// enough frequency to amortise its own cost.
pub fn adversarial_mix(scale: usize) -> Corpus {
    let mut rng = StdRng::seed_from_u64(0xADBADBAD);
    let base64_alphabet: &[u8] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let hex_alphabet: &[u8] = b"0123456789abcdef";
    // Printable ASCII: 0x20 (space) through 0x7E (~). The 95-char alphabet
    // is wider than `random_alnum`'s 64-char one and includes `\\`, `"`,
    // `{`, etc. — characters that often anchor multi-byte symbols on other
    // corpora but never recur enough here.
    let ascii_printable: Vec<u8> = (0x20u8..=0x7Eu8).collect();

    let strings: Vec<Vec<u8>> = (0..scale)
        .map(|i| match i & 0b11 {
            0 => {
                // session: 22-char base64-shaped id
                (0..22)
                    .map(|_| {
                        *base64_alphabet
                            .choose(&mut rng)
                            .expect("alphabet is non-empty")
                    })
                    .collect()
            }
            1 => {
                // period9: 9-byte random motif, repeated 3..=7 times
                let motif: Vec<u8> = (0..9)
                    .map(|_| {
                        *base64_alphabet
                            .choose(&mut rng)
                            .expect("alphabet is non-empty")
                    })
                    .collect();
                let reps = rng.random_range(3..=7);
                let mut buf = Vec::with_capacity(motif.len() * reps);
                for _ in 0..reps {
                    buf.extend_from_slice(&motif);
                }
                buf
            }
            2 => {
                // hex: 40-char lowercase-hex blob
                (0..40)
                    .map(|_| {
                        *hex_alphabet
                            .choose(&mut rng)
                            .expect("alphabet is non-empty")
                    })
                    .collect()
            }
            _ => {
                // ascii: variable-length printable
                let len = rng.random_range(8..=24);
                (0..len)
                    .map(|_| {
                        *ascii_printable
                            .choose(&mut rng)
                            .expect("alphabet is non-empty")
                    })
                    .collect()
            }
        })
        .collect();

    Corpus {
        name: "adversarial_mix",
        strings,
        // The needles only ever match if a session/hex/printable row
        // randomly happens to include the substring; that's roughly what we
        // want — a low-selectivity predicate that forces the pushdown path
        // to walk every row.
        needles: vec![b"abc".to_vec(), b"xyz".to_vec(), b"123".to_vec()],
    }
}
