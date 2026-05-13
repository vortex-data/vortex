// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! String-corpus generators (synthetic + real-world).
//!
//! Each synthetic generator produces a deterministic seeded corpus that
//! exercises a different property a string compressor cares about (skewed
//! dictionaries, long shared prefixes, random noise, URL-shaped strings,
//! fragmented bag-of-words, etc.). The companion [`real_world`] module
//! adds loaders for the vendored corpora under `data/`.

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;

pub mod real_world;

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

/// Build the suite of every dataset (synthetic + vendored real-world) used
/// by the bench / report run. Real-world corpora that fail to load (file
/// missing) are silently dropped so a slim checkout still produces a
/// usable report.
pub fn all_datasets(scale: usize) -> Vec<Corpus> {
    let mut all = vec![
        skewed_dictionary(scale),
        url_like(scale),
        random_bytes(scale),
        long_shared_prefix(scale),
        natural_words(scale),
        json_like(scale),
        short_codes(scale),
        high_cardinality_enum(scale),
        log_templates(scale),
        adversarial_mix(scale),
        real_world::pride_and_prejudice(scale),
        real_world::english_words(scale),
        real_world::gov_hostnames(scale),
        real_world::airport_records(scale),
        real_world::world_cities(scale),
    ];
    all.retain(|c| !c.is_empty());
    all
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

/// FSST-12 sweet spot: a 512-entry vocabulary of 3-5-byte tokens, each row
/// is 5-10 tokens separated by `|`. FSST-8 caps at 255 symbols so it can
/// keep at most ~half the vocabulary in its table and falls back to
/// byte-level codes for the rest. FSST-12 (≤4096 symbols) fits the entire
/// vocabulary and represents each token as a single 1.5-byte code, beating
/// every other backend on ratio.
pub fn high_cardinality_enum(scale: usize) -> Corpus {
    let mut rng = StdRng::seed_from_u64(0xC0DEC0DE);
    let mut vocab: Vec<String> = Vec::with_capacity(512);
    // Two-letter prefix (676 combinations) + two-digit suffix gives plenty
    // of headroom; we keep only the first 512 unique entries.
    let mut seen = std::collections::HashSet::with_capacity(512);
    let letters: Vec<u8> = (b'a'..=b'z').collect();
    while vocab.len() < 512 {
        let a = *letters.choose(&mut rng).unwrap();
        let b = *letters.choose(&mut rng).unwrap();
        let n: u16 = rng.random_range(10..100);
        let s = format!("{}{}{n:02}", a as char, b as char);
        if seen.insert(s.clone()) {
            vocab.push(s);
        }
    }

    let strings: Vec<Vec<u8>> = (0..scale)
        .map(|_| {
            let n = rng.random_range(5..=10);
            let mut buf = Vec::with_capacity(n * 5);
            for i in 0..n {
                if i > 0 {
                    buf.push(b'|');
                }
                buf.extend_from_slice(vocab.choose(&mut rng).unwrap().as_bytes());
            }
            buf
        })
        .collect();

    let needle = vocab[0].as_bytes().to_vec();
    let prefix_needle = vocab[3].as_bytes()[..2].to_vec();
    Corpus { name: "fsst12_high_card", strings, needles: vec![needle, prefix_needle] }
}

/// OnPair (no token-size cap) sweet spot: synthetic structured log lines
/// where a ~250-byte template appears verbatim on most rows, with only a
/// handful of short variable fields filled in.
///
/// FSST's symbol table caps each *symbol* at 8 bytes so the long template
/// degrades into ~30 chained 8-byte codes per row. OnPair16 caps tokens at
/// 16 bytes, requiring ~16 codes. OnPair has no cap — once the LPM trainer
/// stitches the full template into a single dictionary entry, every
/// occurrence costs exactly one 16-bit token (2 bytes).
///
/// The template includes a long Lorem-Ipsum-style suffix specifically so
/// that the bytes saved by capturing one giant token dominate the cost of
/// storing it once in the dictionary header.
pub fn log_templates(scale: usize) -> Corpus {
    let mut rng = StdRng::seed_from_u64(0x10610610);
    // One pronounced template — the more it recurs, the cleaner OnPair's
    // win. Including a long quasi-natural suffix beyond the structured log
    // header pushes the captured-token length well past FSST's 8-byte
    // symbol cap and OnPair16's 16-byte token cap.
    let template: &[u8] =
        b"[2026-05-13T17:42:00.000000Z] [service=vortex-ingest] [region=us-east-1] [pod=ingest-7f9c4-r2s8q] [trace_id=00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01] [REQUEST] method=GET path=/v1/users/profile/preferences subsystem=preferences result=%STATUS% id=%ID%";
    let statuses = [b"ok".as_slice(), b"throttled", b"rejected", b"timeout"];
    let strings: Vec<Vec<u8>> = (0..scale)
        .map(|_| {
            let id: u64 = rng.random();
            let status = *statuses.choose(&mut rng).unwrap();
            // Walk the template substituting placeholders. We avoid format!
            // here because doing the substitution by hand keeps the
            // surrounding template bytes byte-identical across rows, which
            // is the whole point.
            let mut out = Vec::with_capacity(template.len() + 32);
            let mut i = 0;
            while i < template.len() {
                if template[i..].starts_with(b"%ID%") {
                    out.extend_from_slice(id.to_string().as_bytes());
                    i += b"%ID%".len();
                } else if template[i..].starts_with(b"%STATUS%") {
                    out.extend_from_slice(status);
                    i += b"%STATUS%".len();
                } else {
                    out.push(template[i]);
                    i += 1;
                }
            }
            out
        })
        .collect();

    Corpus {
        name: "log_templates",
        strings,
        needles: vec![
            b"[REQUEST]".to_vec(),
            b"trace_id=00".to_vec(),
            b"result=throttled".to_vec(),
        ],
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
