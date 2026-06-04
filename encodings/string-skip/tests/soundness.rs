// SPDX-License-Identifier: Apache-2.0
//! **Soundness invariant**: if a predicate truly matches any row in a
//! chunk, the skip index MUST return `true` for that chunk. False
//! positives are allowed; false negatives are bugs.
//!
//! These property tests generate random rows, build all the skip
//! indexes, and verify no FN across a large random predicate space.

use proptest::collection::vec;
use proptest::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use string_skip::BigramTiers;
use string_skip::ChunkStats;
use string_skip::DictIndex;
use string_skip::DictPresence;
use string_skip::HybridBloom;
use string_skip::Pred;
use string_skip::TieredBloom;
use string_skip::UbiquitousBigrams;
use string_skip::chunk_might_match;
use string_skip::dict::TokenDict;
use string_skip::dict::tokenize_needle;
use string_skip::prune::ChunkSkipState;

/// A minimal test dict + code stream. The dict always includes all 256
/// single-byte tokens plus the supplied multi-byte tokens, sorted.
struct TestColumn {
    dict: TestDict,
    index: DictIndex,
    codes: Vec<u16>,
    offsets: Vec<u32>,
    rows: Vec<Vec<u8>>,
}

struct TestDict {
    toks: Vec<Vec<u8>>,
}
impl TestDict {
    fn new(extras: Vec<&str>) -> Self {
        let mut toks: Vec<Vec<u8>> = (0..=255u8).map(|b| vec![b]).collect();
        for e in extras {
            toks.push(e.as_bytes().to_vec());
        }
        toks.sort();
        toks.dedup();
        Self { toks }
    }
}
impl TokenDict for TestDict {
    fn len(&self) -> usize {
        self.toks.len()
    }
    fn token_bytes(&self, id: u16) -> &[u8] {
        &self.toks[id as usize]
    }
}

impl TestColumn {
    fn build(rows: Vec<Vec<u8>>, extras: Vec<&str>) -> Self {
        let dict = TestDict::new(extras);
        let index = DictIndex::build(&dict);
        let mut codes = Vec::new();
        let mut offsets = vec![0u32];
        for row in &rows {
            let toks = tokenize_needle(&dict, &index, row).expect("tokenize");
            codes.extend(toks);
            offsets.push(codes.len() as u32);
        }
        Self {
            dict,
            index,
            codes,
            offsets,
            rows,
        }
    }
}

/// Pure-byte generator: alphanumeric ASCII, ASCII punct, ASCII whitespace.
fn ascii_byte() -> impl Strategy<Value = u8> {
    prop_oneof![
        Just(b'/'),
        Just(b'.'),
        Just(b'_'),
        Just(b'-'),
        Just(b' '),
        Just(b'a'),
        Just(b'b'),
        Just(b'c'),
        (b'a'..=b'z'),
        (b'A'..=b'Z'),
        (b'0'..=b'9'),
    ]
}

fn arb_row(min_len: usize, max_len: usize) -> impl Strategy<Value = Vec<u8>> {
    vec(ascii_byte(), min_len..=max_len)
}

fn arb_rows(n: usize) -> impl Strategy<Value = Vec<Vec<u8>>> {
    vec(arb_row(3, 30), n..=n)
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 50,
        max_shrink_iters: 100,
        .. ProptestConfig::default()
    })]

    /// **The soundness invariant**: if any row truly matches, every
    /// skip index must say `might_match = true`.
    #[test]
    fn soundness_chunk_might_match(
        rows in arb_rows(50),
    ) {
        let col = TestColumn::build(rows.clone(), vec!["http://", "www.", ".com", ".org"]);

        let chunk_stats = ChunkStats::from_rows(&col.rows);
        let presence = DictPresence::build(&col.codes, col.dict.len());
        let ubiq = UbiquitousBigrams::empty();
        let tiers = BigramTiers::empty();
        let bloom = HybridBloom::build(
            &col.codes, &col.offsets, 0, col.rows.len(), 16, &ubiq);
        let tiered = TieredBloom::build(
            &col.codes, &col.offsets, 0, col.rows.len(), 16, &tiers);

        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xc0ffee);

        // Generate ~20 predicates and check each
        for _ in 0..20 {
            let pred = make_pred_from_rows(&mut rng, &col.rows);
            let truly = pred.matches_any(&col.rows);

            let state = ChunkSkipState {
                stats: &chunk_stats,
                presence: &presence,
                bloom: Some(&bloom),
                tiered: None,
                ubiq: &ubiq,
                tiers: &tiers,
                dict: &col.dict,
                index: &col.index,
            };
            let pred_says = chunk_might_match(&pred, &state);
            prop_assert!(
                !truly || pred_says,
                "FN with HybridBloom: pred = {pred:?}, truly = {truly}, says = {pred_says}"
            );

            let state_t = ChunkSkipState {
                stats: &chunk_stats,
                presence: &presence,
                bloom: None,
                tiered: Some(&tiered),
                ubiq: &ubiq,
                tiers: &tiers,
                dict: &col.dict,
                index: &col.index,
            };
            let pred_says = chunk_might_match(&pred, &state_t);
            prop_assert!(
                !truly || pred_says,
                "FN with TieredBloom: pred = {pred:?}, truly = {truly}, says = {pred_says}"
            );
        }
    }
}

/// Hand-rolled predicate generator (proptest's flat_map gets unwieldy
/// for such a heterogeneous AST).
fn make_pred_from_rows<R: Rng>(rng: &mut R, rows: &[Vec<u8>]) -> Pred {
    let kind = rng.random_range(0..14u8);
    let row = &rows[rng.random_range(0..rows.len())];
    let off = if row.len() <= 1 {
        0
    } else {
        rng.random_range(0..row.len() - 1)
    };
    let len = if row.len() <= off + 1 {
        1
    } else {
        rng.random_range(1..=row.len() - off).min(10)
    };
    let slice = row[off..off + len].to_vec();
    match kind {
        0 => Pred::Eq(row.clone()),
        1 => Pred::Lt(row.clone()),
        2 => Pred::Gt(row.clone()),
        3 => {
            let other = &rows[rng.random_range(0..rows.len())];
            let (lo, hi) = if row <= other {
                (row.clone(), other.clone())
            } else {
                (other.clone(), row.clone())
            };
            Pred::Between(lo, hi)
        }
        4 => Pred::Prefix(row[..len.min(row.len())].to_vec()),
        5 => Pred::Suffix(row[row.len().saturating_sub(len)..].to_vec()),
        6 => Pred::Contains(slice.clone()),
        7 if slice.len() >= 3 => {
            let mid = slice.len() / 2;
            Pred::PrefixSuffix(slice[..mid].to_vec(), slice[mid..].to_vec())
        }
        8 if slice.len() >= 3 => {
            let half = slice.len() / 2;
            Pred::SingleWildcard(
                slice[..half.saturating_sub(1).max(1)].to_vec(),
                slice[half + 1.min(slice.len() - half)..].to_vec(),
            )
        }
        9 => Pred::LengthGt(rng.random_range(0..50)),
        10 => Pred::LengthBetween(rng.random_range(0..30), rng.random_range(20..60)),
        11 => Pred::IsNotNull,
        12 => {
            let n = rng.random_range(1..=5);
            let vals: Vec<Vec<u8>> = (0..n)
                .map(|_| rows[rng.random_range(0..rows.len())].clone())
                .collect();
            Pred::InSet(vals)
        }
        _ => Pred::Contains(slice),
    }
}

/// A specific regression test for the substring/cover-enumeration logic.
#[test]
fn substring_no_fn_with_long_extension_tokens() {
    // Reproduces the AYERS-MIRACLE / iroverlanet.ru/ failure mode where
    // a long dict token extends past the needle in the actual row.
    let rows: Vec<Vec<u8>> = vec![
        b"http://www.adidas.com/men/shoes/iroverlanet.ru/sneakers".to_vec(),
        b"http://www.adidas.com/women/dresses".to_vec(),
    ];
    // Force a multi-byte dict token that includes the needle bytes.
    let col = TestColumn::build(
        rows.clone(),
        vec!["http://www.", "adidas.com/", "iroverlanet.ru/"],
    );

    let chunk_stats = ChunkStats::from_rows(&col.rows);
    let presence = DictPresence::build(&col.codes, col.dict.len());
    let ubiq = UbiquitousBigrams::empty();
    let tiers = BigramTiers::empty();
    let bloom = HybridBloom::build(&col.codes, &col.offsets, 0, col.rows.len(), 16, &ubiq);

    // We don't actually need that substring to match; the key check
    // is that whatever DOES match returns true. Try several:
    for needle in [&b"iroverlanet"[..], &b"adidas"[..], &b"sneakers"[..]] {
        let p = Pred::Contains(needle.to_vec());
        let truly = p.matches_any(&col.rows);
        let state = ChunkSkipState {
            stats: &chunk_stats,
            presence: &presence,
            bloom: Some(&bloom),
            tiered: None,
            ubiq: &ubiq,
            tiers: &tiers,
            dict: &col.dict,
            index: &col.index,
        };
        let says = chunk_might_match(&p, &state);
        assert!(
            !truly || says,
            "FN for needle {:?}: truly={truly}, says={says}",
            std::str::from_utf8(needle).unwrap()
        );
    }
}
