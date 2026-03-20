// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Direct comparison benchmark: Arrow LIKE on strings vs FSST DFA LIKE on compressed codes.
//!
//! Three benchmark groups:
//! 1. **arrow_prefix / fsst_prefix** — same data, arrow operates on VarBinView, FSST on codes
//! 2. **arrow_contains / fsst_contains** — same, for `%needle%` patterns
//! 3. **symlen_prefix / symlen_contains** — sweep across different average symbol lengths
//!    (controlled by varying alphabet entropy in generated data)

#![allow(clippy::unwrap_used, dead_code)]

use std::fmt;
use std::sync::LazyLock;

use divan::Bencher;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::scalar_fn::ScalarFnArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::session::ArraySession;
use vortex_fsst::FSSTArray;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_fsst::test_utils::NUM_STRINGS;
use vortex_fsst::test_utils::generate_clickbench_urls;
use vortex_fsst::test_utils::generate_emails;
use vortex_fsst::test_utils::generate_file_paths;
use vortex_fsst::test_utils::generate_log_lines;
use vortex_fsst::test_utils::generate_short_urls;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

const N: usize = NUM_STRINGS;

// ---------------------------------------------------------------------------
// Real-world datasets: FSST arrays + their uncompressed VarBinView counterparts
// ---------------------------------------------------------------------------

struct DatasetPair {
    fsst: FSSTArray,
    arrow: ArrayRef,
}

fn make_pair_from_strings(strings: &[String]) -> DatasetPair {
    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    let fsst = fsst_compress(varbin.clone(), &compressor);
    // Use VarBinArray directly as the arrow baseline — the `arrow_like` fallback
    // converts to Arrow StringArray internally.
    let arrow = varbin.into_array();
    DatasetPair { fsst, arrow }
}

static PAIR_URLS: LazyLock<DatasetPair> =
    LazyLock::new(|| make_pair_from_strings(&generate_short_urls(N)));
static PAIR_CB: LazyLock<DatasetPair> =
    LazyLock::new(|| make_pair_from_strings(&generate_clickbench_urls(N)));
static PAIR_LOG: LazyLock<DatasetPair> =
    LazyLock::new(|| make_pair_from_strings(&generate_log_lines(N)));
static PAIR_PATH: LazyLock<DatasetPair> =
    LazyLock::new(|| make_pair_from_strings(&generate_file_paths(N)));
static PAIR_EMAIL: LazyLock<DatasetPair> =
    LazyLock::new(|| make_pair_from_strings(&generate_emails(N)));

// ---------------------------------------------------------------------------
// Dataset enum for divan args
// ---------------------------------------------------------------------------

enum Dataset {
    Urls,
    Cb,
    Log,
    Path,
    Email,
}

impl fmt::Display for Dataset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Urls => f.write_str("urls"),
            Self::Cb => f.write_str("cb_urls"),
            Self::Log => f.write_str("log"),
            Self::Path => f.write_str("path"),
            Self::Email => f.write_str("email"),
        }
    }
}

impl Dataset {
    fn pair(&self) -> &'static DatasetPair {
        match self {
            Self::Urls => &PAIR_URLS,
            Self::Cb => &PAIR_CB,
            Self::Log => &PAIR_LOG,
            Self::Path => &PAIR_PATH,
            Self::Email => &PAIR_EMAIL,
        }
    }

    fn prefix_pattern(&self) -> &'static str {
        match self {
            Self::Urls => "https%",
            Self::Cb => "https://www.%",
            Self::Log => "192.168%",
            Self::Path => "/home%",
            Self::Email => "john%",
        }
    }

    fn contains_pattern(&self) -> &'static str {
        match self {
            Self::Urls => "%google%",
            Self::Cb => "%yandex%",
            Self::Log => "%Googlebot%",
            Self::Path => "%target/release%",
            Self::Email => "%gmail%",
        }
    }
}

// ---------------------------------------------------------------------------
// Bench helpers
// ---------------------------------------------------------------------------

fn bench_like_on(bencher: Bencher, array: &ArrayRef, pattern: &str) {
    let len = array.len();
    let pat = ConstantArray::new(pattern, len).into_array();
    bencher.bench_local(|| {
        Like.try_new_array(len, LikeOptions::default(), [array.clone(), pat.clone()])
            .unwrap()
            .into_array()
            .execute::<Canonical>(&mut SESSION.create_execution_ctx())
            .unwrap()
    });
}

// ---------------------------------------------------------------------------
// Group 1: Arrow LIKE vs FSST DFA LIKE — prefix
// ---------------------------------------------------------------------------

#[divan::bench(args = [
    Dataset::Urls, Dataset::Cb, Dataset::Log, Dataset::Path, Dataset::Email,
])]
fn arrow_prefix(bencher: Bencher, dataset: &Dataset) {
    let pair = dataset.pair();
    bench_like_on(bencher, &pair.arrow, dataset.prefix_pattern());
}

#[divan::bench(args = [
    Dataset::Urls, Dataset::Cb, Dataset::Log, Dataset::Path, Dataset::Email,
])]
fn fsst_prefix(bencher: Bencher, dataset: &Dataset) {
    let pair = dataset.pair();
    let fsst_arr = pair.fsst.clone().into_array();
    bench_like_on(bencher, &fsst_arr, dataset.prefix_pattern());
}

// ---------------------------------------------------------------------------
// Group 2: Arrow LIKE vs FSST DFA LIKE — contains
// ---------------------------------------------------------------------------

#[divan::bench(args = [
    Dataset::Urls, Dataset::Cb, Dataset::Log, Dataset::Path, Dataset::Email,
])]
fn arrow_contains(bencher: Bencher, dataset: &Dataset) {
    let pair = dataset.pair();
    bench_like_on(bencher, &pair.arrow, dataset.contains_pattern());
}

#[divan::bench(args = [
    Dataset::Urls, Dataset::Cb, Dataset::Log, Dataset::Path, Dataset::Email,
])]
fn fsst_contains(bencher: Bencher, dataset: &Dataset) {
    let pair = dataset.pair();
    let fsst_arr = pair.fsst.clone().into_array();
    bench_like_on(bencher, &fsst_arr, dataset.contains_pattern());
}

// ---------------------------------------------------------------------------
// Group 3: Symbol length sweep — controlled entropy
//
// We generate strings with varying `unique_chars` parameter:
//   - fewer unique chars → more repetition → longer symbols → better compression
//   - more unique chars → higher entropy → shorter symbols → worse compression
//
// All strings have the same average length (40 bytes) and we use the same
// pattern for all: prefix "aaa%" and contains "%aab%".
// ---------------------------------------------------------------------------

/// Represents a controlled-entropy dataset for the symbol length sweep.
struct SymlenData {
    fsst: FSSTArray,
    arrow: ArrayRef,
    mean_sym_len: f64,
    n_symbols: usize,
}

fn make_symlen_data(unique_chars: u8) -> SymlenData {
    use rand::RngExt;
    use rand::SeedableRng;
    use rand::prelude::StdRng;

    let mut rng = StdRng::seed_from_u64(0);
    let avg_str_len = 40usize;
    let strings: Vec<Option<Box<[u8]>>> = (0..N)
        .map(|_| {
            let len = avg_str_len * rng.random_range(50..=150) / 100;
            Some(
                (0..len)
                    .map(|_| rng.random_range(b'a'..(b'a' + unique_chars)))
                    .collect::<Vec<u8>>()
                    .into_boxed_slice(),
            )
        })
        .collect();

    let varbin = VarBinArray::from_iter(
        strings.iter().map(|s| s.as_deref()),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    let fsst = fsst_compress(varbin.clone(), &compressor);

    let sym_lengths = fsst.symbol_lengths();
    let n_symbols = sym_lengths.len();
    let mean_sym_len = if n_symbols > 0 {
        sym_lengths.iter().map(|&l| l as f64).sum::<f64>() / n_symbols as f64
    } else {
        0.0
    };

    let arrow = varbin.into_array();
    SymlenData {
        fsst,
        arrow,
        mean_sym_len,
        n_symbols,
    }
}

/// Entropy levels: unique_chars controls alphabet size.
/// 4 chars → ~4 byte avg symbols (great compression)
/// 8 chars → ~3 byte avg symbols (good compression)
/// 16 chars → ~2 byte avg symbols (moderate compression)
/// 26 chars → ~1.5 byte avg symbols (poor compression)
enum Entropy {
    Low4,
    Med8,
    High16,
    Max26,
}

impl fmt::Display for Entropy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low4 => f.write_str("4chr"),
            Self::Med8 => f.write_str("8chr"),
            Self::High16 => f.write_str("16chr"),
            Self::Max26 => f.write_str("26chr"),
        }
    }
}

impl Entropy {
    fn unique_chars(&self) -> u8 {
        match self {
            Self::Low4 => 4,
            Self::Med8 => 8,
            Self::High16 => 16,
            Self::Max26 => 26,
        }
    }
}

static SYMLEN_4: LazyLock<SymlenData> = LazyLock::new(|| make_symlen_data(4));
static SYMLEN_8: LazyLock<SymlenData> = LazyLock::new(|| make_symlen_data(8));
static SYMLEN_16: LazyLock<SymlenData> = LazyLock::new(|| make_symlen_data(16));
static SYMLEN_26: LazyLock<SymlenData> = LazyLock::new(|| make_symlen_data(26));

impl Entropy {
    fn data(&self) -> &'static SymlenData {
        match self {
            Self::Low4 => &SYMLEN_4,
            Self::Med8 => &SYMLEN_8,
            Self::High16 => &SYMLEN_16,
            Self::Max26 => &SYMLEN_26,
        }
    }
}

const SYMLEN_PREFIX_PATTERN: &str = "aaa%";
const SYMLEN_CONTAINS_PATTERN: &str = "%aab%";

// --- Symbol length sweep: prefix ---

#[divan::bench(args = [Entropy::Low4, Entropy::Med8, Entropy::High16, Entropy::Max26])]
fn symlen_arrow_prefix(bencher: Bencher, entropy: &Entropy) {
    let data = entropy.data();
    bench_like_on(bencher, &data.arrow, SYMLEN_PREFIX_PATTERN);
}

#[divan::bench(args = [Entropy::Low4, Entropy::Med8, Entropy::High16, Entropy::Max26])]
fn symlen_fsst_prefix(bencher: Bencher, entropy: &Entropy) {
    let data = entropy.data();
    let arr = data.fsst.clone().into_array();
    bench_like_on(bencher, &arr, SYMLEN_PREFIX_PATTERN);
}

// --- Symbol length sweep: contains ---

#[divan::bench(args = [Entropy::Low4, Entropy::Med8, Entropy::High16, Entropy::Max26])]
fn symlen_arrow_contains(bencher: Bencher, entropy: &Entropy) {
    let data = entropy.data();
    bench_like_on(bencher, &data.arrow, SYMLEN_CONTAINS_PATTERN);
}

#[divan::bench(args = [Entropy::Low4, Entropy::Med8, Entropy::High16, Entropy::Max26])]
fn symlen_fsst_contains(bencher: Bencher, entropy: &Entropy) {
    let data = entropy.data();
    let arr = data.fsst.clone().into_array();
    bench_like_on(bencher, &arr, SYMLEN_CONTAINS_PATTERN);
}
