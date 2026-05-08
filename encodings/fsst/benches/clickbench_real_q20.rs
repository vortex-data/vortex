// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Real ClickBench Q20/Q22 bench: `LIKE '%google%'` over the actual
//! `hits_0.parquet` URL column. Mirrors the synthetic
//! `clickbench_url_google` bench but uses the upstream ClickBench
//! corpus (~1M rows in shard 0). Set `CLICKBENCH_HITS_0` to the path of
//! `hits_0.parquet`; defaults to `/home/ec2-user/clickbench-data/hits_0.parquet`.

#![expect(clippy::unwrap_used)]

use std::env;
use std::fs::File;
use std::path::PathBuf;
use std::sync::LazyLock;

use arrow_array::Array as _;
use arrow_array::StringArray;
use divan::Bencher;
use memchr::memmem::Finder;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::ProjectionMask;
use vortex_array::Canonical;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::scalar_fn::ScalarFnFactoryExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar_fn::fns::like::Like;
use vortex_array::scalar_fn::fns::like::LikeOptions;
use vortex_array::session::ArraySession;
use vortex_fsst::FSSTArray;
use vortex_fsst::dfa::FsstMatcher;
use vortex_fsst::FSSTArrayExt;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_session::VortexSession;

fn main() {
    divan::main();
}

fn hits_0_path() -> PathBuf {
    env::var_os("CLICKBENCH_HITS_0")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/home/ec2-user/clickbench-data/hits_0.parquet"))
}

/// Load the URL column from `hits_0.parquet` into a flat `Vec<String>`.
/// Run once and cached. Skips rows with NULL URLs.
fn load_real_urls() -> Vec<String> {
    let path = hits_0_path();
    let file = File::open(&path)
        .unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();

    // Find the URL column index.
    let schema = builder.schema().clone();
    let url_idx = schema
        .fields()
        .iter()
        .position(|f| f.name() == "URL")
        .expect("hits parquet has a URL column");
    let mask = ProjectionMask::roots(builder.parquet_schema(), [url_idx]);
    let reader = builder.with_projection(mask).build().unwrap();

    let mut out: Vec<String> = Vec::with_capacity(1_000_000);
    for batch in reader {
        let batch = batch.unwrap();
        let col = batch.column(0);
        let strs = col
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("URL column is StringArray");
        for i in 0..strs.len() {
            if strs.is_null(i) {
                continue;
            }
            out.push(strs.value(i).to_string());
        }
    }
    out
}

static URLS: LazyLock<Vec<String>> = LazyLock::new(load_real_urls);

static FSST_URLS: LazyLock<FSSTArray> = LazyLock::new(|| {
    let urls: &Vec<String> = &URLS;
    let varbin = VarBinArray::from_iter(
        urls.iter().map(|s| Some(s.as_str())),
        DType::Utf8(Nullability::NonNullable),
    );
    let compressor = fsst_train_compressor(&varbin);
    let len = varbin.len();
    let dtype = varbin.dtype().clone();
    fsst_compress(varbin, len, &dtype, &compressor)
});

static DECOMPRESSED_CONCAT: LazyLock<Vec<u8>> = LazyLock::new(|| {
    let total: usize = URLS.iter().map(|s| s.len()).sum();
    let mut out = Vec::with_capacity(total);
    for s in URLS.iter() {
        out.extend_from_slice(s.as_bytes());
    }
    out
});

static SESSION: LazyLock<VortexSession> =
    LazyLock::new(|| VortexSession::empty().with::<ArraySession>());

const PATTERN: &str = "%google%";
const NEEDLE: &[u8] = b"google";

/// Real-data analogue of `clickbench_url_google::like_google_full`:
/// ClickBench Q20/Q22 (`SELECT COUNT(*) FROM hits WHERE URL LIKE
/// '%google%'`) on the actual ClickBench `hits_0.parquet` URL column.
#[divan::bench]
fn like_google_full_real(bencher: Bencher) {
    let fsst = &*FSST_URLS;
    let len = fsst.len();
    let arr = fsst.clone().into_array();
    let pat = ConstantArray::new(PATTERN, len).into_array();
    bencher
        .with_inputs(|| SESSION.create_execution_ctx())
        .bench_refs(|ctx| {
            Like.try_new_array(len, LikeOptions::default(), [arr.clone(), pat.clone()])
                .unwrap()
                .into_array()
                .execute::<Canonical>(ctx)
                .unwrap()
        });
}

/// Per-string DFA matching, no bitbuf packing — analogue of
/// `clickbench_url_google::dfa_inner_only` on real data.
#[divan::bench]
fn dfa_inner_only_real(bencher: Bencher) {
    #[expect(deprecated)]
    use vortex_array::ToCanonical;
    use vortex_array::arrays::varbin::VarBinArrayExt;

    let fsst = &*FSST_URLS;
    let n = fsst.len();
    let view = fsst.as_view();
    let codes = view.codes();
    #[expect(deprecated)]
    let offsets = codes.offsets().to_primitive();
    let offsets: Vec<u32> = offsets.as_slice::<i32>().iter().map(|&v| v as u32).collect();
    let bytes = codes.bytes().as_slice().to_vec();
    let matcher = FsstMatcher::try_new(
        view.symbols().as_slice(),
        view.symbol_lengths().as_slice(),
        b"%google%",
    )
    .unwrap()
    .unwrap();
    bencher.bench_local(|| {
        let mut count: u64 = 0;
        let mut start = offsets[0] as usize;
        for i in 0..n {
            let end = offsets[i + 1] as usize;
            if matcher.matches(&bytes[start..end]) {
                count += 1;
            }
            start = end;
        }
        count
    });
}

/// One global memmem over the entire concatenated decompressed corpus —
/// theoretical floor for the matching work alone on real data.
#[divan::bench]
fn memmem_concat_corpus_real(bencher: Bencher) {
    let bytes = &*DECOMPRESSED_CONCAT;
    let finder = Finder::new(NEEDLE);
    bencher.bench_local(|| finder.find_iter(bytes).count() as u64);
}

