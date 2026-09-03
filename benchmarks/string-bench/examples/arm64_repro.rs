// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reproduction of the string-codec length mismatch seen when the benchmarks moved to
//! `c7gd.metal` (Graviton3): reading a Vortex file back fails with
//! `OnPair codes decode to more bytes than uncompressed_lengths records` or
//! `FSST decoded N bytes, expected M` / `output buffer sized too small`.
//!
//! Root cause (see the dependency-free `sve_widening_sum` example for the 40-line version):
//! with `-C target-cpu=native` on Graviton3, i.e. `neoverse-v1`, rustc 1.98 / LLVM 22 miscompiles
//! a widening `u8 -> usize` sum whenever the SVE vector length is 256 bits or more. Both
//! `OnPairDecodePlan::new` and `FsstDecodePlan::new` size their output buffer with exactly that
//! sum over the `u8` `uncompressed_lengths` child of arrays read back from a file, so the buffer
//! comes out about half the needed size. In-memory arrays keep `i32` lengths and are unaffected,
//! which is why only the file stages below fail. Building with `-C target-feature=-sve,-sve2`
//! avoids it.
//!
//! The same column is pushed through five independent stages so the failing layer is
//! visible at a glance:
//!
//! 1. `onpair-memory`: `onpair_compress` then canonicalize, no file involved.
//! 2. `fsst-memory`: `fsst_compress` then canonicalize, no file involved.
//! 3. `onpair-file`: write an in-memory Vortex file forcing the OnPair scheme, read it back,
//!    canonicalize every row split.
//! 4. `fsst-file`: same with the FSST scheme forced.
//! 5. `default-file`: same with the default btrblocks compressor (whatever it picks).
//!
//! Every stage runs even if an earlier one fails. A failing file stage prints the encoding
//! tree of the first row split that does not decode. `--dump DIR` additionally writes every
//! decoded integer child of each row split, with its widening sum on the first line, so two
//! runs can be diffed. The process exits non-zero if any stage fails.
//!
//! Build exactly as the benchmark jobs do and run against the ClickBench `URL` column:
//!
//! ```text
//! RUSTFLAGS="-C target-cpu=native" cargo build --profile release_debug \
//!     -p string-bench --example arm64_repro --features unstable_encodings
//! target/release_debug/examples/arm64_repro                # full shard 0 (~1M rows)
//! target/release_debug/examples/arm64_repro --rows 100000  # first 100k rows only
//! target/release_debug/examples/arm64_repro --shard 3
//! ```
//!
//! Off Graviton hardware, cross-compile with `-C target-cpu=neoverse-v1` and run under
//! `qemu-aarch64 -cpu max,sve-default-vector-length=32` (256-bit SVE, Graviton3's width); the
//! same binary passes with `sve-default-vector-length=16`.

use std::fmt::Write as _;
use std::io::Cursor;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use bytes::Bytes;
use clap::Parser;
use futures::TryStreamExt;
use num_traits::AsPrimitive;
use string_bench::SESSION;
use string_bench::load_clickbench_url;
use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::builtins::ArrayBuiltins;
use vortex::array::dtype::DType;
use vortex::array::dtype::PType;
use vortex::array::match_each_integer_ptype;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::layout::LayoutStrategy;
use vortex_bench::benchmark_write_options;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::SchemeId;
use vortex_btrblocks::schemes::string::FSSTScheme;
use vortex_btrblocks::schemes::string::NullDominatedSparseScheme;
use vortex_btrblocks::schemes::string::OnPairScheme;
use vortex_btrblocks::schemes::string::StringDictScheme;
use vortex_fsst::FSST;
use vortex_fsst::FSSTArraySlotsExt;
use vortex_fsst::fsst_compress;
use vortex_fsst::fsst_train_compressor;
use vortex_onpair::DEFAULT_CONFIG;
use vortex_onpair::OnPair;
use vortex_onpair::OnPairArraySlotsExt;
use vortex_onpair::onpair_compress;

#[derive(Parser, Debug)]
struct Args {
    /// ClickBench `hits` shard whose `URL` column is used as input.
    #[arg(long, default_value_t = 0)]
    shard: u32,
    /// Only use the first N rows of the column. Useful to bisect a failure down.
    #[arg(long)]
    rows: Option<usize>,
    /// Directory to write the decoded integer children (`codes`, `codes_offsets`,
    /// `dict_offsets`, `uncompressed_lengths`) of every OnPair row split read back from the
    /// file, one value per line. Diff two runs (e.g. different SVE vector lengths) to find
    /// the child that decodes differently.
    #[arg(long)]
    dump: Option<PathBuf>,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<ExitCode> {
    let args = Args::parse();
    let session = &*SESSION;
    let mut ctx = session.create_execution_ctx();

    let column = load_clickbench_url(args.shard, &mut ctx).await?;
    let mut expected = column.array.clone().execute::<VarBinViewArray>(&mut ctx)?;
    if let Some(rows) = args.rows {
        let rows = rows.min(expected.len());
        expected = expected
            .into_array()
            .slice(0..rows)?
            .execute::<VarBinViewArray>(&mut ctx)?;
    }
    println!(
        "input: {} rows={} dtype={}",
        column.name,
        expected.len(),
        expected.dtype()
    );
    let input = expected.clone().into_array();

    let mut failed = false;
    let mut report = |name: &str, result: Result<()>| match result {
        Ok(()) => println!("PASS {name}"),
        Err(e) => {
            failed = true;
            println!("FAIL {name}: {e:#}");
        }
    };

    report("onpair-memory", {
        let mut ctx = session.create_execution_ctx();
        onpair_compress(&input, DEFAULT_CONFIG, &mut ctx)
            .context("onpair_compress")
            .and_then(|encoded| check_chunks(&expected, &[encoded], &mut ctx))
    });

    report("fsst-memory", {
        let mut ctx = session.create_execution_ctx();
        fsst_train_compressor(&input, &mut ctx)
            .context("fsst_train_compressor")
            .and_then(|compressor| {
                fsst_compress(&input, &compressor, &mut ctx).context("fsst_compress")
            })
            .and_then(|encoded| check_chunks(&expected, &[encoded.into_array()], &mut ctx))
    });

    for (name, strategy) in [
        ("onpair-file", forced_string_scheme(Some(OnPairScheme.id()))),
        ("fsst-file", forced_string_scheme(Some(FSSTScheme.id()))),
        ("default-file", forced_string_scheme(None)),
    ] {
        let mut ctx = session.create_execution_ctx();
        let dump = args.dump.as_ref();
        let result = async {
            let data = write_file(&input, &strategy).await?;
            let chunks = read_file(data).await?;
            if let Some(dir) = dump {
                dump_onpair_children(&dir.join(name), &chunks, &mut ctx)?;
            }
            check_chunks(&expected, &chunks, &mut ctx)
        }
        .await;
        report(name, result);
    }

    Ok(if failed {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    })
}

/// The btrblocks string schemes the default compressor chooses between. Forcing one keeps
/// only that scheme selectable; `None` leaves the default set untouched.
fn forced_string_scheme(forced: Option<SchemeId>) -> Arc<dyn LayoutStrategy> {
    let all = [
        StringDictScheme.id(),
        FSSTScheme.id(),
        OnPairScheme.id(),
        NullDominatedSparseScheme.id(),
    ];
    let mut compressor = BtrBlocksCompressorBuilder::default();
    if let Some(forced) = forced {
        compressor = compressor.exclude_schemes(all.into_iter().filter(|&id| id != forced));
    }
    WriteStrategyBuilder::default()
        .with_btrblocks_builder(compressor)
        .build()
}

/// Write `input` to an in-memory Vortex file with the benchmark's write options.
async fn write_file(input: &ArrayRef, strategy: &Arc<dyn LayoutStrategy>) -> Result<Bytes> {
    let session = &*SESSION;
    let mut buf = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        benchmark_write_options(session.write_options())
            .with_strategy(Arc::clone(strategy))
            .write(&mut cursor, input.to_array_stream())
            .await
            .context("write")?;
    }
    println!("  wrote {} bytes", buf.len());
    Ok(Bytes::from(buf))
}

/// Scan the whole file back as one array per row split, without canonicalizing.
async fn read_file(data: Bytes) -> Result<Vec<ArrayRef>> {
    let session = &*SESSION;
    let file = session.open_options().open_buffer(data).context("open")?;
    let chunks: Vec<ArrayRef> = file
        .scan()
        .context("scan")?
        .into_array_stream()
        .context("stream")?
        .try_collect()
        .await
        .context("collect")?;
    println!("  read {} row splits", chunks.len());
    Ok(chunks)
}

/// Canonicalize each chunk on its own and compare it byte-for-byte against the matching
/// rows of `expected`. Reports the first chunk that fails, with its encoding tree.
fn check_chunks(
    expected: &VarBinViewArray,
    chunks: &[ArrayRef],
    ctx: &mut ExecutionCtx,
) -> Result<()> {
    let mut offset = 0usize;
    for (i, chunk) in chunks.iter().enumerate() {
        let len = chunk.len();
        let canonical = match chunk.clone().execute::<VarBinViewArray>(ctx) {
            Ok(canonical) => canonical,
            Err(e) => {
                bail!(
                    "row split {i} (rows {offset}..{}) failed to canonicalize: {e}\n{}",
                    offset + len,
                    chunk.display_tree()
                );
            }
        };
        if canonical.len() != len {
            bail!(
                "row split {i}: canonical len {} != chunk len {len}",
                canonical.len()
            );
        }
        for row in 0..len {
            let want = expected.bytes_at(offset + row);
            let got = canonical.bytes_at(row);
            if want.as_slice() != got.as_slice() {
                bail!(
                    "row split {i}: row {} differs (want {} bytes, got {} bytes)\n{}",
                    offset + row,
                    want.len(),
                    got.len(),
                    chunk.display_tree()
                );
            }
        }
        offset += len;
    }
    if offset != expected.len() {
        bail!("chunks cover {offset} rows, expected {}", expected.len());
    }
    Ok(())
}

/// Decode the integer children of every OnPair and FSST row split independently and write
/// each as one value per line under `dir`, so two runs can be diffed child by child. The
/// `*-cast-*` files go through the same widening `cast` the decoders use before executing.
fn dump_onpair_children(dir: &PathBuf, chunks: &[ArrayRef], ctx: &mut ExecutionCtx) -> Result<()> {
    std::fs::create_dir_all(dir)?;
    for (i, chunk) in chunks.iter().enumerate() {
        let mut children: Vec<(String, ArrayRef)> = Vec::new();
        if let Some(onpair) = chunk.as_opt::<OnPair>() {
            for (name, child, widen) in [
                ("codes", onpair.codes(), PType::U16),
                ("codes_offsets", onpair.codes_offsets(), PType::U64),
                ("dict_offsets", onpair.dict_offsets(), PType::U32),
                (
                    "uncompressed_lengths",
                    onpair.uncompressed_lengths(),
                    PType::U64,
                ),
            ] {
                children.push((name.to_string(), child.clone()));
                let dtype = DType::Primitive(widen, child.dtype().nullability());
                children.push((format!("{name}-cast-{widen}"), child.cast(dtype)?));
            }
        } else if let Some(fsst) = chunk.as_opt::<FSST>() {
            let lengths = fsst.uncompressed_lengths();
            children.push(("uncompressed_lengths".to_string(), lengths.clone()));
            let dtype = DType::Primitive(PType::U64, lengths.dtype().nullability());
            children.push((
                "uncompressed_lengths-cast-u64".to_string(),
                lengths.cast(dtype)?,
            ));
        } else {
            continue;
        }
        for (child_name, child) in children {
            let path = dir.join(format!("split-{i:02}-{child_name}.txt"));
            let text = match child.execute::<PrimitiveArray>(ctx) {
                Ok(prim) => {
                    let mut out = String::new();
                    match_each_integer_ptype!(prim.ptype(), |P| {
                        // The same widening sum the decoders use to size their output buffer.
                        let total: usize = prim
                            .as_slice::<P>()
                            .iter()
                            .map(|&v| AsPrimitive::<usize>::as_(v))
                            .sum();
                        writeln!(out, "sum={total} ptype={}", prim.ptype())?;
                        for v in prim.as_slice::<P>() {
                            writeln!(out, "{v}")?;
                        }
                    });
                    out
                }
                Err(e) => format!("ERROR: {e}\n"),
            };
            std::fs::write(&path, text)?;
        }
    }
    println!("  dumped children to {}", dir.display());
    Ok(())
}
