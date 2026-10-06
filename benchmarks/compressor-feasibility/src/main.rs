// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Feasibility experiments for learned compression scheme selection, integers only.
//!
//! For each integer chunk this compresses with the production compressor and with variants:
//! each scheme forced at the root (children chosen as in production), sampling replaced by
//! closed-form estimates, and heuristic caps removed. It records bytes, compression time,
//! decode time and per-chunk features as CSV for offline analysis.

mod features;
mod model;
mod wrap;

use std::collections::BTreeMap;
use std::fs;
use std::fs::File;
use std::hint::black_box;
use std::io::BufWriter;
use std::io::Write;
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Context;
use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_array::RecordBatch;
use arrow_schema::DataType;
use clap::Parser;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use tpchgen::generators::CustomerGenerator;
use tpchgen::generators::LineItemGenerator;
use tpchgen::generators::OrderGenerator;
use tpchgen::generators::PartSuppGenerator;
use vortex::VortexSessionDefault;
use vortex::session::VortexSession;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::serde::SerializeOptions;
use vortex_array::validity::Validity;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::CompressionSessionExt;
use vortex_btrblocks::SchemeExt;
use vortex_compressor::scheme::Scheme;

use crate::wrap::Mode;
use crate::wrap::Wrapped;

#[derive(Parser, Debug)]
struct Args {
    /// Directory of Parquet files; every integer column is used.
    #[arg(long)]
    data_dir: Option<PathBuf>,
    /// TPC-H scale factor to generate in-process; 0 disables TPC-H.
    #[arg(long, default_value_t = 1.0)]
    tpch_sf: f64,
    /// Rows per chunk.
    #[arg(long, default_value_t = 65_536)]
    chunk_rows: usize,
    /// At most this many chunks per column, spread evenly through the column.
    #[arg(long, default_value_t = 8)]
    max_chunks_per_column: usize,
    /// Repetitions for decode timing.
    #[arg(long, default_value_t = 5)]
    decode_reps: usize,
    /// Repetitions for compression timing of the default and no-sample variants.
    #[arg(long, default_value_t = 3)]
    compress_reps: usize,
    /// Only run variants whose name starts with one of these prefixes.
    #[arg(long, value_delimiter = ',')]
    variants: Vec<String>,
    /// A label written into every row, e.g. a patched build such as `max_cascade=4`.
    #[arg(long, default_value = "stock")]
    tag: String,
    /// Directory of models from `train.py --export`: `<source>.json` (trained without that
    /// source) or `all.json`. Enables the model-driven compressor variants.
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// Bandwidths (bytes/s) the model-driven compressor optimises for, as `label=value`.
    #[arg(long, value_delimiter = ',', default_value = "s3=1e8,nvme=2e9,mem=2e10")]
    bandwidths: Vec<String>,
    /// Reads per write the model-driven compressor optimises for: compression time is divided
    /// by this, serving cost (transfer plus decode) is paid on every read.
    #[arg(long, value_delimiter = ',', default_value = "1,10,100")]
    reads: Vec<f64>,
    /// Candidates the model-driven compressor may not use, e.g. `pco`.
    #[arg(long, value_delimiter = ',')]
    exclude_candidates: Vec<String>,
    /// Minimum predicted saving before the model's proposal is compressed and verified.
    #[arg(long, default_value_t = 0.1)]
    gate: f64,
    /// Output directory.
    #[arg(long)]
    out: PathBuf,
}

/// One integer column chunk.
struct Chunk {
    source: String,
    column: String,
    index: usize,
    arrow: ArrowArrayRef,
}

struct Variant {
    name: String,
    compressor: BtrBlocksCompressor,
    timed: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.out)?;
    std::panic::set_hook(Box::new(|_| {}));

    let session = VortexSession::default();
    if wrap::SESSION.set(session.clone()).is_err() {
        anyhow::bail!("session already set");
    }
    let mut ctx = session.create_execution_ctx();
    let all_variants = build_variants(&session)?;
    if let Some(production) = all_variants.iter().find(|v| v.name == "default")
        && wrap::SPY_COMPRESSOR.set((*production.compressor).clone()).is_err()
    {
        anyhow::bail!("spy compressor already set");
    }
    let mut estimates = BufWriter::new(File::create(
        args.out.join(format!("estimates-{}.csv", args.tag)),
    )?);
    writeln!(estimates, "source,column,chunk,scheme,kind,ratio")?;
    let variants: Vec<&Variant> = all_variants
        .iter()
        .filter(|v| {
            args.variants.is_empty() || args.variants.iter().any(|f| v.name.starts_with(f.as_str()))
        })
        .collect();
    // The model's candidates map onto compressors we already build.
    let candidates: BTreeMap<String, &BtrBlocksCompressor> = all_variants
        .iter()
        .filter_map(|v| {
            let name = match v.name.as_str() {
                "default" => "production".to_string(),
                "model/runend+sparse" => "sizemodel".to_string(),
                other => other.strip_prefix("forced/")?.to_string(),
            };
            (!args.exclude_candidates.contains(&name)).then_some((name, &v.compressor))
        })
        .collect();
    let bandwidths: Vec<(String, f64)> = args
        .bandwidths
        .iter()
        .map(|b| {
            let (label, value) = b.split_once('=').context("bandwidths look like label=bytes_per_sec")?;
            Ok((label.to_string(), value.parse::<f64>()?))
        })
        .collect::<anyhow::Result<_>>()?;
    let mut models: BTreeMap<String, Option<model::Model>> = BTreeMap::new();
    eprintln!(
        "variants: {}",
        variants
            .iter()
            .map(|v| v.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let chunks = load_chunks(&args)?;
    eprintln!("{} chunks", chunks.len());

    let mut rows = BufWriter::new(File::create(
        args.out.join(format!("rows-{}.csv", args.tag)),
    )?);
    writeln!(
        rows,
        "tag,source,column,chunk,ptype,len,variant,ok,canonical_bytes,bytes,nbytes,root,tree,compress_ns,decode_ns_median,decode_ns_min,error"
    )?;
    let mut feats = BufWriter::new(File::create(
        args.out.join(format!("features-{}.csv", args.tag)),
    )?);
    writeln!(feats, "source,column,chunk,{}", features::header())?;

    let started = Instant::now();
    for (i, chunk) in chunks.iter().enumerate() {
        let canonical = to_vortex(&chunk.arrow, &mut ctx)?;
        let ptype = canonical.ptype();
        let canonical_bytes = serialized_size(&canonical.clone().into_array(), &session)?;
        let input = canonical.into_array();

        writeln!(
            feats,
            "{},{},{},{}",
            chunk.source,
            chunk.column,
            chunk.index,
            features::compute(&chunk.arrow)?
        )?;

        if let Some(dir) = &args.model_dir {
            if !models.contains_key(&chunk.source) {
                let own = dir.join(format!("{}.json", chunk.source));
                let path = if own.exists() { own } else { dir.join("all.json") };
                let loaded = path.exists().then(|| model::Model::load(&path)).transpose()?;
                models.insert(chunk.source.clone(), loaded);
            }
            if let Some(Some(m)) = models.get(&chunk.source) {
                for ((label, bandwidth), reads) in bandwidths
                    .iter()
                    .flat_map(|b| args.reads.iter().map(move |r| (b, *r)))
                {
                    let mut times = Vec::with_capacity(args.compress_reps);
                    let mut outcome = None;
                    for _ in 0..args.compress_reps.max(1) {
                        let start = Instant::now();
                        let result =
                            model::compress(
                            m,
                            &candidates,
                            *bandwidth,
                            reads,
                            args.gate,
                            &input,
                            &session,
                            &mut ctx,
                        )?;
                        times.push(start.elapsed().as_nanos());
                        outcome = Some(result);
                    }
                    times.sort_unstable();
                    let outcome = outcome.context("no repetitions")?;
                    let compressed = outcome.array;
                    let (median, min) = decode_timed(&compressed, args.decode_reps, &mut ctx)?;
                    writeln!(
                        rows,
                        "{},{},{},{},{},{},model@{label}@r{reads},1,{canonical_bytes},{},{},{},proposed={} tried={} kept={},{},{median},{min},",
                        args.tag,
                        chunk.source,
                        chunk.column,
                        chunk.index,
                        ptype,
                        input.len(),
                        serialized_size(&compressed, &session)?,
                        compressed.nbytes(),
                        compressed.encoding_id(),
                        outcome.proposed,
                        u8::from(outcome.tried),
                        outcome.kept,
                        times[times.len() / 2],
                    )?;
                }
            }
        }

        for variant in &variants {
            wrap::ESTIMATES.lock().clear();
            let reps = if variant.timed { args.compress_reps } else { 1 };
            let result = compress_timed(&variant.compressor, &input, reps, &mut ctx);
            let prefix = format!(
                "{},{},{},{},{},{},{}",
                args.tag,
                chunk.source,
                chunk.column,
                chunk.index,
                ptype,
                input.len(),
                variant.name
            );
            match result {
                Ok((compressed, compress_ns)) => {
                    let (median, min) = decode_timed(&compressed, args.decode_reps, &mut ctx)?;
                    let tree = compressed
                        .display_tree_encodings_only()
                        .to_string()
                        .lines()
                        .map(str::trim)
                        .collect::<Vec<_>>()
                        .join(" / ")
                        .replace(',', ";");
                    for (scheme, kind, ratio) in wrap::ESTIMATES.lock().drain(..) {
                        writeln!(
                            estimates,
                            "{},{},{},{},{kind},{ratio}",
                            chunk.source,
                            chunk.column,
                            chunk.index,
                            scheme.trim_start_matches("vortex.int."),
                        )?;
                    }
                    writeln!(
                        rows,
                        "{prefix},1,{canonical_bytes},{},{},{},{tree},{compress_ns},{median},{min},",
                        serialized_size(&compressed, &session)?,
                        compressed.nbytes(),
                        compressed.encoding_id(),
                    )?;
                }
                Err(error) => {
                    let error = error
                        .lines()
                        .next()
                        .unwrap_or_default()
                        .replace(',', ";");
                    writeln!(rows, "{prefix},0,{canonical_bytes},,,,,,,,{error}")?;
                }
            }
        }

        if (i + 1) % 50 == 0 {
            eprintln!(
                "{}/{} chunks, {:.0}s",
                i + 1,
                chunks.len(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    rows.flush()?;
    feats.flush()?;
    estimates.flush()?;
    let profile: Vec<u64> = model::PROFILE
        .iter()
        .map(|p| p.load(std::sync::atomic::Ordering::Relaxed) / 1_000_000)
        .collect();
    eprintln!(
        "model profile (ms): features {}, canonical size {}, inference {}, production compress {}",
        profile[0], profile[1], profile[2], profile[3]
    );
    eprintln!("done in {:.0}s", started.elapsed().as_secs_f64());
    Ok(())
}

fn build_variants(session: &VortexSession) -> anyhow::Result<Vec<Variant>> {
    let probe = Canonical::Primitive(PrimitiveArray::new(
        vortex::buffer::buffer![1i64, 2, 3],
        Validity::NonNullable,
    ));
    let int_schemes: Vec<&'static dyn Scheme> = session
        .compression()
        .schemes()
        .iter()
        .copied()
        .filter(|s| s.matches(&probe))
        .collect();
    let int_ids: Vec<_> = int_schemes.iter().map(|s| s.id()).collect();

    // Replace every integer scheme, keeping registration order, wrapping the ones `wrap` picks.
    let replaced = |wrap: &dyn Fn(&'static dyn Scheme) -> Option<Mode>| {
        let mut builder =
            BtrBlocksCompressorBuilder::from_session(session).exclude_schemes(int_ids.clone());
        for scheme in &int_schemes {
            builder = builder.with_new_scheme(match wrap(*scheme) {
                Some(mode) => Wrapped::leak(*scheme, mode),
                None => *scheme,
            });
        }
        builder.build()
    };
    let named = |name: &str, s: &'static dyn Scheme| s.scheme_name().contains(name);

    let mut variants = vec![
        Variant {
            name: "default".to_string(),
            compressor: BtrBlocksCompressorBuilder::from_session(session).build(),
            timed: true,
        },
        Variant {
            name: "nosample".to_string(),
            compressor: replaced(&|_| Some(Mode::NoSample)),
            timed: true,
        },
    ];
    for cap in ["dict", "sparse", "runend", "rle"] {
        variants.push(Variant {
            name: format!("capoff/{cap}"),
            compressor: replaced(&|s| named(cap, s).then_some(Mode::CapOff)),
            timed: false,
        });
    }
    variants.push(Variant {
        name: "capoff/all".to_string(),
        compressor: replaced(&|s| {
            ["dict", "sparse", "runend", "rle"]
                .iter()
                .any(|cap| named(cap, s))
                .then_some(Mode::CapOff)
        }),
        timed: false,
    });
    variants.push(Variant {
        name: "spy".to_string(),
        compressor: replaced(&|_| Some(Mode::Spy)),
        timed: false,
    });
    variants.push(Variant {
        name: "model/runend+sparse".to_string(),
        compressor: replaced(&|_| Some(Mode::SizeModel)),
        timed: true,
    });
    let wide = wrap::SamplePolicy {
        slices: 4,
        slice_len: 1024,
        serialized: true,
        zero_ok: true,
        all: false,
    };
    variants.push(Variant {
        name: "model+strat-4x1k".to_string(),
        compressor: replaced(&|s| {
            Some(if named("runend", s) || named("sparse", s) {
                Mode::SizeModel
            } else {
                Mode::Estimate(wide)
            })
        }),
        timed: true,
    });
    let estimates = [
        ("est/prod", 16, 64, false, false, false),
        ("est/zero-ok", 16, 64, false, true, false),
        ("est/serialized", 16, 64, true, true, false),
        ("est/serialized-all", 16, 64, true, true, true),
        ("est/contig-1k", 1, 1024, true, true, false),
        ("est/contig-4k", 1, 4096, true, true, false),
        ("est/strat-4x1k", 4, 1024, true, true, false),
        ("est/strat-4x1k-all", 4, 1024, true, true, true),
        ("est/strat-64x64", 64, 64, true, true, false),
    ];
    for (name, slices, slice_len, serialized, zero_ok, all) in estimates {
        let policy = wrap::SamplePolicy {
            slices,
            slice_len,
            serialized,
            zero_ok,
            all,
        };
        variants.push(Variant {
            name: name.to_string(),
            compressor: replaced(&|_| Some(Mode::Estimate(policy))),
            timed: true,
        });
    }
    for scheme in &int_schemes {
        let short = scheme.scheme_name().trim_start_matches("vortex.int.");
        variants.push(Variant {
            name: format!("forced/{short}"),
            compressor: BtrBlocksCompressorBuilder::from_session(session)
                .with_new_scheme(Wrapped::leak(*scheme, Mode::ForceRoot))
                .build(),
            timed: true,
        });
    }

    Ok(variants)
}

/// The size of the array as written: its buffers plus the flatbuffer holding its metadata.
pub(crate) fn serialized_size(array: &ArrayRef, session: &VortexSession) -> anyhow::Result<u64> {
    let buffers = array.serialize(&ArrayContext::empty(), session, &SerializeOptions::default())?;
    Ok(buffers.iter().map(|b| b.len() as u64).sum())
}

fn compress_timed(
    compressor: &BtrBlocksCompressor,
    input: &ArrayRef,
    reps: usize,
    ctx: &mut ExecutionCtx,
) -> Result<(ArrayRef, u128), String> {
    let mut times = Vec::with_capacity(reps);
    let mut last = None;
    for _ in 0..reps {
        let start = Instant::now();
        let result = std::panic::catch_unwind(AssertUnwindSafe(|| compressor.compress(input, ctx)));
        times.push(start.elapsed().as_nanos());
        match result {
            Ok(Ok(array)) => last = Some(array),
            Ok(Err(error)) => return Err(error.to_string()),
            Err(_) => return Err("panic".to_string()),
        }
    }
    times.sort_unstable();
    let compressed = last.ok_or_else(|| "no repetitions".to_string())?;
    Ok((compressed, times[times.len() / 2]))
}

fn decode_timed(
    compressed: &ArrayRef,
    reps: usize,
    ctx: &mut ExecutionCtx,
) -> anyhow::Result<(u128, u128)> {
    // One warm-up decode, then timed repetitions.
    black_box(compressed.clone().execute::<Canonical>(ctx)?);
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        black_box(compressed.clone().execute::<Canonical>(ctx)?);
        times.push(start.elapsed().as_nanos());
    }
    times.sort_unstable();
    Ok((times[times.len() / 2], times[0]))
}

#[allow(deprecated)]
fn to_vortex(array: &ArrowArrayRef, ctx: &mut ExecutionCtx) -> anyhow::Result<PrimitiveArray> {
    use vortex_arrow::FromArrowArray;
    let nullable = array.null_count() > 0;
    let vortex = ArrayRef::from_arrow(array.as_ref(), nullable)?;
    Ok(vortex.execute::<PrimitiveArray>(ctx)?)
}

fn is_int(data_type: &DataType) -> bool {
    data_type.is_integer()
}

/// Picks `max` chunk indices spread evenly over `total`.
fn spread(total: usize, max: usize) -> Vec<usize> {
    if total <= max {
        return (0..total).collect();
    }
    (0..max).map(|i| i * total / max).collect()
}

fn load_chunks(args: &Args) -> anyhow::Result<Vec<Chunk>> {
    let mut chunks = Vec::new();

    if let Some(dir) = &args.data_dir {
        let mut paths: Vec<_> = fs::read_dir(dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|ext| ext == "parquet"))
            .collect();
        paths.sort();
        for path in paths {
            let source = path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let builder = ParquetRecordBatchReaderBuilder::try_new(
                File::open(&path).with_context(|| format!("opening {}", path.display()))?,
            )?;
            let schema = builder.schema().clone();
            let int_fields: Vec<usize> = schema
                .fields()
                .iter()
                .enumerate()
                .filter(|(_, f)| is_int(f.data_type()))
                .map(|(i, _)| i)
                .collect();
            let mask = ProjectionMask::roots(builder.parquet_schema(), int_fields);
            let reader = builder
                .with_projection(mask)
                .with_batch_size(args.chunk_rows)
                .build()?;
            let batches = reader.collect::<Result<Vec<RecordBatch>, _>>()?;
            push_batches(&source, &batches, args.max_chunks_per_column, &mut chunks);
            eprintln!("loaded {source}");
        }
    }

    if args.tpch_sf > 0.0 {
        let sf = args.tpch_sf;
        let batch = args.chunk_rows;
        let tables: Vec<(&str, Vec<RecordBatch>)> = vec![
            (
                "tpch_lineitem",
                tpchgen_arrow::LineItemArrow::new(LineItemGenerator::new(sf, 1, 1))
                    .with_batch_size(batch)
                    .collect(),
            ),
            (
                "tpch_orders",
                tpchgen_arrow::OrderArrow::new(OrderGenerator::new(sf, 1, 1))
                    .with_batch_size(batch)
                    .collect(),
            ),
            (
                "tpch_partsupp",
                tpchgen_arrow::PartSuppArrow::new(PartSuppGenerator::new(sf, 1, 1))
                    .with_batch_size(batch)
                    .collect(),
            ),
            (
                "tpch_customer",
                tpchgen_arrow::CustomerArrow::new(CustomerGenerator::new(sf, 1, 1))
                    .with_batch_size(batch)
                    .collect(),
            ),
        ];
        for (source, batches) in tables {
            push_batches(source, &batches, args.max_chunks_per_column, &mut chunks);
            eprintln!("generated {source}");
        }
    }

    Ok(chunks)
}

fn push_batches(source: &str, batches: &[RecordBatch], max: usize, chunks: &mut Vec<Chunk>) {
    let Some(first) = batches.first() else {
        return;
    };
    let schema = first.schema();
    for (col, field) in schema.fields().iter().enumerate() {
        if !is_int(field.data_type()) {
            continue;
        }
        for index in spread(batches.len(), max) {
            chunks.push(Chunk {
                source: source.to_string(),
                column: field.name().clone(),
                index,
                arrow: batches[index].column(col).clone(),
            });
        }
    }
}
