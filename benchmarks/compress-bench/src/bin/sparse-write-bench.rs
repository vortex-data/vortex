// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Synthetic sparse-wide file size and write-throughput baseline. No external data is required.

use std::fmt;
use std::hint::black_box;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use clap::Parser;
use clap::ValueEnum;
use compress_bench::vortex::VortexCompressor;
use futures::TryStreamExt;
use mimalloc::MiMalloc;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::aggregate_fn::fns::all_non_distinct::all_non_distinct;
use vortex::array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::ListArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::validity::Validity;
use vortex::buffer::BitBuffer;
use vortex::buffer::Buffer;
use vortex::dtype::Nullability;
use vortex::file::OpenOptionsSessionExt;
use vortex::mask::Mask;
use vortex_bench::Format;
use vortex_bench::SESSION;
use vortex_bench::compress::CompressOp;
use vortex_bench::compress::Compressed;
use vortex_bench::compress::CompressedData;
use vortex_bench::compress::Compressor;
use vortex_bench::compress::Uncompressed;
use vortex_bench::measurements::CompressionTimingMeasurement;
use vortex_bench::v3;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Case {
    FlatRuns,
    NestedRuns,
    ListsRuns,
    ListsShuffled,
    NestedChildNulls,
}

impl Case {
    const ALL: [Self; 5] = [
        Self::FlatRuns,
        Self::NestedRuns,
        Self::ListsRuns,
        Self::ListsShuffled,
        Self::NestedChildNulls,
    ];
}

impl fmt::Display for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::FlatRuns => "flat-runs",
            Self::NestedRuns => "nested-runs",
            Self::ListsRuns => "lists-runs",
            Self::ListsShuffled => "lists-shuffled",
            Self::NestedChildNulls => "nested-child-nulls",
        })
    }
}

#[derive(Parser)]
#[command(about = "Measure sparse-wide Vortex file sizes and write times without downloading data")]
struct Args {
    #[arg(long, default_value_t = 100_000)]
    rows: usize,
    #[arg(long, default_value_t = 100)]
    groups: usize,
    #[arg(long, default_value_t = 3)]
    fields: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, value_enum, value_delimiter = ',', default_values_t = Case::ALL)]
    cases: Vec<Case>,
    #[arg(long, default_value_t = 3)]
    iterations: usize,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Existing benchmark ingestion JSONL format: compression_size and compression_time records.
    #[arg(long)]
    ingest_output: Option<PathBuf>,
}

fn main() -> Result<()> {
    vortex_bench::configure_allocator();
    let args = Args::parse();
    ensure!(
        args.rows > 0 && args.rows <= u32::MAX as usize,
        "rows must fit in a positive u32"
    );
    ensure!(
        args.groups > 0 && args.groups <= args.rows,
        "groups must be between 1 and rows"
    );
    ensure!(
        args.rows.is_multiple_of(args.groups),
        "rows must be divisible by groups"
    );
    ensure!(
        args.fields > 0 && args.iterations > 0 && args.threads > 0,
        "fields, iterations, and threads must be positive"
    );
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(args.threads)
        .enable_all()
        .build()?;
    runtime.block_on(run(&args))
}

async fn run(args: &Args) -> Result<()> {
    let mut records = Vec::new();
    println!("case,file_bytes,median_write_ms,rows_per_second");
    for &case in &args.cases {
        let array = make_fixture(args, case)?;
        let dataset = format!(
            "sparse-write/{case}/rows-{}-groups-{}-fields-{}-seed-{}-threads-{}",
            args.rows, args.groups, args.fields, args.seed, args.threads
        );
        let (file_bytes, runs, uncompressed_bytes) = measure(&array, args.iterations).await?;
        let median = median(&runs);
        println!(
            "{case},{file_bytes},{:.3},{:.0}",
            median.as_secs_f64() * 1000.0,
            args.rows as f64 / median.as_secs_f64()
        );
        let timing = CompressionTimingMeasurement {
            name: format!("compress time/{dataset}"),
            time: runs.iter().copied().min().context("no timed iterations")?,
            format: Format::OnDiskVortex,
        };
        records.push(v3::compression_size_record(
            &dataset,
            None,
            Format::OnDiskVortex,
            file_bytes,
            uncompressed_bytes,
        ));
        records.push(v3::compression_time_record(
            &timing,
            &dataset,
            None,
            CompressOp::Compress,
            runs.iter().map(|run| run.as_nanos() as u64).collect(),
        ));
    }
    if let Some(path) = &args.ingest_output {
        v3::write_jsonl_to_path(path, &records)?;
    }
    Ok(())
}

fn make_fixture(args: &Args, case: Case) -> Result<ArrayRef> {
    let mut ctx = SESSION.create_execution_ctx();
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut group_ids: Vec<usize> = (0..args.rows)
        .map(|row| row / (args.rows / args.groups))
        .collect();
    if matches!(case, Case::ListsShuffled) {
        group_ids.shuffle(&mut rng);
    }
    // Values use a separate RNG so shuffling changes only row positions, not the group payloads.
    let mut values_rng = StdRng::seed_from_u64(args.seed);
    let mut wide_fields = Vec::new();
    for group in 0..args.groups {
        let positions: Vec<usize> = group_ids
            .iter()
            .enumerate()
            .filter_map(|(row, &id)| (id == group).then_some(row))
            .collect();
        let present =
            Mask::from_buffer(BitBuffer::from_iter(group_ids.iter().map(|&id| id == group)));
        let group_validity = Validity::from_mask(present.clone(), Nullability::Nullable);
        let mut fields = Vec::new();
        for field in 0..args.fields {
            let mut valid = vec![false; args.rows];
            let mut values = vec![0i64; args.rows];
            for (index, &row) in positions.iter().enumerate() {
                valid[row] = !matches!(case, Case::NestedChildNulls) || (index + field) % 10 != 0;
                if valid[row] {
                    values[row] = values_rng.random_range(0..1_000_000);
                }
            }
            let validity = Validity::from(BitBuffer::from_iter(valid.iter().copied()));
            let array = if matches!(case, Case::ListsRuns | Case::ListsShuffled) {
                let mut offsets = Vec::with_capacity(args.rows + 1);
                let mut elements = Vec::with_capacity(positions.len() * 3);
                offsets.push(0u32);
                for &is_valid in &valid {
                    if is_valid {
                        elements.extend((0..3).map(|_| values_rng.random_range(0..1_000_000i64)));
                    }
                    offsets.push(u32::try_from(elements.len())?);
                }
                ListArray::try_new(
                    PrimitiveArray::new(Buffer::from(elements), Validity::AllValid).into_array(),
                    PrimitiveArray::new(Buffer::from(offsets), Validity::NonNullable).into_array(),
                    validity,
                    &mut ctx,
                )?
                .into_array()
            } else {
                PrimitiveArray::new(Buffer::from(values), validity).into_array()
            };
            fields.push((format!("g{group:03}_f{field}"), array));
        }
        if !matches!(case, Case::FlatRuns) {
            for depth in 0..4 {
                let names = fields.iter().map(|(name, _)| name.as_str()).collect();
                let child = StructArray::try_new(
                    names,
                    fields.into_iter().map(|(_, array)| array),
                    args.rows,
                    group_validity.clone(),
                )?
                .into_array();
                let name = if depth == 3 {
                    format!("g{group:03}")
                } else {
                    format!("level{depth}")
                };
                fields = vec![(name, child)];
            }
        }
        wide_fields.extend(fields);
    }
    struct_from_fields(&wide_fields)
}

fn struct_from_fields(fields: &[(String, ArrayRef)]) -> Result<ArrayRef> {
    Ok(StructArray::from_fields(
        &fields
            .iter()
            .map(|(name, array)| (name.as_str(), array.clone()))
            .collect::<Vec<_>>(),
    )?
    .into_array())
}

async fn measure(array: &ArrayRef, iterations: usize) -> Result<(u64, Vec<Duration>, u64)> {
    let input = Uncompressed::Vortex(array.clone());
    let uncompressed_bytes =
        uncompressed_size_in_bytes(array, &mut SESSION.create_execution_ctx())? as u64;
    let mut runs = Vec::new();
    let mut file_bytes = 0;
    // One warmup includes round-trip verification, outside all timed iterations.
    for iteration in 0..=iterations {
        input.reset();
        let start = Instant::now();
        let output = VortexCompressor.compress(&input).await?;
        let elapsed = start.elapsed();
        if iteration == 0 {
            file_bytes = output.size;
            verify(&output, array).await?;
        } else {
            ensure!(file_bytes == output.size, "serialized size changed between iterations");
            runs.push(elapsed);
        }
        black_box(output);
    }
    Ok((file_bytes, runs, uncompressed_bytes))
}

async fn verify(output: &Compressed, expected: &ArrayRef) -> Result<()> {
    let CompressedData::Bytes(bytes) = &output.data else {
        anyhow::bail!("expected in-memory Vortex bytes");
    };
    let file = SESSION.open_options().open_buffer(bytes.clone())?;
    let chunks: Vec<ArrayRef> = file.scan()?.into_array_stream()?.try_collect().await?;
    let actual = ChunkedArray::try_new(chunks, expected.dtype().clone())?.into_array();
    ensure!(
        actual.len() == expected.len() && actual.dtype() == expected.dtype(),
        "round-trip length or dtype mismatch"
    );
    ensure!(
        all_non_distinct(&actual, expected, &mut SESSION.create_execution_ctx())?,
        "round-trip values or validity mismatch"
    );
    Ok(())
}

fn median(runs: &[Duration]) -> Duration {
    let mut sorted = runs.to_vec();
    sorted.sort_unstable();
    sorted[sorted.len() / 2]
}
