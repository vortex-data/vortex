// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! TPC-H string-only data generator + Vortex roundtrip harness.
//!
//! Adapts the `tpchgen-arrow` row iterators to write Parquet files
//! containing only the `Utf8`/`Utf8View` columns of each TPC-H table at the
//! requested scale factor(s).
//!
//! With `--roundtrip` the str-only parquet is written out as a Vortex file,
//! read back through the parquet arrow schema, and compared row-for-row in
//! bounded-memory chunks (works for SF=10 lineitem at 60M rows).
//!
//! With `--fsst-only` the same comparison is repeated using a Vortex file
//! written with only the FSSTScheme (no cascading on the offsets / lengths
//! children), so the on-disk sizes can be compared against the default
//! BtrBlocks cascade as a pure-FSST baseline.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use anyhow::anyhow;
use arrow_array::Array;
use arrow_array::ArrayRef;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use arrow_schema::SchemaRef;
use clap::Parser;
use futures::StreamExt;
use parquet::arrow::AsyncArrowWriter;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use tokio::fs::File as TokioFile;
use tpchgen::generators::CustomerGenerator;
use tpchgen::generators::LineItemGenerator;
use tpchgen::generators::NationGenerator;
use tpchgen::generators::OrderGenerator;
use tpchgen::generators::PartGenerator;
use tpchgen::generators::PartSuppGenerator;
use tpchgen::generators::RegionGenerator;
use tpchgen::generators::SupplierGenerator;
use tpchgen_arrow::CustomerArrow;
use tpchgen_arrow::LineItemArrow;
use tpchgen_arrow::NationArrow;
use tpchgen_arrow::OrderArrow;
use tpchgen_arrow::PartArrow;
use tpchgen_arrow::PartSuppArrow;
use tpchgen_arrow::RecordBatchIterator;
use tpchgen_arrow::RegionArrow;
use tpchgen_arrow::SupplierArrow;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::dtype::DType;
use vortex::dtype::arrow::FromArrowType;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex_btrblocks::schemes::string::FSSTScheme;
use vortex_bench::SESSION;
use vortex_bench::conversions::parquet_to_vortex_stream;

#[derive(Parser, Debug)]
struct Args {
    /// Scale factors to generate (e.g. --sf 1.0 --sf 10.0).
    #[arg(long, value_delimiter = ',', default_values_t = vec!["1.0".to_string()])]
    sf: Vec<String>,

    /// Output directory.
    #[arg(long, default_value = "/home/user/vortex/clickbench-test/tpch_str")]
    out: PathBuf,

    /// Batch size for the row generator.
    #[arg(long, default_value_t = 8 * 1024)]
    batch_size: usize,

    /// Number of partitions to split the row range across. Higher values give
    /// finer-grained parallelism and smaller files.
    #[arg(long, default_value_t = 1)]
    parts: i32,

    /// Run a Vortex roundtrip on each generated str-only parquet and verify
    /// row-for-row equality.
    #[arg(long)]
    roundtrip: bool,

    /// Write the Vortex file with only the FSSTScheme (no cascading). This
    /// gives a "pure FSST" baseline to compare against the default
    /// BtrBlocks cascade.
    #[arg(long)]
    fsst_only: bool,

    /// Comma-separated tables to generate (default: all eight).
    #[arg(long, value_delimiter = ',')]
    tables: Vec<String>,
}

fn human(b: u64) -> String {
    let kb = 1024f64;
    let mb = kb * 1024.0;
    let gb = mb * 1024.0;
    let bf = b as f64;
    if bf >= gb {
        format!("{:.2} GiB", bf / gb)
    } else if bf >= mb {
        format!("{:.2} MiB", bf / mb)
    } else if bf >= kb {
        format!("{:.2} KiB", bf / kb)
    } else {
        format!("{b} B")
    }
}

fn is_string_dtype(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

/// Build the string-only schema and column index list for the given full schema.
fn str_only_projection(full: &SchemaRef) -> (Vec<usize>, SchemaRef) {
    let mut idxs = Vec::new();
    let mut fields = Vec::new();
    for (i, f) in full.fields().iter().enumerate() {
        if is_string_dtype(f.data_type()) {
            idxs.push(i);
            fields.push(Field::clone(f));
        }
    }
    (idxs, Arc::new(Schema::new(fields)))
}

async fn write_table(
    iter_box: &mut dyn RecordBatchIterator,
    schema: &SchemaRef,
    out_path: &Path,
) -> Result<(u64, u64)> {
    let (proj, str_schema) = str_only_projection(schema);
    if proj.is_empty() {
        return Ok((0, 0));
    }

    let file = TokioFile::create(out_path).await?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut writer = AsyncArrowWriter::try_new(file, str_schema.clone(), Some(props))?;

    let mut rows: u64 = 0;
    while let Some(batch) = iter_box.next() {
        let projected = batch.project(&proj).map_err(|e| anyhow!(e))?;
        rows += projected.num_rows() as u64;
        writer.write(&projected).await?;
    }
    writer.close().await?;
    let size = tokio::fs::metadata(out_path).await?.len();
    Ok((rows, size))
}

fn build_iter(
    table: &str,
    sf: f64,
    part: i32,
    parts: i32,
    batch_size: usize,
) -> Result<Box<dyn RecordBatchIterator>> {
    Ok(match table {
        "nation" => Box::new(
            NationArrow::new(NationGenerator::new(sf, part, parts)).with_batch_size(batch_size),
        ),
        "region" => Box::new(
            RegionArrow::new(RegionGenerator::new(sf, part, parts)).with_batch_size(batch_size),
        ),
        "supplier" => Box::new(
            SupplierArrow::new(SupplierGenerator::new(sf, part, parts)).with_batch_size(batch_size),
        ),
        "customer" => Box::new(
            CustomerArrow::new(CustomerGenerator::new(sf, part, parts)).with_batch_size(batch_size),
        ),
        "part" => Box::new(
            PartArrow::new(PartGenerator::new(sf, part, parts)).with_batch_size(batch_size),
        ),
        "partsupp" => Box::new(
            PartSuppArrow::new(PartSuppGenerator::new(sf, part, parts)).with_batch_size(batch_size),
        ),
        "orders" => Box::new(
            OrderArrow::new(OrderGenerator::new(sf, part, parts)).with_batch_size(batch_size),
        ),
        "lineitem" => Box::new(
            LineItemArrow::new(LineItemGenerator::new(sf, part, parts)).with_batch_size(batch_size),
        ),
        other => anyhow::bail!("unknown table {other}"),
    })
}

async fn write_vortex(parquet: &Path, out: &Path, fsst_only: bool) -> Result<u64> {
    let f = TokioFile::open(parquet).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(f).await?;
    let dtype = DType::from_arrow(builder.schema().as_ref());
    let stream = parquet_to_vortex_stream(builder.build()?);

    let mut out_f = tokio::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(out)
        .await?;

    let strategy = if fsst_only {
        // Only the FSST scheme - no further cascading on offsets/lengths.
        // Children fall back to canonical encoding.
        let builder = BtrBlocksCompressorBuilder::empty().with_new_scheme(&FSSTScheme);
        WriteStrategyBuilder::default()
            .with_btrblocks_builder(builder)
            .build()
    } else {
        WriteStrategyBuilder::default().build()
    };

    SESSION
        .write_options()
        .with_strategy(strategy)
        .write(&mut out_f, ArrayStreamAdapter::new(dtype, stream))
        .await?;
    drop(out_f);
    Ok(tokio::fs::metadata(out).await?.len())
}

fn first_string_mismatch(
    e: &ArrayRef,
    a: &ArrayRef,
) -> Result<Option<(usize, String, String)>> {
    if e.len() != a.len() {
        anyhow::bail!("length mismatch: {} vs {}", e.len(), a.len());
    }
    if e.data_type() != a.data_type() {
        anyhow::bail!("dtype mismatch: {:?} vs {:?}", e.data_type(), a.data_type());
    }
    let len = e.len();
    match e.data_type() {
        DataType::Utf8 => {
            let e = e.as_string::<i32>();
            let a = a.as_string::<i32>();
            for i in 0..len {
                if e.is_null(i) != a.is_null(i) {
                    return Ok(Some((
                        i,
                        format!("null={}", e.is_null(i)),
                        format!("null={}", a.is_null(i)),
                    )));
                }
                if !e.is_null(i) && e.value(i) != a.value(i) {
                    return Ok(Some((
                        i,
                        format!("{:?}", e.value(i)),
                        format!("{:?}", a.value(i)),
                    )));
                }
            }
            Ok(None)
        }
        DataType::LargeUtf8 => {
            let e = e.as_string::<i64>();
            let a = a.as_string::<i64>();
            for i in 0..len {
                if e.is_null(i) != a.is_null(i) {
                    return Ok(Some((
                        i,
                        format!("null={}", e.is_null(i)),
                        format!("null={}", a.is_null(i)),
                    )));
                }
                if !e.is_null(i) && e.value(i) != a.value(i) {
                    return Ok(Some((
                        i,
                        format!("{:?}", e.value(i)),
                        format!("{:?}", a.value(i)),
                    )));
                }
            }
            Ok(None)
        }
        DataType::Utf8View => {
            let e = e.as_string_view();
            let a = a.as_string_view();
            for i in 0..len {
                if e.is_null(i) != a.is_null(i) {
                    return Ok(Some((
                        i,
                        format!("null={}", e.is_null(i)),
                        format!("null={}", a.is_null(i)),
                    )));
                }
                if !e.is_null(i) && e.value(i) != a.value(i) {
                    return Ok(Some((
                        i,
                        format!("{:?}", e.value(i)),
                        format!("{:?}", a.value(i)),
                    )));
                }
            }
            Ok(None)
        }
        other => anyhow::bail!("unsupported column dtype for str-only roundtrip: {other:?}"),
    }
}

/// Stream both sides one row-chunk at a time and compare. This keeps memory
/// bounded for large inputs.
async fn roundtrip_str_only(parquet: &Path, fsst_only: bool) -> Result<()> {
    let vortex_path = parquet.with_extension(if fsst_only { "fsst.vortex" } else { "vortex" });
    let parquet_size = tokio::fs::metadata(parquet).await?.len();
    let t = Instant::now();
    let vortex_size = write_vortex(parquet, &vortex_path, fsst_only).await?;
    eprintln!(
        "    write vortex: {} -> {} ({:.2}%) in {:?}",
        human(parquet_size),
        human(vortex_size),
        100.0 * vortex_size as f64 / parquet_size as f64,
        t.elapsed()
    );

    // Open both sides as streams.
    let pq_file = TokioFile::open(parquet).await?;
    let pq_builder = ParquetRecordBatchStreamBuilder::new(pq_file).await?;
    let pq_schema = pq_builder.schema().clone();
    let mut pq_stream = Box::pin(pq_builder.build()?);

    let vortex_file = SESSION.open_options().open_path(&vortex_path).await?;
    let vortex_scan = vortex_file.scan()?;
    let vortex_stream = vortex_scan.into_record_batch_stream(Arc::clone(&pq_schema))?;
    let mut vortex_stream = Box::pin(vortex_stream);

    // Maintain a queue of remaining rows for each side. As batches arrive we
    // align them into matching row windows by trimming whichever side has
    // more rows and comparing the overlapping prefix.
    let mut pq_buf: Option<RecordBatch> = pq_stream.next().await.transpose()?;
    let mut vx_buf: Option<RecordBatch> = vortex_stream.next().await.transpose()?;
    let mut total_rows: usize = 0;

    while pq_buf.is_some() && vx_buf.is_some() {
        let p = pq_buf.as_ref().unwrap();
        let v = vx_buf.as_ref().unwrap();
        let n = p.num_rows().min(v.num_rows());

        // Slice both sides to n rows and compare column-wise.
        for col in 0..pq_schema.fields().len() {
            let pc: ArrayRef = p.column(col).slice(0, n);
            let vc: ArrayRef = v.column(col).slice(0, n);
            if let Some((idx, exp, got)) = first_string_mismatch(&pc, &vc)? {
                anyhow::bail!(
                    "MISMATCH in {} column {col} {} row {idx} (window starting at row {}): expected={exp} got={got}",
                    parquet.display(),
                    pq_schema.field(col).name(),
                    total_rows,
                );
            }
        }
        total_rows += n;

        // Advance each side: if a side had more than n rows, slice off the head;
        // otherwise pull the next batch.
        let p_rem = p.num_rows() - n;
        let v_rem = v.num_rows() - n;

        pq_buf = if p_rem == 0 {
            pq_stream.next().await.transpose()?
        } else {
            Some(p.slice(n, p_rem))
        };
        vx_buf = if v_rem == 0 {
            vortex_stream.next().await.transpose()?
        } else {
            Some(v.slice(n, v_rem))
        };
    }

    // Both streams must be exhausted at the same time.
    if pq_buf.is_some() || vx_buf.is_some() {
        anyhow::bail!(
            "row count mismatch after {} rows: parquet remaining = {:?}, vortex remaining = {:?}",
            total_rows,
            pq_buf.as_ref().map(|b| b.num_rows()),
            vx_buf.as_ref().map(|b| b.num_rows()),
        );
    }

    eprintln!(
        "    roundtrip OK: {total_rows} rows across {} columns",
        pq_schema.fields().len()
    );
    Ok(())
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args = Args::parse();
    tokio::fs::create_dir_all(&args.out).await?;

    let default_tables = [
        "nation", "region", "supplier", "part", "customer", "partsupp", "orders", "lineitem",
    ];
    let tables: Vec<&str> = if args.tables.is_empty() {
        default_tables.iter().copied().collect()
    } else {
        args.tables.iter().map(String::as_str).collect()
    };

    for sf_str in &args.sf {
        let sf: f64 = sf_str.parse()?;
        let sf_dir = args.out.join(format!("sf-{sf_str}"));
        tokio::fs::create_dir_all(&sf_dir).await?;
        eprintln!("\n###  TPC-H SF={sf}  out={}", sf_dir.display());

        for table in &tables {
            for part_idx in 1..=args.parts {
                let mut iter = build_iter(table, sf, part_idx, args.parts, args.batch_size)?;
                let schema = iter.schema().clone();
                let (proj, str_schema) = str_only_projection(&schema);
                if proj.is_empty() {
                    eprintln!("  {table}: no string columns, skipping");
                    continue;
                }

                let file_name = if args.parts == 1 {
                    format!("{table}_str.parquet")
                } else {
                    format!("{table}_str_{part_idx}.parquet")
                };
                let out = sf_dir.join(&file_name);

                let t = Instant::now();
                let (rows, size) = write_table(iter.as_mut(), &schema, &out).await?;
                eprintln!(
                    "  {table}: {rows} rows, {} cols ({}), wrote {} in {:?}",
                    str_schema.fields().len(),
                    str_schema
                        .fields()
                        .iter()
                        .map(|f| f.name().as_str())
                        .collect::<Vec<_>>()
                        .join(","),
                    human(size),
                    t.elapsed()
                );

                if args.roundtrip {
                    roundtrip_str_only(&out, false).await?;
                }
                if args.fsst_only {
                    roundtrip_str_only(&out, true).await?;
                }
            }
        }
    }

    Ok(())
}
