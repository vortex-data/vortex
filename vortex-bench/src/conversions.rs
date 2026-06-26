// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use futures::StreamExt;
use futures::TryStreamExt;
use parquet::arrow::AsyncArrowWriter;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::async_reader::ParquetRecordBatchStream;
use parquet::file::metadata::KeyValue;
use parquet::file::metadata::ParquetMetaData;
use sysinfo::System;
use tokio::fs::File;
use tokio::fs::OpenOptions;
use tokio::fs::create_dir_all;
use tokio::io::AsyncWriteExt;
use tracing::Instrument;
use tracing::info;
use tracing::trace;
use vortex::VortexSessionDefault;
use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::ChunkedArray;
use vortex::array::arrays::ExtensionArray;
use vortex::array::arrays::Struct;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::array::arrow::FromArrowArray;
use vortex::array::builders::builder_with_capacity;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::array::stream::ArrayStreamExt;
use vortex::compressor::BtrBlocksCompressorBuilder;
use vortex::dtype::DType;
use vortex::dtype::FieldPath;
use vortex::dtype::StructFields;
use vortex::dtype::arrow::FromArrowType;
use vortex::dtype::extension::ExtDType;
use vortex::dtype::extension::ExtDTypeRef;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::file::VortexWriteOptions;
use vortex::file::WriteOptionsSessionExt;
use vortex::file::WriteStrategyBuilder;
use vortex::layout::LayoutStrategy;
use vortex::layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex::layout::layouts::compressed::CompressingStrategy;
use vortex::layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_set::HashSet;
use vortex_geo::extension::GeoMetadata;
use vortex_geo::extension::WellKnownBinary;

use crate::CompactionStrategy;
use crate::Format;
use crate::SESSION;
use crate::utils::file::idempotent_async;

/// Memory budget per concurrent conversion stream in GB. This is somewhat arbitary.
const MEMORY_PER_STREAM_GB: u64 = 4;

/// Minimum number of concurrent conversion streams.
const MIN_CONCURRENCY: u64 = 1;

/// Maximum number of concurrent conversion streams. This is somewhat arbitary.
const MAX_CONCURRENCY: u64 = 16;

/// Returns the available system memory in bytes.
fn available_memory_bytes() -> u64 {
    System::new_all().available_memory()
}

/// Calculate appropriate concurrency based on available memory.
fn calculate_concurrency() -> usize {
    let available_gb = available_memory_bytes() / (1024 * 1024 * 1024);
    let concurrency = (available_gb / MEMORY_PER_STREAM_GB).clamp(MIN_CONCURRENCY, MAX_CONCURRENCY);

    info!(
        "Available memory: {}GB, maximum concurrency is: {}",
        available_gb, concurrency
    );

    concurrency as usize
}

/// Read a Parquet file and return it as a Vortex [`ChunkedArray`].
///
/// Note: This loads the entire file into memory. For large files, use the streaming conversion like
/// in [`parquet_to_vortex_stream`] instead.
pub async fn parquet_to_vortex_chunks(parquet_path: PathBuf) -> anyhow::Result<ChunkedArray> {
    let file = File::open(parquet_path).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;
    let reader = builder.build()?;

    let chunks: Vec<ArrayRef> = parquet_to_vortex_stream(reader)
        .map(|r| r.map_err(anyhow::Error::from))
        .try_collect()
        .await?;

    Ok(ChunkedArray::from_iter(chunks))
}

/// Create a streaming Vortex array from a Parquet reader.
///
/// Streams record batches and converts them to Vortex arrays on-the-fly, avoiding loading the
/// entire file into memory.
pub fn parquet_to_vortex_stream(
    reader: ParquetRecordBatchStream<File>,
) -> impl futures::Stream<Item = VortexResult<ArrayRef>> {
    reader.map(move |result| {
        result.map_err(|e| vortex_err!(External: e)).and_then(|rb| {
            let chunk = ArrayRef::from_arrow(rb, false)?;
            let mut builder = builder_with_capacity(chunk.dtype(), chunk.len());

            // Canonicalize the chunk.
            chunk.append_to_builder(
                builder.as_mut(),
                &mut VortexSession::default().create_execution_ctx(),
            )?;

            Ok(builder.finish())
        })
    })
}

/// Convert a single Parquet file to Vortex format using streaming.
///
/// Streams data directly from Parquet to Vortex without loading the entire file into memory.
pub async fn convert_parquet_file_to_vortex(
    parquet_path: &Path,
    output_path: &Path,
    compaction: CompactionStrategy,
) -> anyhow::Result<()> {
    let file = File::open(parquet_path).await?;
    let builder = ParquetRecordBatchStreamBuilder::new(file).await?;

    // GeoParquet geometry tagging.
    let geo_columns = geoparquet_columns(builder.metadata());
    let dtype = tag_geo_dtype(DType::from_arrow(builder.schema().as_ref()), &geo_columns)?;
    let stream = parquet_to_vortex_stream(builder.build()?)
        .map(move |chunk| chunk.and_then(|chunk| tag_geo_array(chunk, &geo_columns)));

    let mut output_file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .create(true)
        .open(output_path)
        .await?;

    write_options_for(compaction, &dtype, is_spatialbench(parquet_path))
        .write(
            &mut output_file,
            ArrayStreamExt::boxed(ArrayStreamAdapter::new(dtype, stream)),
        )
        .await?;

    Ok(())
}

/// Whether `path` points at SpatialBench data.
fn is_spatialbench(path: &Path) -> bool {
    path.components()
        .any(|component| component.as_os_str() == "spatialbench")
}

/// Vortex write options for converting `dtype`-shaped data.
///
/// For SpatialBench (`skip_binary_dict`), the geometry blobs are large and
/// unique, so the dictionary builder balloons memory (tens of GB) for zero gain.
fn write_options_for(
    compaction: CompactionStrategy,
    dtype: &DType,
    skip_binary_dict: bool,
) -> VortexWriteOptions {
    let binary_fields: Vec<_> = match dtype {
        DType::Struct(fields, _) if skip_binary_dict => fields
            .names()
            .iter()
            .zip(fields.fields())
            .filter(|(_, field)| matches!(field, DType::Binary(_)))
            .map(|(name, _)| name.clone())
            .collect(),
        _ => Vec::new(),
    };
    if binary_fields.is_empty() {
        return compaction.apply_options(SESSION.write_options());
    }

    let mut builder = WriteStrategyBuilder::default();
    if matches!(compaction, CompactionStrategy::Compact) {
        builder =
            builder.with_btrblocks_builder(BtrBlocksCompressorBuilder::default().with_compact());
    }
    for name in binary_fields {
        builder = builder.with_field_writer(FieldPath::from_name(name), no_dict_layout());
    }
    SESSION.write_options().with_strategy(builder.build())
}

/// A chunked + compressed layout that skips dictionary encoding for opaque `Binary` blobs.
fn no_dict_layout() -> Arc<dyn LayoutStrategy> {
    Arc::new(CompressingStrategy::new(
        ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
        BtrBlocksCompressorBuilder::default().build(),
    ))
}

/// Convert all Parquet files in a directory to Vortex format.
///
/// This function reads Parquet files from `{input_path}/parquet/` and writes Vortex files to
/// `{input_path}/vortex-file-compressed/` (for Default compaction) or
/// `{input_path}/vortex-compact/` (for Compact compaction).
///
/// The conversion is idempotent: existing Vortex files will not be regenerated.
pub async fn convert_parquet_directory_to_vortex(
    input_path: &Path,
    compaction: CompactionStrategy,
) -> anyhow::Result<()> {
    let (format, dir_name) = match compaction {
        CompactionStrategy::Compact => (Format::VortexCompact, Format::VortexCompact.name()),
        CompactionStrategy::Default => (Format::OnDiskVortex, Format::OnDiskVortex.name()),
    };

    let vortex_dir = input_path.join(dir_name);
    let parquet_path = input_path.join(Format::Parquet.name());
    create_dir_all(&vortex_dir).await?;

    let parquet_inputs = fs::read_dir(&parquet_path)?.collect::<std::io::Result<Vec<_>>>()?;
    trace!(
        "Found {} parquet files in {}",
        parquet_inputs.len(),
        parquet_path.to_str().unwrap()
    );

    let iter = parquet_inputs
        .iter()
        .filter(|entry| entry.path().extension().is_some_and(|e| e == "parquet"));

    let concurrency = calculate_concurrency();
    futures::stream::iter(iter)
        .map(|dir_entry| {
            let filename = {
                let mut temp = dir_entry.path();
                temp.set_extension("");
                temp.file_name().unwrap().to_str().unwrap().to_string()
            };
            let parquet_file_path = parquet_path.join(format!("{filename}.parquet"));
            let output_path = vortex_dir.join(format!("{filename}.{}", format.ext()));

            tokio::spawn(
                async move {
                    idempotent_async(output_path.as_path(), move |vtx_file| async move {
                        info!(
                            "Processing file '{filename}' with {:?} strategy",
                            compaction
                        );
                        convert_parquet_file_to_vortex(&parquet_file_path, &vtx_file, compaction)
                            .await
                    })
                    .await
                    .expect("Failed to write Vortex file")
                }
                .in_current_span(),
            )
        })
        .buffer_unordered(concurrency)
        .try_collect::<Vec<_>>()
        .await?;

    Ok(())
}

/// Convert a Parquet file to Vortex format with the specified compaction strategy.
///
/// Uses `idempotent_async` to skip conversion if the output file already exists.
pub async fn write_parquet_as_vortex(
    parquet_path: PathBuf,
    vortex_path: &str,
    compaction: CompactionStrategy,
) -> anyhow::Result<PathBuf> {
    idempotent_async(vortex_path, |output_fname| async move {
        let mut output_file = File::create(&output_fname).await?;
        let data = parquet_to_vortex_chunks(parquet_path).await?;
        let write_options = compaction.apply_options(SESSION.write_options());
        write_options
            .write(&mut output_file, data.into_array().to_array_stream())
            .await?;
        output_file.flush().await?;
        Ok(())
    })
    .await
}

/// Add GeoParquet `geo` file metadata to externally-sourced parquet we don't generate (e.g. SpatialBench `zone`).
pub async fn add_geoparquet_metadata(parquet_path: &Path, geo_json: &str) -> anyhow::Result<()> {
    let builder = ParquetRecordBatchStreamBuilder::new(File::open(parquet_path).await?).await?;
    let already_tagged = builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .is_some_and(|kvs| kvs.iter().any(|kv| kv.key == "geo"));
    if already_tagged {
        return Ok(());
    }

    let schema = Arc::clone(builder.schema());
    let mut reader = builder.build()?;

    let tmp_path = parquet_path.with_extension("parquet.tmp");
    let mut writer = AsyncArrowWriter::try_new(File::create(&tmp_path).await?, schema, None)?;
    while let Some(batch) = reader.try_next().await? {
        writer.write(&batch).await?;
    }
    writer.append_key_value_metadata(KeyValue::new("geo".to_string(), Some(geo_json.to_string())));
    writer.close().await?;
    tokio::fs::rename(&tmp_path, parquet_path).await?;
    Ok(())
}

/// Column names a parquet file's GeoParquet `geo` metadata marks as geometry.
fn geoparquet_columns(metadata: &ParquetMetaData) -> HashSet<String> {
    metadata
        .file_metadata()
        .key_value_metadata()
        .and_then(|kvs| kvs.iter().find(|kv| kv.key == "geo"))
        .and_then(|kv| kv.value.as_deref())
        .and_then(|geo| serde_json::from_str::<serde_json::Value>(geo).ok())
        .and_then(|value| {
            value
                .get("columns")
                .and_then(serde_json::Value::as_object)
                .map(|columns| columns.keys().cloned().collect())
        })
        .unwrap_or_default()
}

/// The erased `vortex.geo.wkb` extension dtype over a binary `storage` dtype.
fn wkb_ext_dtype(storage: &DType) -> VortexResult<ExtDTypeRef> {
    Ok(ExtDType::<WellKnownBinary>::try_new(GeoMetadata { crs: None }, storage.clone())?.erased())
}

/// Re-type the named binary columns of a struct `dtype` as `vortex.geo.wkb`, so the column
/// self-describes as geometry.
fn tag_geo_dtype(dtype: DType, geo_columns: &HashSet<String>) -> VortexResult<DType> {
    if geo_columns.is_empty() {
        return Ok(dtype);
    }
    let DType::Struct(fields, nullability) = &dtype else {
        return Ok(dtype);
    };
    let names = fields.names().clone();
    let tagged = names
        .iter()
        .zip(fields.fields())
        .map(|(name, field)| {
            if geo_columns.contains(name.as_ref()) && field.is_binary() {
                Ok(DType::Extension(wkb_ext_dtype(&field)?))
            } else {
                Ok(field)
            }
        })
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(DType::Struct(
        StructFields::new(names, tagged),
        *nullability,
    ))
}

/// Wrap the named binary columns of a struct `chunk` as `vortex.geo.wkb` extension arrays.
fn tag_geo_array(chunk: ArrayRef, geo_columns: &HashSet<String>) -> VortexResult<ArrayRef> {
    if geo_columns.is_empty() {
        return Ok(chunk);
    }
    let Some(struct_array) = chunk.as_opt::<Struct>() else {
        return Ok(chunk);
    };
    let names = struct_array.names().clone();
    let validity = struct_array.struct_validity();
    let len = struct_array.len();
    let tagged = names
        .iter()
        .zip(struct_array.iter_unmasked_fields())
        .map(|(name, field)| {
            if geo_columns.contains(name.as_ref()) && field.dtype().is_binary() {
                let ext = wkb_ext_dtype(field.dtype())?;
                Ok(ExtensionArray::try_new(ext, field.clone())?.into_array())
            } else {
                Ok(field.clone())
            }
        })
        .collect::<VortexResult<Vec<_>>>()?;
    Ok(StructArray::try_new(names, tagged, len, validity)?.into_array())
}
