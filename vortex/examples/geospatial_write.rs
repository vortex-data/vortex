// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use futures::StreamExt;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ArrowReaderBuilder;
use tokio::fs::File;
use tokio::io::AsyncWriteExt;
use vortex::VortexSessionDefault;
use vortex_array::ArrayRef;
use vortex_array::arrow::FromArrowArray;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_dtype::DType;
use vortex_dtype::FieldName;
use vortex_dtype::FieldPath;
use vortex_dtype::arrow::FromArrowType;
use vortex_error::VortexError;
use vortex_file::WriteOptionsSessionExt;
use vortex_file::WriteStrategyBuilder;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::buffered::BufferedStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::compact::CompactCompressor;
use vortex_layout::layouts::compressed::CompressingStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::path::PathStrategy;
use vortex_layout::layouts::repartition::RepartitionStrategy;
use vortex_layout::layouts::repartition::RepartitionWriterOptions;
use vortex_layout::layouts::geo::GeoStrategy;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

/// Special strategy for writing chunks of data where we add extra index structures
/// to pushdown `ST_Contains` queries.
pub static COMPACT_RTREE_STRATEGY: LazyLock<Arc<dyn LayoutStrategy>> =
    LazyLock::new(|| make_rtree_strategy());

#[tokio::main]
pub async fn main() {
    // Load data from the Parquet dataset into our special format with the RTree indices.
    let f = File::open(
        "/Users/aduffy/Downloads/BuildingsParquet/custom_download_20251204_095222.parquet",
    )
    .await
    .unwrap();

    let mut reader = ParquetRecordBatchStreamBuilder::new(f).await.unwrap();

    let schema = reader.parquet_schema();

    // Drop the bbox column, since we don't use it for pruning and instead use a custom RTreeLayout
    let projection_mask = ProjectionMask::roots(&schema, [0, 1, 2, 3, 4, 5, 6, 7, 9]);

    reader = reader.with_projection(projection_mask);
    let mut reader = reader.build().unwrap();

    let dtype = DType::from_arrow(reader.schema().as_ref());

    let array_stream = reader
        .map(|record_batch| {
            record_batch
                .map_err(|e| VortexError::generic(e.into()))
                .map(|rb| ArrayRef::from_arrow(rb, false))
        })
        .boxed();

    // Setup the Vortex write to stream the records out.
    let mut file = File::create("buildings_rtree.vortex").await.unwrap();

    let session = VortexSession::default();
    let summary = session
        .write_options()
        .with_strategy(COMPACT_RTREE_STRATEGY.clone())
        .write(&mut file, ArrayStreamAdapter::new(dtype, array_stream))
        .await
        .unwrap();
    drop(summary);

    file.shutdown().await.unwrap();
}

/// Make a strategy which has special handling for DType::Binary chunks named "geometry".
fn make_rtree_strategy() -> Arc<dyn LayoutStrategy> {
    let validity = Arc::new(FlatLayoutStrategy::default());
    let fallback = WriteStrategyBuilder::new()
        .with_compressor(CompactCompressor::default())
        .build();

    // override the handling of the "geometry" column
    let leaf_writers = HashMap::from_iter([(
        FieldPath::from_name(FieldName::from("geometry")),
        geometry_writer(),
    )]);

    Arc::new(PathStrategy::new(leaf_writers, validity, fallback))
}

fn geometry_writer() -> Arc<dyn LayoutStrategy> {
    // 7. for each chunk create a flat layout
    let chunked = ChunkedLayoutStrategy::new(FlatLayoutStrategy::default());
    // 6. buffer chunks so they end up with closer segment ids physically
    let buffered = BufferedStrategy::new(chunked, 2 * 1024 * 1024); // 2MB
    // 5. compress each chunk with ZSTD/PCodec
    let compressing = CompressingStrategy::new_compact(buffered, CompactCompressor::default());

    // 4. prior to compression, coalesce up to a minimum size
    let coalescing = RepartitionStrategy::new(
        compressing,
        RepartitionWriterOptions {
            block_size_minimum: 1024 * 1024,
            block_len_multiple: 8_192,
            canonicalize: true,
        },
    );

    // 2.1. | 3.1. compress stats tables and dict values.
    let compress_then_flat = CompressingStrategy::new_compact(
        FlatLayoutStrategy::default(),
        CompactCompressor::default(),
    );

    // 2. calculate rtree for each block
    let stats = GeoStrategy::new(Arc::new(coalescing), Arc::new(compress_then_flat), 8_192);

    // 1. repartition each column to fixed row counts
    let repartition = RepartitionStrategy::new(
        stats,
        RepartitionWriterOptions {
            // No minimum block size in bytes
            block_size_minimum: 0,
            // Always repartition into 8K row blocks
            block_len_multiple: 8_192,
            canonicalize: false,
        },
    );

    Arc::new(repartition)
}
