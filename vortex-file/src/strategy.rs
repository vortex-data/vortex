// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the default layout strategy for a Vortex file.

use std::sync::Arc;

use vortex_array::stats::PRUNING_STATS;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::buffered::BufferedStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::compressed::BtrBlocksCompressedStrategy;
use vortex_layout::layouts::dict::writer::DictStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::repartition::{RepartitionStrategy, RepartitionWriterOptions};
use vortex_layout::layouts::struct_::writer::StructStrategy;
use vortex_layout::layouts::zoned::writer::{ZonedLayoutOptions, ZonedStrategy};

const ROW_BLOCK_SIZE: usize = 8192;

pub struct VortexLayoutStrategy;

impl VortexLayoutStrategy {
    pub fn new() -> Arc<dyn LayoutStrategy> {
        // 7. for each chunk create a flat layout
        let chunked = Arc::new(ChunkedLayoutStrategy::default());
        // 6. buffer chunks so they end up with closer segment ids physically
        let buffered = Arc::new(BufferedStrategy::new(chunked, 2 << 20)); // 2MB
        // 5. compress each chunk
        let compressing = Arc::new(BtrBlocksCompressedStrategy::new(buffered, 16));

        // 4. prior to compression, coalesce up to a minimum size
        let coalescing = Arc::new(RepartitionStrategy::new(
            compressing,
            RepartitionWriterOptions {
                block_size_minimum: 1 << 20,
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        ));

        // 2.1. | 3.1. compress stats tables and dict values.
        let compress_then_flat = Arc::new(BtrBlocksCompressedStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            1,
        ));

        // 3. apply dict encoding or fallback
        let dict = Arc::new(DictStrategy::new(
            coalescing.clone(),
            compress_then_flat.clone(),
            coalescing,
            Default::default(),
        ));

        // 2. calculate stats for each row group
        let stats = Arc::new(ZonedStrategy::new(
            dict,
            compress_then_flat,
            ZonedLayoutOptions {
                block_size: ROW_BLOCK_SIZE,
                stats: PRUNING_STATS.into(),
                max_variable_length_statistics_size: 64,
                parallelism: 16,
            },
        ));

        // 1. repartition each column to fixed row counts
        let repartition = Arc::new(RepartitionStrategy::new(
            stats,
            RepartitionWriterOptions {
                // No minimum block size in bytes
                block_size_minimum: 0,
                // Always repartition into 8K row blocks
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        ));

        // 0. start with splitting columns
        Arc::new(StructStrategy::new(repartition))
    }

    #[cfg(feature = "zstd")]
    pub fn compact(
        compressor: vortex_layout::layouts::compact::CompactCompressor,
    ) -> Arc<dyn LayoutStrategy> {
        use vortex_layout::layouts::compact::CompactCompressedStrategy;

        // 6. for each chunk create a flat layout
        let chunked = Arc::new(ChunkedLayoutStrategy::default());
        // 5. buffer chunks so they end up with closer segment ids physically
        let buffered = Arc::new(BufferedStrategy::new(chunked, 2 << 20)); // 2MB
        // 4. compress each chunk
        let compressing = Arc::new(CompactCompressedStrategy::new(
            buffered,
            16,
            compressor.clone(),
        ));

        // 3. prior to compression, coalesce up to a minimum size
        let coalescing = Arc::new(RepartitionStrategy::new(
            compressing,
            RepartitionWriterOptions {
                block_size_minimum: 1 << 20,
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        ));

        // 2.1. compress stats tables
        let compress_then_flat = Arc::new(CompactCompressedStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            1,
            compressor,
        ));

        // TODO: start applying dictionary encoding for variable-length fields
        // when helpful. It is probably best to avoid doing this for small
        // fixed-length fields like numbers.

        // 2. calculate stats for each row group
        let stats = Arc::new(ZonedStrategy::new(
            coalescing,
            compress_then_flat,
            ZonedLayoutOptions {
                block_size: ROW_BLOCK_SIZE,
                stats: PRUNING_STATS.into(),
                max_variable_length_statistics_size: 64,
                parallelism: 16,
            },
        ));

        // 1. repartition each column to fixed row counts
        let repartition = Arc::new(RepartitionStrategy::new(
            stats,
            RepartitionWriterOptions {
                // No minimum block size in bytes
                block_size_minimum: 0,
                // Always repartition into 8K row blocks
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        ));

        // 0. start with splitting columns
        Arc::new(StructStrategy::new(repartition))
    }
}
