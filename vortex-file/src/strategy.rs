// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the default layout strategy for a Vortex file.

use std::sync::Arc;

use vortex_array::stats::PRUNING_STATS;
use vortex_layout::layouts::buffered::BufferedStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::compressed::{CompressingStrategy, CompressorPlugin};
use vortex_layout::layouts::dict::writer::DictStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::repartition::{RepartitionStrategy, RepartitionWriterOptions};
use vortex_layout::layouts::struct_::writer::StructStrategy;
use vortex_layout::layouts::zoned::writer::{ZonedLayoutOptions, ZonedStrategy};
use vortex_layout::{LayoutStrategy, LocalExecutor, TaskExecutor};

const ONE_MEG: u64 = 1 << 20;
const ROW_BLOCK_SIZE: usize = 8192;

/// Build a new [writer strategy][LayoutStrategy] to compress and reorganize chunks of a Vortex file.
///
/// Vortex provides an out-of-the-box file writer that optimizes the layout of chunks on-disk,
/// repartitioning and compressing them to strike a balance between size on-disk,
/// bulk decoding performance, and IOPS required to perform an indexed read.
pub struct WriteStrategyBuilder {
    executor: Option<Arc<dyn TaskExecutor>>,
    compressor: Option<Arc<dyn CompressorPlugin>>,
}

impl Default for WriteStrategyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl WriteStrategyBuilder {
    /// Create a new empty builder. It can be further configured, and then finally built
    /// yielding the [`WriteStrategy`].
    pub const fn new() -> Self {
        Self {
            executor: None,
            compressor: None,
        }
    }

    /// Override the executor for spawning blocking expensive work.
    ///
    /// If not provided, this defaults to an executor that blocks the current thread the
    /// `write_stream` task runs on.
    pub fn with_executor(mut self, executor: Arc<dyn TaskExecutor>) -> Self {
        self.executor = Some(executor.clone());
        self
    }

    /// Override the [compressor][CompressorPlugin] used for compressing chunks in the file.
    ///
    /// If not provided, this will use a BtrBlocks-style cascading compressor that tries to balance
    /// total size with decoding performance.
    pub fn with_compressor<C: CompressorPlugin>(mut self, compressor: C) -> Self {
        self.compressor = Some(Arc::new(compressor));
        self
    }

    /// Builds the canonical [`LayoutStrategy`] implementation, with the configured overrides
    /// applied.
    pub fn build(self) -> Arc<dyn LayoutStrategy> {
        let executor = self.executor.unwrap_or_else(|| Arc::new(LocalExecutor));

        // 7. for each chunk create a flat layout
        let chunked = ChunkedLayoutStrategy::new(FlatLayoutStrategy::default());
        // 6. buffer chunks so they end up with closer segment ids physically
        let buffered = BufferedStrategy::new(chunked, 2 * ONE_MEG); // 2MB
        // 5. compress each chunk
        let compressing = if let Some(ref compressor) = self.compressor {
            CompressingStrategy::new_opaque(buffered, compressor.clone(), executor.clone(), 16)
        } else {
            CompressingStrategy::new_btrblocks(buffered, executor.clone(), 16)
        };

        // 4. prior to compression, coalesce up to a minimum size
        let coalescing = RepartitionStrategy::new(
            compressing,
            RepartitionWriterOptions {
                block_size_minimum: ONE_MEG,
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        );

        // 2.1. | 3.1. compress stats tables and dict values.
        let compress_then_flat = if let Some(ref compressor) = self.compressor {
            CompressingStrategy::new_opaque(
                FlatLayoutStrategy::default(),
                compressor.clone(),
                executor.clone(),
                1,
            )
        } else {
            CompressingStrategy::new_btrblocks(FlatLayoutStrategy::default(), executor.clone(), 1)
        };

        // 3. apply dict encoding or fallback
        let dict = DictStrategy::new(
            coalescing.clone(),
            compress_then_flat.clone(),
            coalescing,
            Default::default(),
            executor.clone(),
        );

        // 2. calculate stats for each row group
        let stats = ZonedStrategy::new(
            dict,
            compress_then_flat,
            ZonedLayoutOptions {
                block_size: ROW_BLOCK_SIZE,
                stats: PRUNING_STATS.into(),
                max_variable_length_statistics_size: 64,
                parallelism: 16,
            },
            executor.clone(),
        );

        // 1. repartition each column to fixed row counts
        let repartition = RepartitionStrategy::new(
            stats,
            RepartitionWriterOptions {
                // No minimum block size in bytes
                block_size_minimum: 0,
                // Always repartition into 8K row blocks
                block_len_multiple: ROW_BLOCK_SIZE,
            },
        );

        // 0. start with splitting columns
        Arc::new(StructStrategy::new(repartition))
    }
}
