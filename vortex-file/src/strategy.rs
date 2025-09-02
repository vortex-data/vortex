// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the default layout strategy for a Vortex file.

use std::sync::Arc;

use vortex_array::stats::PRUNING_STATS;
use vortex_btrblocks::BtrBlocksCompressor;
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

enum ZoneMapConfiguration {
    NoZoneMaps,
    Resized {
        row_size: usize,
        compressor: CompressorConfiguration,
    },
}

impl Default for ZoneMapConfiguration {
    fn default() -> Self {
        Self::Resized {
            row_size: ROW_BLOCK_SIZE,
            compressor: CompressorConfiguration::btr_blocks(),
        }
    }
}

impl ZoneMapConfiguration {
    fn row_block_size(&self) -> Option<usize> {
        match self {
            ZoneMapConfiguration::NoZoneMaps => None,
            ZoneMapConfiguration::Resized { row_size, .. } => Some(*row_size),
        }
    }
}

enum CompressorConfiguration {
    NoCompressor,
    Compressor(Arc<dyn CompressorPlugin>),
}

impl CompressorConfiguration {
    fn btr_blocks() -> Self {
        Self::Compressor(Arc::new(BtrBlocksCompressor {
            exclude_int_dict_encoding: false,
        }))
    }

    fn btr_blocks_without_dict_of_int() -> Self {
        Self::Compressor(Arc::new(BtrBlocksCompressor {
            exclude_int_dict_encoding: true,
        }))
    }
}

/// Build a new [writer strategy][LayoutStrategy] to compress and reorganize chunks of a Vortex file.
///
/// Vortex provides an out-of-the-box file writer that optimizes the layout of chunks on-disk,
/// repartitioning and compressing them to strike a balance between size on-disk,
/// bulk decoding performance, and IOPS required to perform an indexed read.
pub struct WriteStrategyBuilder {
    executor: Option<Arc<dyn TaskExecutor>>,
    compressor: Option<CompressorConfiguration>,
    dictionary_compressor: Option<CompressorConfiguration>,
    zone_maps: Option<ZoneMapConfiguration>,
}

impl Default for WriteStrategyBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl WriteStrategyBuilder {
    /// Create a new empty builder. It can be further configured, and then finally built
    /// yielding the [`LayoutStrategy`].
    pub const fn new() -> Self {
        Self {
            executor: None,
            compressor: None,
            dictionary_compressor: None,
            zone_maps: None,
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

    /// Override the [compressor][CompressorPlugin] used for intra-chunk compression in the file.
    ///
    /// If not provided, this will use a BtrBlocks-style cascading compressor that tries to balance
    /// total size with decoding performance.
    pub fn with_compressor<C: CompressorPlugin>(mut self, compressor: C) -> Self {
        self.compressor = Some(CompressorConfiguration::Compressor(Arc::new(compressor)));
        self
    }

    /// Disable intra-chunk compression.
    pub fn without_compressor(mut self) -> Self {
        self.compressor = Some(CompressorConfiguration::NoCompressor);
        self
    }

    /// Override the compressor for the values of inter-chunk dictionaries.
    ///
    /// Use [without_dictionary] to entirely disable inter-chunk dictionaries.
    ///
    /// Use [with_compressor] to set the compressor used for the codes (i.e. keys) of the dictionary.
    ///
    /// If not provided, this will use a BtrBlocks-style cascading compressor that tries to balance
    /// total size with decoding performance.
    pub fn with_dictionary_compressor<C: CompressorPlugin>(mut self, compressor: C) -> Self {
        // TODO(DK): In theory, one might want dictionaries without value compression. We would need
        // a new enum, DictionaryConfiguration, with three options:
        // 1. NoDictionary
        // 2. UncompressedValues
        // 3. CompressedValues
        self.dictionary_compressor =
            Some(CompressorConfiguration::Compressor(Arc::new(compressor)));
        self
    }

    /// Disable inter-chunk dictionary compression.
    pub fn without_dictionary(mut self) -> Self {
        self.dictionary_compressor = Some(CompressorConfiguration::NoCompressor);
        self
    }

    /// Override the size of the zone maps and the compressor for the zone stats.
    ///
    /// If not provided, this uses 8Ki zones with a BtrBlocks-style cascading compressor on the stats.
    pub fn with_zone_maps<C: CompressorPlugin>(mut self, row_size: usize, compressor: C) -> Self {
        // TODO(DK): In theory, one might want zone maps without stat compression. Either the
        // compressor parameter could be an Option or we could add a new method
        // `with_uncompressed_zone_maps`.
        self.zone_maps = Some(ZoneMapConfiguration::Resized {
            row_size,
            compressor: CompressorConfiguration::Compressor(Arc::new(compressor)),
        });
        self
    }

    /// Disable zone maps.
    pub fn without_zone_maps(mut self) -> Self {
        self.zone_maps = Some(ZoneMapConfiguration::NoZoneMaps);
        self
    }

    /// Builds the canonical [`LayoutStrategy`] implementation, with the configured overrides
    /// applied.
    pub fn build(self) -> Arc<dyn LayoutStrategy> {
        let executor = self.executor.unwrap_or_else(|| Arc::new(LocalExecutor));

        let zone_maps = self.zone_maps.unwrap_or_default();
        let dictionary_compressor = self
            .dictionary_compressor
            .unwrap_or_else(CompressorConfiguration::btr_blocks);
        let compressor = self
            .compressor
            .unwrap_or_else(|| match &dictionary_compressor {
                CompressorConfiguration::NoCompressor => CompressorConfiguration::btr_blocks(),
                CompressorConfiguration::Compressor(..) => {
                    // If dictionary compression is enabled, do not try to dict compress the codes.
                    CompressorConfiguration::btr_blocks_without_dict_of_int()
                }
            });

        // 4. The last step is always: Buffer a couple megabytes (so that small I/O operations have
        // a better chance of reading other values from the same column) and write each chunk as a
        // flat layout.
        let mut strategy: Box<dyn LayoutStrategy> = Box::new(BufferedStrategy::new(
            ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()),
            2 * ONE_MEG, // 2MB
        ));

        // 3. Intra-chunk compression.
        //
        // If dictionary compression is enabled (see 2), then this compresses the *codes*.
        if let CompressorConfiguration::Compressor(compressor) = compressor {
            let compressing_strategy =
                CompressingStrategy::new_opaque(strategy, compressor.clone(), executor.clone(), 16);

            // If zone maps are enabled, ensure zones never span multiple compressed chunks.
            let row_block_size = zone_maps.row_block_size().unwrap_or(ROW_BLOCK_SIZE);

            strategy = Box::new(RepartitionStrategy::new(
                compressing_strategy,
                RepartitionWriterOptions {
                    block_size_minimum: ONE_MEG,
                    block_len_multiple: row_block_size,
                },
            ));
        };

        // 2. Inter-chunk dictionary compression.
        if let CompressorConfiguration::Compressor(values_compressor) = dictionary_compressor {
            let values_strategy = CompressingStrategy::new_opaque(
                FlatLayoutStrategy::default(),
                values_compressor,
                executor.clone(),
                1,
            );

            strategy = Box::new(DictStrategy::new(
                strategy,
                values_strategy,
                None::<Box<dyn LayoutStrategy>>,
                Default::default(),
                executor.clone(),
            ));
        }

        // 1. Create zone maps.
        if let ZoneMapConfiguration::Resized {
            row_size,
            compressor: stats_compressor,
        } = zone_maps
        {
            let zone_stats_strategy: Box<dyn LayoutStrategy> = match stats_compressor {
                CompressorConfiguration::NoCompressor => Box::new(FlatLayoutStrategy::default()),
                CompressorConfiguration::Compressor(stats_compressor) => {
                    Box::new(CompressingStrategy::new_opaque(
                        FlatLayoutStrategy::default(),
                        stats_compressor,
                        executor.clone(),
                        1,
                    ))
                }
            };
            let zone_strategy = ZonedStrategy::new(
                strategy,
                zone_stats_strategy,
                ZonedLayoutOptions {
                    block_size: row_size,
                    stats: PRUNING_STATS.into(),
                    max_variable_length_statistics_size: 64,
                    parallelism: 16,
                },
                executor.clone(),
            );
            strategy = Box::new(RepartitionStrategy::new(
                zone_strategy,
                RepartitionWriterOptions {
                    // No minimum block size in bytes
                    block_size_minimum: 0,
                    block_len_multiple: row_size,
                },
            ));
        }

        // 0. start with splitting columns
        Arc::new(StructStrategy::new(strategy))
    }
}
