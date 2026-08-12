// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the default layout strategy for a Vortex file.

use std::num::NonZeroUsize;
use std::sync::Arc;

use vortex_array::dtype::FieldPath;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::integer::IntDictScheme;
use vortex_error::VortexExpect;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::buffered::BufferedStrategy;
use vortex_layout::layouts::cdc::CdcRepartitionStrategy;
use vortex_layout::layouts::cdc::ContentDefinedChunkingOptions;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::collect::CollectStrategy;
use vortex_layout::layouts::compressed::CompressingStrategy;
use vortex_layout::layouts::compressed::CompressorPlugin;
use vortex_layout::layouts::dict::writer::DictStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::list::writer::ListLayoutStrategy;
use vortex_layout::layouts::repartition::RepartitionStrategy;
use vortex_layout::layouts::repartition::RepartitionWriterOptions;
use vortex_layout::layouts::table::TableStrategy;
use vortex_layout::layouts::table::use_experimental_list_layout;
use vortex_layout::layouts::zoned::writer::ZonedLayoutOptions;
use vortex_layout::layouts::zoned::writer::ZonedStrategy;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

const ONE_MEG: u64 = 1 << 20;

/// How the compressor was configured on [`WriteStrategyBuilder`].
enum CompressorConfig {
    /// A [`BtrBlocksCompressorBuilder`] that [`WriteStrategyBuilder::build`] will finalize.
    /// `IntDictScheme` is automatically excluded from the data compressor to prevent recursive
    /// dictionary encoding.
    BtrBlocks(BtrBlocksCompressorBuilder),
    /// An opaque compressor used as-is for both data and stats compression.
    Opaque(Arc<dyn CompressorPlugin>),
}

/// Build a new [writer strategy](LayoutStrategy) to compress and reorganize chunks of a Vortex
/// file.
///
/// Vortex provides an out-of-the-box file writer that optimizes the layout of chunks on-disk,
/// repartitioning and compressing them to strike a balance between size on-disk,
/// bulk decoding performance, and IOPS required to perform an indexed read.
///
/// The default pipeline first splits struct columns, repartitions rows into fixed-size row blocks,
/// computes zoned statistics, applies dictionary encoding where useful, coalesces chunks toward
/// segment-sized blocks, compresses arrays, buffers nearby chunks, and finally writes flat leaf
/// layouts.
pub struct WriteStrategyBuilder {
    compressor: CompressorConfig,
    row_block_size: usize,
    data_block_target_bytes: Option<u64>,
    field_writers: HashMap<FieldPath, Arc<dyn LayoutStrategy>>,
    flat_strategy: Option<Arc<dyn LayoutStrategy>>,
    probe_compressor: Option<Arc<dyn CompressorPlugin>>,
    /// Whether to write list fields using [`ListLayoutStrategy`].
    ///
    /// [`ListLayoutStrategy`]: vortex_layout::layouts::list::writer::ListLayoutStrategy
    use_list_layout: bool,
    content_defined_chunking: Option<ContentDefinedChunkingOptions>,
}

impl WriteStrategyBuilder {
    /// Create a new builder that compresses with the schemes registered on `session`. It can be
    /// further configured, and then finally built yielding the [`LayoutStrategy`].
    pub fn from_session(session: &VortexSession) -> Self {
        Self {
            compressor: CompressorConfig::BtrBlocks(BtrBlocksCompressorBuilder::from_session(
                session,
            )),
            row_block_size: 8192,
            data_block_target_bytes: Some(ONE_MEG),
            field_writers: HashMap::new(),
            flat_strategy: None,
            probe_compressor: None,
            use_list_layout: use_experimental_list_layout(),
            content_defined_chunking: None,
        }
    }

    /// Override the row block size used for row repartitioning and zoned statistics.
    ///
    /// Larger blocks reduce footer/statistics overhead. Smaller blocks can improve pruning and
    /// random-access locality.
    pub fn with_row_block_size(mut self, row_block_size: usize) -> Self {
        self.row_block_size = row_block_size;
        self
    }

    /// Override the target uncompressed byte size used to coalesce data blocks.
    ///
    /// Passing `None` disables byte-size coalescing, so blocks retain the row granularity set by
    /// [`Self::with_row_block_size`].
    pub fn with_data_block_target_bytes(mut self, target_bytes: Option<u64>) -> Self {
        self.data_block_target_bytes = target_bytes;
        self
    }

    /// Enable writing list fields with [`ListLayoutStrategy`].
    ///
    /// **Note**: this is an unstable and experimental layout that is expected to change.
    /// Using it may lead to unreadable files in the future.
    ///
    /// [`ListLayoutStrategy`]: vortex_layout::layouts::list::writer::ListLayoutStrategy
    pub fn with_list_layout(mut self) -> Self {
        self.use_list_layout = true;
        self
    }

    /// Partition columns at content-defined boundaries instead of fixed row counts.
    ///
    /// **Note**: this is an unstable and experimental write mode that trades read
    /// optimizations for deduplication-friendly output. Boundaries are chosen by a rolling
    /// hash over the column's values (see
    /// [`CdcRepartitionStrategy`]), so files written from
    /// edited versions of the same data stay mostly byte-identical and deduplicate well on
    /// content-addressed stores such as Hugging Face Xet, which chunk files with the same
    /// GEAR rolling hash. Zone maps are still written over fixed row blocks; only the
    /// cross-chunk dictionary layer is omitted (dictionary state shared across chunks would
    /// couple a chunk's bytes to data outside it), and BtrBlocks' per-chunk dictionary
    /// schemes still apply.
    pub fn with_content_defined_chunking(mut self, options: ContentDefinedChunkingOptions) -> Self {
        self.content_defined_chunking = Some(options);
        self
    }

    /// Override the write layout for a specific field somewhere in the nested schema tree.
    ///
    /// The field path is matched after the root struct is split into columns. This is useful when a
    /// column needs a custom compression/layout policy while the rest of the file uses defaults.
    pub fn with_field_writer(
        mut self,
        field: impl Into<FieldPath>,
        writer: Arc<dyn LayoutStrategy>,
    ) -> Self {
        self.field_writers.insert(field.into(), writer);
        self
    }

    /// Override the flat layout strategy used for leaf chunks.
    ///
    /// By default, this uses [`FlatLayoutStrategy`]. This can be used to substitute a custom
    /// layout strategy, e.g. one that inlines constant array buffers for GPU reads.
    pub fn with_flat_strategy(mut self, flat: Arc<dyn LayoutStrategy>) -> Self {
        self.flat_strategy = Some(flat);
        self
    }

    /// Override the default [`BtrBlocksCompressorBuilder`] used for compression.
    ///
    /// The builder produces two compressors: one for data and one for stats.
    /// An explicitly built compressor is used as configured.
    pub fn with_btrblocks_builder(mut self, builder: BtrBlocksCompressorBuilder) -> Self {
        self.compressor = CompressorConfig::BtrBlocks(builder);
        self
    }

    /// Set the compressor to an opaque [`CompressorPlugin`].
    ///
    /// The compressor is used as-is for both data and stats compression. Use this when the
    /// compressor is already fully configured and should not be modified by the builder.
    pub fn with_compressor<C: CompressorPlugin>(mut self, compressor: C) -> Self {
        self.compressor = CompressorConfig::Opaque(Arc::new(compressor));
        self
    }

    /// Override the compressor used to probe whether a column is dict-eligible.
    pub fn with_probe_compressor<C: CompressorPlugin>(mut self, compressor: C) -> Self {
        self.probe_compressor = Some(Arc::new(compressor));
        self
    }

    /// Builds the canonical [`LayoutStrategy`] implementation, with the configured overrides
    /// applied.
    pub fn build(self) -> Arc<dyn LayoutStrategy> {
        let flat: Arc<dyn LayoutStrategy> = if let Some(flat) = self.flat_strategy {
            flat
        } else {
            Arc::new(FlatLayoutStrategy::default())
        };

        let compressor = self.compressor;

        // Experimental: repartition columns at content-defined boundaries so files written from
        // edited versions of the same data deduplicate well on content-addressed stores. Zone
        // maps are kept: statistics are computed over the fixed row blocks produced upstream of
        // the CDC repartitioner, which is free to re-chunk the data child afterwards because
        // ZonedLayout maps rows to zones by zone length alone. Only the cross-chunk dictionary
        // layer is omitted (a chunk's bytes must depend only on the chunk's own content); the
        // full BtrBlocks compressor, including its per-chunk dictionary schemes, still runs on
        // every chunk.
        if let Some(cdc_options) = self.content_defined_chunking {
            let data_compressor: Arc<dyn CompressorPlugin> = match &compressor {
                CompressorConfig::BtrBlocks(builder) => Arc::new(builder.clone().build()),
                CompressorConfig::Opaque(compressor) => Arc::clone(compressor),
            };
            // No BufferedStrategy here: it regroups chunks by accumulated byte counts, so an
            // edit anywhere shifts the physical interleaving of every later segment. Keeping
            // segment order a pure function of sequence ids keeps unchanged regions of two file
            // versions byte-identical, at the cost of a larger reorder window while writing.
            let chunked = ChunkedLayoutStrategy::new(Arc::clone(&flat));
            let compressing = CompressingStrategy::new(chunked, data_compressor);
            let cdc = CdcRepartitionStrategy::new(compressing, cdc_options);

            let stats_compressor: Arc<dyn CompressorPlugin> = match compressor {
                CompressorConfig::BtrBlocks(builder) => Arc::new(builder.build()),
                CompressorConfig::Opaque(compressor) => compressor,
            };
            let compress_then_flat = CompressingStrategy::new(flat, stats_compressor);

            let row_block_size =
                NonZeroUsize::new(self.row_block_size).vortex_expect("must be non 0");
            let stats = ZonedStrategy::new(
                cdc,
                compress_then_flat.clone(),
                ZonedLayoutOptions {
                    block_size: row_block_size,
                    ..Default::default()
                },
            );
            // One fixed-size chunk per zone for the stats accumulator; the CDC repartitioner
            // below re-cuts the same rows at content-defined boundaries, which is invariant to
            // how the incoming stream is split.
            let repartition = RepartitionStrategy::new(
                stats,
                RepartitionWriterOptions {
                    block_size_minimum: 0,
                    block_len_multiple: self.row_block_size,
                    block_size_target: None,
                    canonicalize: false,
                },
            );

            let validity_strategy = CollectStrategy::new(compress_then_flat);
            let table_strategy =
                TableStrategy::new(Arc::new(validity_strategy), Arc::new(repartition))
                    .with_field_writers(self.field_writers);
            return Arc::new(table_strategy);
        }

        // 7. for each chunk create a flat layout
        let chunked = ChunkedLayoutStrategy::new(Arc::clone(&flat));
        // 6. buffer chunks so they end up with closer segment ids physically
        let buffered = BufferedStrategy::new(chunked, 2 * ONE_MEG); // 2MB

        // 5. compress each chunk.
        // Exclude IntDictScheme from the data compressor because DictStrategy (step 3) already
        // dictionary-encodes columns. Allowing IntDictScheme here would redundantly
        // dictionary-encode the integer codes produced by that earlier step.
        let data_compressor: Arc<dyn CompressorPlugin> = match &compressor {
            CompressorConfig::BtrBlocks(builder) => Arc::new(
                builder
                    .clone()
                    .exclude_schemes([IntDictScheme.id()])
                    .build(),
            ),
            CompressorConfig::Opaque(compressor) => Arc::clone(compressor),
        };
        let compressing = CompressingStrategy::new(buffered, data_compressor);

        // 4. prior to compression, coalesce up to a minimum size
        let coalescing = RepartitionStrategy::new(
            compressing,
            RepartitionWriterOptions {
                // Write stream partitions roughly become segments. Because Vortex never reads less
                // than one segment, the size of segments and, therefore, partitions, must be small
                // enough to both (1) allow fine-grained random access reads and (2) allow
                // sufficient read concurrency for the desired throughput. One megabyte is small
                // enough to achieve this for S3 (Durner et al., "Exploiting Cloud Object Storage for
                // High-Performance Analytics", VLDB Vol 16, Iss 11).
                block_size_minimum: self.data_block_target_bytes.unwrap_or(0),
                block_len_multiple: self.row_block_size,
                block_size_target: self.data_block_target_bytes,
                canonicalize: true,
            },
        );

        // 2.1. | 3.1. compress stats tables and dict values.
        let stats_compressor: Arc<dyn CompressorPlugin> = match compressor {
            CompressorConfig::BtrBlocks(builder) => Arc::new(builder.build()),
            CompressorConfig::Opaque(compressor) => compressor,
        };
        let compress_then_flat = CompressingStrategy::new(flat, Arc::clone(&stats_compressor));

        // 3. apply dict encoding or fallback
        let probe_compressor = if let Some(probe_compressor) = self.probe_compressor {
            probe_compressor
        } else {
            Arc::clone(&stats_compressor)
        };
        let dict = DictStrategy::new(
            coalescing.clone(),
            compress_then_flat.clone(),
            coalescing,
            Default::default(),
            probe_compressor,
        );

        let row_block_size = NonZeroUsize::new(self.row_block_size).vortex_expect("must be non 0");

        // 2. calculate stats for each row group
        let stats = ZonedStrategy::new(
            dict,
            compress_then_flat.clone(),
            ZonedLayoutOptions {
                block_size: row_block_size,
                ..Default::default()
            },
        );

        // 1. repartition each column to fixed row counts
        let repartition = RepartitionStrategy::new(
            stats,
            RepartitionWriterOptions {
                // No minimum block size in bytes
                block_size_minimum: 0,
                // Always repartition into 8K row blocks
                block_len_multiple: self.row_block_size,
                block_size_target: None,
                canonicalize: false,
            },
        );

        // 0. start with splitting columns
        let validity_strategy = CollectStrategy::new(compress_then_flat.clone());

        // Take any field overrides from the builder and apply them to the final strategy.
        let mut table_strategy =
            TableStrategy::new(Arc::new(validity_strategy), Arc::new(repartition))
                .with_field_writers(self.field_writers);

        if self.use_list_layout {
            // We need a closure here to enable recursive application of list layout.
            table_strategy = table_strategy.with_list_layout_factory(
                move |list_layout: ListLayoutStrategy| -> Arc<dyn LayoutStrategy> {
                    let zoned = ZonedStrategy::new(
                        list_layout,
                        compress_then_flat.clone(),
                        ZonedLayoutOptions {
                            block_size: row_block_size,
                            ..Default::default()
                        },
                    );
                    Arc::new(RepartitionStrategy::new(
                        zoned,
                        RepartitionWriterOptions {
                            block_size_minimum: 0,
                            block_len_multiple: row_block_size.get(),
                            block_size_target: None,
                            canonicalize: false,
                        },
                    ))
                },
            );
        }

        Arc::new(table_strategy)
    }
}
