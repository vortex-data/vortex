// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the default layout strategy for a Vortex file.

use std::num::NonZeroUsize;
use std::sync::Arc;

use vortex_array::ArrayId;
use vortex_array::dtype::FieldPath;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::integer::IntDictScheme;
use vortex_error::VortexExpect;
use vortex_layout::LayoutStrategy;
use vortex_layout::LayoutStrategyEncodingValidator;
use vortex_layout::layouts::buffered::BufferedStrategy;
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
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

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
    allow_encodings: Option<HashSet<ArrayId>>,
    flat_strategy: Option<Arc<dyn LayoutStrategy>>,
    probe_compressor: Option<Arc<dyn CompressorPlugin>>,
    page_size: Option<usize>,
    /// Whether to write list fields using [`ListLayoutStrategy`].
    ///
    /// [`ListLayoutStrategy`]: vortex_layout::layouts::list::writer::ListLayoutStrategy
    use_list_layout: bool,
}

impl Default for WriteStrategyBuilder {
    /// Create a new empty builder. It can be further configured,
    /// and then finally built yielding the [`LayoutStrategy`].
    fn default() -> Self {
        Self {
            compressor: CompressorConfig::BtrBlocks(BtrBlocksCompressorBuilder::default()),
            row_block_size: 8192,
            data_block_target_bytes: Some(ONE_MEG),
            field_writers: HashMap::new(),
            allow_encodings: None,
            flat_strategy: None,
            probe_compressor: None,
            // Off by default: a reader without the paged layout encoding cannot descend into a
            // page, so paging changes who can read the file.
            page_size: None,
            use_list_layout: use_experimental_list_layout(),
        }
    }
}

impl WriteStrategyBuilder {
    /// Override the row block size used for row repartitioning and zoned statistics.
    ///
    /// Larger blocks reduce footer/statistics overhead. Smaller blocks can improve pruning and
    /// random-access locality.
    pub fn with_row_block_size(mut self, row_block_size: usize) -> Self {
        self.row_block_size = row_block_size;
        self
    }

    /// Group each chunked layout's children into pages of at most `page_size` chunks, each
    /// serialized into its own segment.
    ///
    /// A layout is a single recursive flatbuffer verified against a table limit, so a file with
    /// enough chunks becomes one its own reader rejects. Paging bounds the tables in every
    /// flatbuffer to roughly `page_size`, at the cost of one extra segment read per page touched.
    /// Zero, or leaving this unset, keeps chunks inline.
    ///
    /// Note that a reader without the `vortex.paged` layout encoding registered can inspect a
    /// paged file's layout tree but cannot scan it.
    pub fn with_page_size(mut self, page_size: usize) -> Self {
        self.page_size = Some(page_size);
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

    /// Override the allowed array encodings for file writing.
    ///
    /// The configured flat leaf strategy is wrapped in a [`LayoutStrategyEncodingValidator`]
    /// that recursively checks every chunk before passing it to the leaf writer. [`build`](Self::build)
    /// also restricts any [`BtrBlocksCompressorBuilder`] to these encodings, independent of the
    /// order in which the builder and this policy were configured.
    pub fn with_allow_encodings(mut self, allow_encodings: HashSet<ArrayId>) -> Self {
        self.allow_encodings = Some(allow_encodings);
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
    /// The builder is finalized during [`build`](Self::build), producing two compressors: one for
    /// data (with `IntDictScheme` excluded) and one for stats. Both are restricted to the
    /// configured allowed encodings at build time.
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

        // Restrict the compressor to the allowed encodings so no compressor derived below (data,
        // stats, and the defaulted probe compressor) can produce an encoding outside the policy,
        // regardless of the order in which the builder and the policy were configured.
        let compressor = match self.compressor {
            CompressorConfig::BtrBlocks(builder) => {
                CompressorConfig::BtrBlocks(match &self.allow_encodings {
                    Some(allow_encodings) => builder.retain_allowed_encodings(allow_encodings),
                    None => builder,
                })
            }
            opaque => opaque,
        };

        let flat: Arc<dyn LayoutStrategy> = if let Some(allow_encodings) = self.allow_encodings {
            Arc::new(LayoutStrategyEncodingValidator::new(flat, allow_encodings))
        } else {
            flat
        };

        // 7. for each chunk create a flat layout
        let chunked = ChunkedLayoutStrategy::new(Arc::clone(&flat))
            .with_page_size(self.page_size.unwrap_or(0));
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
