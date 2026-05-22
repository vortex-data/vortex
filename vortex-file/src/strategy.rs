// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the default layout strategy for a Vortex file.

use std::sync::Arc;
use std::sync::LazyLock;

use vortex_alp::ALP;
use vortex_alp::ALPRD;
use vortex_array::ArrayId;
use vortex_array::VTable;
use vortex_array::arrays::Bool;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Decimal;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Extension;
use vortex_array::arrays::FixedSizeList;
use vortex_array::arrays::List;
use vortex_array::arrays::ListView;
use vortex_array::arrays::Masked;
use vortex_array::arrays::Null;
use vortex_array::arrays::Patched;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::Struct;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::VarBinView;
use vortex_array::arrays::Variant;
use vortex_array::arrays::patched::use_experimental_patches;
use vortex_array::dtype::FieldPath;
use vortex_btrblocks::BtrBlocksCompressorBuilder;
use vortex_btrblocks::SchemeExt;
use vortex_btrblocks::schemes::integer::IntDictScheme;
use vortex_bytebool::ByteBool;
use vortex_datetime_parts::DateTimeParts;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::Delta;
use vortex_fastlanes::FoR;
use vortex_fastlanes::RLE;
use vortex_fsst::FSST;
use vortex_layout::LayoutStrategy;
use vortex_layout::layouts::array_tree::writer as array_tree_writer;
use vortex_layout::layouts::buffered::BufferedStrategy;
use vortex_layout::layouts::chunked::writer::ChunkedLayoutStrategy;
use vortex_layout::layouts::collect::CollectStrategy;
use vortex_layout::layouts::compressed::CompressingStrategy;
use vortex_layout::layouts::compressed::CompressorPlugin;
use vortex_layout::layouts::dict::writer::DictStrategy;
use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
use vortex_layout::layouts::repartition::RepartitionStrategy;
use vortex_layout::layouts::repartition::RepartitionWriterOptions;
use vortex_layout::layouts::table::TableStrategy;
use vortex_layout::layouts::zoned::writer::ZonedLayoutOptions;
use vortex_layout::layouts::zoned::writer::ZonedStrategy;
#[cfg(feature = "unstable_encodings")]
use vortex_onpair::OnPair;
use vortex_pco::Pco;
use vortex_runend::RunEnd;
use vortex_sequence::Sequence;
use vortex_sparse::Sparse;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;
use vortex_zigzag::ZigZag;
#[cfg(feature = "zstd")]
use vortex_zstd::Zstd;
#[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
use vortex_zstd::ZstdBuffers;

const ONE_MEG: u64 = 1 << 20;

/// Static registry of all allowed array encodings for file writing.
///
/// This includes all canonical encodings from vortex-array plus all compressed
/// encodings from the various encoding crates.
pub static ALLOWED_ENCODINGS: LazyLock<HashSet<ArrayId>> = LazyLock::new(|| {
    let mut allowed = HashSet::new();

    // Canonical encodings from vortex-array
    allowed.insert(Null.id());
    allowed.insert(Bool.id());
    allowed.insert(Primitive.id());
    allowed.insert(Decimal.id());
    allowed.insert(VarBin.id());
    allowed.insert(VarBinView.id());
    allowed.insert(List.id());
    allowed.insert(ListView.id());
    allowed.insert(FixedSizeList.id());
    allowed.insert(Struct.id());
    allowed.insert(Extension.id());
    allowed.insert(Chunked.id());
    allowed.insert(Constant.id());
    allowed.insert(Masked.id());
    allowed.insert(Dict.id());
    allowed.insert(Variant.id());

    // Compressed encodings from encoding crates
    allowed.insert(ALP.id());
    allowed.insert(ALPRD.id());
    allowed.insert(BitPacked.id());
    allowed.insert(ByteBool.id());
    allowed.insert(DateTimeParts.id());
    allowed.insert(DecimalByteParts.id());
    allowed.insert(Delta.id());
    allowed.insert(FoR.id());
    allowed.insert(FSST.id());
    #[cfg(feature = "unstable_encodings")]
    allowed.insert(OnPair.id());
    allowed.insert(Pco.id());
    allowed.insert(RLE.id());
    allowed.insert(RunEnd.id());
    allowed.insert(Sequence.id());
    allowed.insert(Sparse.id());
    allowed.insert(ZigZag.id());

    // Experimental encodings

    if use_experimental_patches() {
        allowed.insert(Patched.id());
    }

    #[cfg(feature = "zstd")]
    allowed.insert(Zstd.id());
    #[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
    allowed.insert(ZstdBuffers.id());

    allowed
});

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
pub struct WriteStrategyBuilder {
    compressor: CompressorConfig,
    row_block_size: usize,
    field_writers: HashMap<FieldPath, Arc<dyn LayoutStrategy>>,
    allow_encodings: Option<HashSet<ArrayId>>,
    flat_strategy: Option<Arc<dyn LayoutStrategy>>,
    array_tree: bool,
}

impl Default for WriteStrategyBuilder {
    /// Create a new empty builder. It can be further configured,
    /// and then finally built yielding the [`LayoutStrategy`].
    fn default() -> Self {
        Self {
            compressor: CompressorConfig::BtrBlocks(BtrBlocksCompressorBuilder::default()),
            row_block_size: 8192,
            field_writers: HashMap::new(),
            allow_encodings: Some(ALLOWED_ENCODINGS.clone()),
            flat_strategy: None,
            array_tree: false,
        }
    }
}

impl WriteStrategyBuilder {
    /// Override the row block size used to determine the zone map sizes.
    pub fn with_row_block_size(mut self, row_block_size: usize) -> Self {
        self.row_block_size = row_block_size;
        self
    }

    /// Override the default write layout for a specific field somewhere in the nested
    /// schema tree.
    pub fn with_field_writer(
        mut self,
        field: impl Into<FieldPath>,
        writer: Arc<dyn LayoutStrategy>,
    ) -> Self {
        self.field_writers.insert(field.into(), writer);
        self
    }

    /// Override the allowed array encodings for normalization.
    pub fn with_allow_encodings(mut self, allow_encodings: HashSet<ArrayId>) -> Self {
        self.allow_encodings = Some(allow_encodings);
        self
    }

    /// Override the flat layout strategy used for leaf chunks.
    ///
    /// By default, this uses [`FlatLayoutStrategy`]. This can be used to substitute a custom
    /// layout strategy, e.g. one that inlines constant array buffers for GPU reads.
    ///
    /// Passing a custom flat strategy implicitly disables the array-tree outlining feature
    /// (see [`Self::with_array_tree`]), since the custom strategy owns the leaf format.
    pub fn with_flat_strategy(mut self, flat: Arc<dyn LayoutStrategy>) -> Self {
        self.flat_strategy = Some(flat);
        self
    }

    /// Enable array-tree outlining: each chunk's encoding tree (without per-chunk statistics)
    /// is collected into a single auxiliary segment per column rather than being inlined
    /// alongside the chunk's data.
    ///
    /// Disabled by default. When enabled, the written file uses two encodings that older
    /// readers will not understand:
    /// [`vortex_layout::layouts::array_tree::ArrayTreeFlatLayout`] at the data leaves, and a
    /// wrapping [`vortex_layout::layouts::array_tree::ArrayTreeLayout`] that owns the
    /// consolidated auxiliary segment. Files written by this builder with the feature on
    /// require a reader that recognizes both encodings.
    ///
    /// Has no effect if a custom flat strategy is provided via
    /// [`Self::with_flat_strategy`] — the user-supplied leaf format wins.
    pub fn with_array_tree(mut self, array_tree: bool) -> Self {
        self.array_tree = array_tree;
        self
    }

    /// Override the default [`BtrBlocksCompressorBuilder`] used for compression.
    ///
    /// The builder is finalized during [`build`](Self::build), producing two compressors: one for
    /// data (with `IntDictScheme` excluded) and one for stats.
    pub fn with_btrblocks_builder(mut self, builder: BtrBlocksCompressorBuilder) -> Self {
        self.compressor = CompressorConfig::BtrBlocks(builder);
        self
    }

    /// Set the compressor to an opaque [`CompressorPlugin`].
    ///
    /// The compressor is used as-is for both data and stats compression.
    pub fn with_compressor<C: CompressorPlugin>(mut self, compressor: C) -> Self {
        self.compressor = CompressorConfig::Opaque(Arc::new(compressor));
        self
    }

    /// Builds the canonical [`LayoutStrategy`] implementation, with the configured overrides
    /// applied.
    pub fn build(self) -> Arc<dyn LayoutStrategy> {
        let flat: Arc<dyn LayoutStrategy> = if let Some(flat) = &self.flat_strategy {
            Arc::clone(flat)
        } else if let Some(allow_encodings) = &self.allow_encodings {
            Arc::new(FlatLayoutStrategy::default().with_allow_encodings(allow_encodings.clone()))
        } else {
            Arc::new(FlatLayoutStrategy::default())
        };

        // Data compressor: excludes IntDictScheme because DictStrategy (step 3 below) already
        // dictionary-encodes columns; allowing it here would redundantly dictionary-encode the
        // integer codes produced by that earlier step.
        let data_compressor: Arc<dyn CompressorPlugin> = match &self.compressor {
            CompressorConfig::BtrBlocks(builder) => Arc::new(
                builder
                    .clone()
                    .exclude_schemes([IntDictScheme.id()])
                    .build(),
            ),
            CompressorConfig::Opaque(compressor) => Arc::clone(compressor),
        };
        // Stats compressor: used for zone-map tables, dict values, and (when enabled) the
        // consolidated array-trees segment.
        let stats_compressor: Arc<dyn CompressorPlugin> = match &self.compressor {
            CompressorConfig::BtrBlocks(builder) => Arc::new(builder.clone().build()),
            CompressorConfig::Opaque(compressor) => Arc::clone(compressor),
        };
        let compress_then_flat =
            CompressingStrategy::new(Arc::clone(&flat), Arc::clone(&stats_compressor));
        let compress_then_flat_arc: Arc<dyn LayoutStrategy> = Arc::new(compress_then_flat.clone());

        let array_tree_enabled = self.array_tree && self.flat_strategy.is_none();
        let (data_leaf, array_tree_collector): (Arc<dyn LayoutStrategy>, _) = if !array_tree_enabled
        {
            (Arc::clone(&flat), None)
        } else {
            let data_flat = if let Some(allow_encodings) = &self.allow_encodings {
                FlatLayoutStrategy::default().with_allow_encodings(allow_encodings.clone())
            } else {
                FlatLayoutStrategy::default()
            };
            let (collector, leaf) =
                array_tree_writer::writer(data_flat, Arc::clone(&compress_then_flat_arc));
            (Arc::new(leaf), Some(collector))
        };

        // 7. for each chunk create a flat layout
        let chunked = ChunkedLayoutStrategy::new(data_leaf);
        // 6. buffer chunks so they end up with closer segment ids physically
        let buffered = BufferedStrategy::new(chunked, 2 * ONE_MEG); // 2MB

        // 5. compress each chunk.
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
                block_size_minimum: ONE_MEG,
                block_len_multiple: self.row_block_size,
                block_size_target: Some(ONE_MEG),
                canonicalize: true,
            },
        );

        // 3. apply dict encoding or fallback
        let dict = DictStrategy::new(
            coalescing.clone(),
            compress_then_flat.clone(),
            coalescing,
            Default::default(),
        );

        // 2.5. wrap dict in the array-tree collector if outlining is enabled.
        let data_pipeline: Arc<dyn LayoutStrategy> = if let Some(collector) = array_tree_collector {
            Arc::new(collector.wrap(dict))
        } else {
            Arc::new(dict)
        };

        // 2. calculate stats for each row group
        let stats = ZonedStrategy::new(
            data_pipeline,
            compress_then_flat.clone(),
            ZonedLayoutOptions {
                block_size: self.row_block_size,
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
        let validity_strategy = CollectStrategy::new(compress_then_flat);

        // Take any field overrides from the builder and apply them to the final strategy.
        let table_strategy = TableStrategy::new(Arc::new(validity_strategy), Arc::new(repartition))
            .with_field_writers(self.field_writers);

        Arc::new(table_strategy)
    }
}
