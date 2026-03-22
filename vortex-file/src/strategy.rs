// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module defines the default layout strategy for a Vortex file.

use std::sync::Arc;
use std::sync::LazyLock;

use vortex_alp::ALP;
// Compressed encodings from encoding crates
// Canonical array encodings from vortex-array
use vortex_alp::ALPRD;
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
use vortex_array::arrays::Primitive;
use vortex_array::arrays::Struct;
use vortex_array::arrays::VarBin;
use vortex_array::arrays::VarBinView;
use vortex_array::dtype::FieldPath;
use vortex_array::session::ArrayRegistry;
use vortex_array::session::ArraySession;
#[cfg(feature = "zstd")]
use vortex_btrblocks::BtrBlocksCompressorBuilder;
#[cfg(feature = "zstd")]
use vortex_btrblocks::FloatCode;
#[cfg(feature = "zstd")]
use vortex_btrblocks::IntCode;
#[cfg(feature = "zstd")]
use vortex_btrblocks::StringCode;
use vortex_bytebool::ByteBool;
use vortex_datetime_parts::DateTimeParts;
use vortex_decimal_byte_parts::DecimalByteParts;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::Delta;
use vortex_fastlanes::FoR;
use vortex_fastlanes::RLE;
use vortex_fsst::FSST;
use vortex_layout::LayoutStrategy;
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
use vortex_pco::Pco;
use vortex_runend::RunEnd;
use vortex_sequence::Sequence;
use vortex_sparse::Sparse;
use vortex_utils::aliases::hash_map::HashMap;
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
pub static ALLOWED_ENCODINGS: LazyLock<ArrayRegistry> = LazyLock::new(|| {
    let session = ArraySession::empty();

    // Canonical encodings from vortex-array
    session.register(Null);
    session.register(Bool);
    session.register(Primitive);
    session.register(Decimal);
    session.register(VarBin);
    session.register(VarBinView);
    session.register(List);
    session.register(ListView);
    session.register(FixedSizeList);
    session.register(Struct);
    session.register(Extension);
    session.register(Chunked);
    session.register(Constant);
    session.register(Masked);
    session.register(Dict);

    // Compressed encodings from encoding crates
    session.register(ALP);
    session.register(ALPRD);
    session.register(BitPacked);
    session.register(ByteBool);
    session.register(DateTimeParts);
    session.register(DecimalByteParts);
    session.register(Delta);
    session.register(FoR);
    session.register(FSST);
    session.register(Pco);
    session.register(RLE);
    session.register(RunEnd);
    session.register(Sequence);
    session.register(Sparse);
    session.register(ZigZag);

    #[cfg(feature = "zstd")]
    session.register(Zstd);
    #[cfg(all(feature = "zstd", feature = "unstable_encodings"))]
    session.register(ZstdBuffers);

    session.registry().clone()
});

/// Build a new [writer strategy][LayoutStrategy] to compress and reorganize chunks of a Vortex file.
///
/// Vortex provides an out-of-the-box file writer that optimizes the layout of chunks on-disk,
/// repartitioning and compressing them to strike a balance between size on-disk,
/// bulk decoding performance, and IOPS required to perform an indexed read.
pub struct WriteStrategyBuilder {
    compressor: Option<Arc<dyn CompressorPlugin>>,
    row_block_size: usize,
    field_writers: HashMap<FieldPath, Arc<dyn LayoutStrategy>>,
    allow_encodings: Option<ArrayRegistry>,
    flat_strategy: Option<Arc<dyn LayoutStrategy>>,
}

impl Default for WriteStrategyBuilder {
    /// Create a new empty builder. It can be further configured,
    /// and then finally built yielding the [`LayoutStrategy`].
    fn default() -> Self {
        Self {
            compressor: None,
            row_block_size: 8192,
            field_writers: HashMap::new(),
            allow_encodings: Some(ALLOWED_ENCODINGS.clone()),
            flat_strategy: None,
        }
    }
}

impl WriteStrategyBuilder {
    /// Override the [compressor][CompressorPlugin] used for compressing chunks in the file.
    ///
    /// If not provided, this will use a BtrBlocks-style cascading compressor that tries to balance
    /// total size with decoding performance.
    pub fn with_compressor<C: CompressorPlugin>(mut self, compressor: C) -> Self {
        self.compressor = Some(Arc::new(compressor));
        self
    }

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
    pub fn with_allow_encodings(mut self, allow_encodings: ArrayRegistry) -> Self {
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

    /// Configure a write strategy that emits only CUDA-compatible encodings.
    ///
    /// This configures BtrBlocks to exclude schemes without CUDA kernel support.
    /// With the `unstable_encodings` feature, strings use buffer-level Zstd compression
    /// (`ZstdBuffersArray`) which preserves the array buffer layout for zero-conversion
    /// GPU decompression. Without it, strings use interleaved Zstd compression.
    #[cfg(feature = "zstd")]
    pub fn with_cuda_compatible_encodings(mut self) -> Self {
        let mut builder = BtrBlocksCompressorBuilder::default()
            .exclude_int([IntCode::Sparse, IntCode::Rle])
            .exclude_float([FloatCode::Rle, FloatCode::Sparse])
            .exclude_string([StringCode::Dict, StringCode::Fsst]);

        #[cfg(feature = "unstable_encodings")]
        {
            builder = builder.include_string([StringCode::ZstdBuffers]);
        }
        #[cfg(not(feature = "unstable_encodings"))]
        {
            builder = builder.include_string([StringCode::Zstd]);
        }

        self.compressor = Some(Arc::new(builder.build()));
        self
    }

    /// Configure a write strategy that uses compact encodings (Pco for numerics, Zstd for
    /// strings/binary).
    ///
    /// This provides better compression ratios than the default BtrBlocks strategy,
    /// especially for floating-point heavy datasets.
    #[cfg(feature = "zstd")]
    pub fn with_compact_encodings(mut self) -> Self {
        let btrblocks = BtrBlocksCompressorBuilder::default()
            .include_string([StringCode::Zstd])
            .include_int([IntCode::Pco])
            .include_float([FloatCode::Pco])
            .build();

        self.compressor = Some(Arc::new(btrblocks));
        self
    }

    /// Builds the canonical [`LayoutStrategy`] implementation, with the configured overrides
    /// applied.
    pub fn build(self) -> Arc<dyn LayoutStrategy> {
        let flat: Arc<dyn LayoutStrategy> = if let Some(flat) = self.flat_strategy {
            flat
        } else if let Some(allow_encodings) = self.allow_encodings {
            Arc::new(FlatLayoutStrategy::default().with_allow_encodings(allow_encodings))
        } else {
            Arc::new(FlatLayoutStrategy::default())
        };

        // 7. for each chunk create a flat layout
        let chunked = ChunkedLayoutStrategy::new(flat.clone());
        // 6. buffer chunks so they end up with closer segment ids physically
        let buffered = BufferedStrategy::new(chunked, 2 * ONE_MEG); // 2MB
        // 5. compress each chunk
        let compressing = if let Some(ref compressor) = self.compressor {
            CompressingStrategy::new_opaque(buffered, compressor.clone())
        } else {
            CompressingStrategy::new_btrblocks(buffered, true)
        };

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

        // 2.1. | 3.1. compress stats tables and dict values.
        let compress_then_flat = if let Some(ref compressor) = self.compressor {
            CompressingStrategy::new_opaque(flat, compressor.clone())
        } else {
            CompressingStrategy::new_btrblocks(flat, false)
        };

        // 3. apply dict encoding or fallback
        let dict = DictStrategy::new(
            coalescing.clone(),
            compress_then_flat.clone(),
            coalescing,
            Default::default(),
        );

        // 2. calculate stats for each row group
        let stats = ZonedStrategy::new(
            dict,
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
