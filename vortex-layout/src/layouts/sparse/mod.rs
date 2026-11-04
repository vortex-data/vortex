// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An experimental layout for null-dominated columns.
//!
//! A sparse layout stores only the rows of a nullable column that are present. Its `indices`
//! child holds the sorted row positions of those rows and its `values` child holds the rows
//! themselves, compacted, in the column's own dtype. Every other row is null, and so is
//! everything nested beneath it: a sparse struct stores its fields only for its present rows.
//!
//! The metadata holds the number of present rows in each block of `block_len` rows. A scan
//! maps any row range to the values it may touch from that alone, without reading `indices`.

pub mod writer;

use std::ops::Range;
use std::sync::Arc;

use vortex_array::ProstMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::Layout;
use crate::LayoutChildType;
use crate::LayoutDeserializeArgs;
use crate::LayoutId;
use crate::LayoutParts;
use crate::LayoutReaderContext;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::VTable;
use crate::children::OwnedLayoutChildren;
use crate::segments::SegmentSource;

/// Child index of the present-row positions.
pub const INDICES_CHILD_INDEX: usize = 0;
/// Child index of the compacted present rows.
pub const VALUES_CHILD_INDEX: usize = 1;

/// The dtype of the `indices` child.
pub const INDICES_DTYPE: DType = DType::Primitive(PType::U64, Nullability::NonNullable);

/// Sparse layout vtable.
#[derive(Clone, Debug)]
pub struct Sparse;

/// Sparse-layout-specific data.
#[derive(Clone, Debug)]
pub struct SparseData {
    blocks: SparseBlocks,
}

/// A sparse layout: a nullable column holding only its present rows.
pub type SparseLayout = Layout<Sparse>;

/// The present-row count of every block of rows, as cumulative offsets into the values.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseBlocks {
    block_len: u64,
    /// `offsets[b]` is the number of present rows before block `b`; the last entry is the total.
    offsets: Arc<[u64]>,
}

impl SparseBlocks {
    /// Builds the block offsets from the present-row count of each block of `block_len` rows.
    pub fn try_new(
        block_len: u64,
        counts: impl IntoIterator<Item = u64>,
        row_count: u64,
    ) -> VortexResult<Self> {
        vortex_ensure!(block_len > 0, "Sparse layout block length must be positive");
        let mut offsets = vec![0u64];
        let mut total = 0u64;
        for count in counts {
            total = total
                .checked_add(count)
                .ok_or_else(|| vortex_err!("Sparse layout present-row count overflow"))?;
            offsets.push(total);
        }
        let blocks = u64::try_from(offsets.len() - 1)?;
        vortex_ensure_eq!(
            blocks,
            row_count.div_ceil(block_len),
            "Sparse layout has {blocks} blocks for {row_count} rows of block length {block_len}"
        );
        for (block, window) in offsets.windows(2).enumerate() {
            let rows = block_len.min(row_count - u64::try_from(block)? * block_len);
            vortex_ensure!(
                window[1] - window[0] <= rows,
                "Sparse layout block {block} has more present rows than rows"
            );
        }
        Ok(Self {
            block_len,
            offsets: offsets.into(),
        })
    }

    /// The number of rows in each block.
    pub fn block_len(&self) -> u64 {
        self.block_len
    }

    /// The number of present rows.
    pub fn present_count(&self) -> u64 {
        self.offsets.last().copied().unwrap_or(0)
    }

    /// The present-row count of each block.
    pub fn counts(&self) -> impl Iterator<Item = u64> + '_ {
        self.offsets.windows(2).map(|window| window[1] - window[0])
    }

    /// The range of values that may hold the present rows of `rows`: the values of every block
    /// `rows` overlaps.
    pub fn values_range(&self, rows: &Range<u64>) -> Range<u64> {
        if rows.start >= rows.end {
            return 0..0;
        }
        let blocks = self.offsets.len() - 1;
        let first = usize::try_from(rows.start / self.block_len)
            .unwrap_or(usize::MAX)
            .min(blocks);
        let end = usize::try_from(rows.end.div_ceil(self.block_len))
            .unwrap_or(usize::MAX)
            .min(blocks);
        self.offsets[first]..self.offsets[end]
    }
}

impl VTable for Sparse {
    type LayoutData = SparseData;
    type Metadata = ProstMetadata<SparseLayoutMetadata>;

    fn id(&self) -> LayoutId {
        static ID: CachedId = CachedId::new("vortex.sparse");
        *ID
    }

    fn metadata(layout: &Layout<Self>) -> Self::Metadata {
        ProstMetadata(SparseLayoutMetadata {
            block_len: layout.blocks.block_len,
            block_counts: layout.blocks.counts().collect(),
        })
    }

    fn deserialize(
        &self,
        args: &LayoutDeserializeArgs<'_>,
        metadata: &SparseLayoutMetadata,
    ) -> VortexResult<Self::LayoutData> {
        validate_dtype(args.dtype)?;
        vortex_ensure_eq!(
            args.children.nchildren(),
            2,
            "Sparse layout expects indices and values children"
        );
        let blocks = SparseBlocks::try_new(
            metadata.block_len,
            metadata.block_counts.iter().copied(),
            args.row_count,
        )?;
        let indices = args.children.child(INDICES_CHILD_INDEX, &INDICES_DTYPE)?;
        let values = args.children.child(VALUES_CHILD_INDEX, args.dtype)?;
        validate_children(&blocks, indices.row_count(), values.row_count())?;
        Ok(SparseData { blocks })
    }

    fn child_dtype(layout: &Layout<Self>, idx: usize) -> VortexResult<DType> {
        match idx {
            INDICES_CHILD_INDEX => Ok(INDICES_DTYPE),
            VALUES_CHILD_INDEX => Ok(layout.dtype().clone()),
            _ => vortex_bail!("Invalid child index {idx} for SparseLayout"),
        }
    }

    fn child_type(_layout: &Layout<Self>, idx: usize) -> LayoutChildType {
        match idx {
            INDICES_CHILD_INDEX => LayoutChildType::Auxiliary("indices".into()),
            VALUES_CHILD_INDEX => LayoutChildType::Auxiliary("values".into()),
            _ => vortex_panic!("Invalid child index {idx} for SparseLayout"),
        }
    }

    fn new_reader(
        _layout: &Layout<Self>,
        _name: Arc<str>,
        _segment_source: Arc<dyn SegmentSource>,
        _session: &VortexSession,
        _ctx: &LayoutReaderContext,
    ) -> VortexResult<LayoutReaderRef> {
        vortex_bail!("The sparse layout is only readable by the v2 scan")
    }
}

impl Layout<Sparse> {
    /// Constructs a sparse layout of `row_count` rows from its children.
    ///
    /// `indices` holds the sorted positions of the present rows, `values` holds those rows in
    /// order, and `blocks` holds the present-row count of each block of rows.
    pub fn try_new(
        dtype: DType,
        row_count: u64,
        blocks: SparseBlocks,
        indices: LayoutRef,
        values: LayoutRef,
    ) -> VortexResult<Self> {
        validate_dtype(&dtype)?;
        vortex_ensure_eq!(indices.dtype(), &INDICES_DTYPE);
        vortex_ensure_eq!(values.dtype(), &dtype);
        vortex_ensure_eq!(
            u64::try_from(blocks.offsets.len() - 1)?,
            row_count.div_ceil(blocks.block_len)
        );
        validate_children(&blocks, indices.row_count(), values.row_count())?;
        Ok(LayoutParts::new(
            Sparse,
            dtype,
            row_count,
            Vec::new(),
            OwnedLayoutChildren::layout_children(vec![indices, values]),
            SparseData { blocks },
        )
        .into_typed())
    }

    /// Returns the present-row positions child.
    pub fn indices(&self) -> VortexResult<LayoutRef> {
        self.slot(INDICES_CHILD_INDEX)?
            .ok_or_else(|| vortex_err!("SparseLayout indices slot is absent"))
    }

    /// Returns the compacted present rows child.
    pub fn values(&self) -> VortexResult<LayoutRef> {
        self.slot(VALUES_CHILD_INDEX)?
            .ok_or_else(|| vortex_err!("SparseLayout values slot is absent"))
    }

    /// Returns the present-row count of each block of rows.
    pub fn blocks(&self) -> &SparseBlocks {
        &self.blocks
    }
}

fn validate_dtype(dtype: &DType) -> VortexResult<()> {
    vortex_ensure!(
        dtype.is_nullable(),
        "Sparse layout requires a nullable dtype, got {dtype}"
    );
    Ok(())
}

fn validate_children(blocks: &SparseBlocks, indices: u64, values: u64) -> VortexResult<()> {
    vortex_ensure_eq!(
        indices,
        blocks.present_count(),
        "Sparse layout indices row count does not match its present rows"
    );
    vortex_ensure_eq!(
        values,
        blocks.present_count(),
        "Sparse layout values row count does not match its present rows"
    );
    Ok(())
}

/// Serialized sparse layout metadata.
#[derive(prost::Message)]
pub struct SparseLayoutMetadata {
    /// The number of rows in each block.
    #[prost(uint64, tag = "1")]
    block_len: u64,
    /// The number of present rows in each block.
    #[prost(uint64, repeated, tag = "2")]
    block_counts: Vec<u64>,
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;

    use super::SparseBlocks;

    #[rstest]
    #[case::empty(0..0, 0..0)]
    #[case::inside_first_block(1..3, 0..2)]
    #[case::whole_first_block(0..4, 0..2)]
    #[case::spans_blocks(3..5, 0..2)]
    #[case::second_block(4..8, 2..2)]
    #[case::last_partial_block(8..10, 2..5)]
    #[case::everything(0..10, 0..5)]
    fn values_range_covers_overlapping_blocks(
        #[case] rows: std::ops::Range<u64>,
        #[case] expected: std::ops::Range<u64>,
    ) -> VortexResult<()> {
        // Ten rows in blocks of four: 2, 0 and 3 present rows.
        let blocks = SparseBlocks::try_new(4, [2, 0, 3], 10)?;
        assert_eq!(blocks.values_range(&rows), expected);
        Ok(())
    }

    #[test]
    fn rejects_mismatched_block_count() {
        assert!(SparseBlocks::try_new(4, [1, 1], 10).is_err());
    }

    #[test]
    fn rejects_overfull_block() {
        assert!(SparseBlocks::try_new(4, [2, 0, 3], 9).is_err());
    }
}
