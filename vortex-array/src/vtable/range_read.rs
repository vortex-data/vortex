// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Types for encoding-level range read planning.
//!
//! Each encoding can implement [`VTable::plan_range_read`](super::VTable::plan_range_read) to describe how its buffers and
//! children should be handled when only a sub-range of rows is needed. The layout planner
//! uses this information to issue targeted IO instead of reading the full segment.

use std::ops::Range;

use crate::dtype::DType;

/// Describes how a single encoding node handles a sub-segment range read.
///
/// Returned by [`VTable::plan_range_read`](super::VTable::plan_range_read) to tell the planner which buffer sub-ranges
/// are needed and how to handle children.
#[derive(Debug, Clone)]
pub struct EncodingRangeRead {
    /// For each of this encoding's buffers (by local index), the sub-range needed.
    pub buffer_sub_ranges: Vec<BufferSubRange>,
    /// For each child (by local index), how to handle it.
    pub children: Vec<ChildRangeRead>,
    /// How to compute the decode parameters.
    pub decode_info: RangeDecodeInfo,
}

/// Specifies which portion of a buffer is needed for a range read.
#[derive(Debug, Clone)]
pub enum BufferSubRange {
    /// The entire buffer is needed.
    Full,
    /// Only the specified byte range within the buffer is needed.
    Range(Range<usize>),
}

/// Specifies how a child should be handled during a range read.
#[derive(Debug, Clone)]
pub enum ChildRangeRead {
    /// Recurse into this child's encoding tree with the given row range.
    Recurse {
        /// The row range to request from this child.
        row_range: Range<usize>,
        /// The total row count of this child (before sub-ranging).
        row_count: usize,
        /// The DType of this child.
        dtype: DType,
    },
    /// Include all of this child's buffers fully (no sub-ranging).
    Full,
}

/// Describes how to compute `decode_len` and `post_slice` for the range read.
#[derive(Debug, Clone)]
pub enum RangeDecodeInfo {
    /// Self-contained decode parameters (for leaf encodings like Primitive, BitPacked, Bool).
    Leaf {
        /// The number of rows to pass to `decode()`.
        decode_len: usize,
        /// After decoding, slice to this range. `None` means no slicing needed.
        post_slice: Option<Range<usize>>,
    },
    /// Delegate to a child's decode info, optionally dividing by a factor.
    ///
    /// - `divisor = 1`: transparent delegation (FoR, ZigZag, Dict, ALP, ALPRD).
    /// - `divisor = list_size`: for FixedSizeList, where the child operates in element space.
    ///
    /// The planner will check divisibility and fall back to a full read if it fails.
    FromChild {
        /// The child index whose decode info to use.
        child_idx: usize,
        /// Divisor to apply to `decode_len` and `post_slice` ranges.
        divisor: usize,
    },
}
