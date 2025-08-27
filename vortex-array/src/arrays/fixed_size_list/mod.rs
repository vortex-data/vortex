#![allow(unused)] // TODO(connor)[FixedSizeList]: Remove this!

use vortex_dtype::DType;

use crate::{ArrayRef, stats::ArrayStats, validity::Validity};

/// The canonical encoding for fixed-size list arrays.
#[derive(Clone, Debug)]
pub struct FixedSizeListArray {
    /// The [`DType`] of the fixed-size list.
    ///
    /// This type **must** be the variant [`DType::FixedSizeList`].
    dtype: DType,

    /// TODO(connor): Is this field worth storing for convenience (even though it is in `DType`)?
    list_size: u32,

    /// The values data array, where each fixed-size list scalar is a slice of this `values` array.
    ///
    /// The fixed-size list scalars (or the elements of the array) are contiguous (regardless of
    /// nullability for easy lookups), each with equal size in memory.
    values: ArrayRef,

    /// The validity / null map of the array.
    ///
    /// Note that this null map refers to the fixed-size list scalars, **not** the elements of the
    /// _individual_ fixed-size list scalars. The `values` array will track individual value
    /// nullability.
    validity: Validity,

    /// The stats for this array.
    stats_set: ArrayStats,
}
