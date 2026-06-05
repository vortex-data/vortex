// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`Compaction`] encoding: normalize an array into a compact canonical form.
//!
//! A [`CompactionArray`] wraps a single child. Executing it produces a logically equivalent array
//! where every inner array has been normalized to its most compact representation:
//!
//! - **`ListView`** arrays are rebuilt to be zero-copy convertible to an Arrow-style `ListArray`
//!   (overlapping views deduplicated, leading/trailing garbage trimmed).
//! - **`VarBinView`** arrays have their data buffers garbage collected.
//! - **`Dict`** arrays are either decoded to a flat canonical array, or garbage collected in place
//!   (dead values removed, codes remapped) — whichever is estimated to be cheaper. This is driven
//!   by [`Dict`]'s [`CompactKernel`] via the parent-execution machinery (i.e. it is an
//!   `execute_parent` of `compaction(dict(..))`).
//! - **`Struct`** fields are recursively compacted.
//!
//! Like [`Slice`](crate::arrays::Slice), this is a transient, non-serializable encoding that only
//! exists to drive execution; it is not registered as a default encoding.

mod array;
mod compact;
mod vtable;

#[cfg(test)]
mod tests;

pub use array::CompactionArrayExt;
use vortex_error::VortexResult;
pub use vtable::Compaction;
pub use vtable::CompactionArray;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::arrays::Dict;
use crate::arrays::dict::DictArrayExt;
use crate::kernel::ExecuteParentKernel;
use crate::matcher::Matcher;

/// A kernel that lets a child encoding compact itself when it is the direct child of a
/// [`Compaction`] array, instead of being decoded to canonical and compacted structurally.
///
/// Implementations may read buffers. Return `Ok(None)` to decline, in which case the child is
/// decoded to canonical and [`compact_canonical`] is applied.
pub trait CompactKernel: VTable {
    /// Attempt to compact `array` directly. See the trait docs.
    fn compact(
        array: ArrayView<'_, Self>,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>>;
}

/// Adaptor that lifts a [`CompactKernel`] into an [`ExecuteParentKernel`] for the [`Compaction`]
/// parent.
#[derive(Default, Debug)]
pub struct CompactExecuteAdaptor<V>(pub V);

impl<V> ExecuteParentKernel<V> for CompactExecuteAdaptor<V>
where
    V: CompactKernel,
{
    type Parent = Compaction;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        _parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        debug_assert_eq!(child_idx, 0, "Compaction array has a single child");
        <V as CompactKernel>::compact(array, ctx)
    }
}

/// Matches arrays that are considered fully compacted: any canonical array, or a [`Dict`] whose
/// values are all referenced (i.e. already garbage collected).
///
/// This is the [`Matcher`] that [`ArrayRef::compact`] executes towards, so that a garbage-collected
/// dictionary is not further decoded to canonical.
pub struct Compacted;

impl Matcher for Compacted {
    type Match<'a> = &'a ArrayRef;

    fn matches(array: &ArrayRef) -> bool {
        if AnyCanonical::matches(array) {
            return true;
        }
        array
            .as_opt::<Dict>()
            .is_some_and(|dict| dict.has_all_values_referenced())
    }

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        Self::matches(array).then_some(array)
    }
}

impl ArrayRef {
    /// Normalize this array into a compact form.
    ///
    /// See the [module docs](self) for the exact normalization rules.
    pub fn compact(self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        CompactionArray::new(self)
            .into_array()
            .execute_until::<Compacted>(ctx)
    }
}
