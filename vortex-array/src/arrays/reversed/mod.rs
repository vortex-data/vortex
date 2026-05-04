// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reverse encoding — yields the elements of the inner array in reverse order.
//!
//! [`ReversedArray`] is a lazy wrapper created by [`ArrayRef::reverse`].  The
//! optimizer is applied immediately after construction, collapsing common patterns
//! before any data is read:
//!
//! * **Double-reversal cancellation**: `Reversed(Reversed(x)) → x` — both wrappers
//!   are eliminated with zero data movement.
//! * **Dict codes reversal**: `Reversed(Dict(codes, values)) → Dict(Reversed(codes), values)` —
//!   only the codes array (typically `u8`/`u16`) is reversed; the values dictionary is
//!   reused unchanged.  This is the primary optimisation: most real-world columns are
//!   dictionary-encoded, so the per-chunk reversal cost is O(n_codes) rather than O(n_rows).
//!
//! For encodings that have no reduce rule the `ReversedArray` wrapper survives to
//! decode time, where [`execute.rs`](self::execute) reverses the canonical form
//! directly:
//!
//! * `Primitive`: iterates the typed buffer backwards — O(n), fully sequential.
//! * `Struct`: calls [`ArrayRef::reverse`] on every child field, enabling per-field
//!   optimisations (e.g. the Dict rule fires on dict-encoded struct fields).
//! * Everything else: falls back to a reversed-index `take`.
//!
//! ## Implementing a custom optimisation
//!
//! Encodings that can be reversed more efficiently than `take(reversed_indices)` should
//! implement [`ReverseReduce`] and register [`ReverseReduceAdaptor`] in their
//! `PARENT_RULES`.  See `dict/compute/reverse.rs` for a worked example.

mod array;
pub(crate) mod execute;
mod rules;
#[cfg(test)]
mod tests;
mod vtable;

pub use array::ReversedArrayExt;
pub use vtable::{Reversed, ReversedArray};

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::{ArrayView, VTable};
use crate::matcher::Matcher;
use crate::optimizer::rules::ArrayParentReduceRule;

/// Metadata-only reversal for encodings that can avoid a full `take`.
///
/// Implement this for your encoding and register [`ReverseReduceAdaptor`] in its
/// `PARENT_RULES` to enable structural reversal optimisation.  The most important
/// case is [`Dict`](crate::arrays::Dict): reversing only requires reversing the
/// codes array; the values dictionary is reused unchanged.
///
/// # Contract
///
/// The returned array, when decoded, must yield the same elements as `array` in
/// reverse order.  Return `None` to fall back to the default execution path.
pub trait ReverseReduce: VTable {
    /// Returns an array equivalent to reversing `array`, or `None` to fall back.
    fn reverse(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>>;
}

/// Adaptor that wraps a [`ReverseReduce`] implementation as an
/// [`ArrayParentReduceRule`].
///
/// Register a `ReverseReduceAdaptor(YourEncoding)` in your encoding's
/// `PARENT_RULES` constant to enable the structural reversal optimisation.
#[derive(Default, Debug)]
pub struct ReverseReduceAdaptor<V>(pub V);

impl<V: ReverseReduce> ArrayParentReduceRule<V> for ReverseReduceAdaptor<V> {
    type Parent = Reversed;

    fn reduce_parent(
        &self,
        array: ArrayView<'_, V>,
        _parent: <Self::Parent as Matcher>::Match<'_>,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        debug_assert_eq!(child_idx, 0, "ReversedArray has exactly one child");
        // A one-element (or empty) array is already its own reverse.
        if array.len() <= 1 {
            return Ok(Some(array.array().clone()));
        }
        <V as ReverseReduce>::reverse(array)
    }
}
