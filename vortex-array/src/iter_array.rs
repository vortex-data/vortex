// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Concrete-iterator traits for element-wise access to array contents.
//!
//! These traits return a *named* `Iter<'a>` (rather than a closure-bound
//! `&mut dyn Iterator`), so callers can hold the iterator, compose it with
//! `zip` / `enumerate` / `collect`, and the optimizer can specialize the
//! loop body per validity case.
//!
//! Implementations are expected to do any one-time decode work (e.g.
//! materializing offsets, collecting buffer references, resolving the
//! validity bitmap) inside [`IterArray::iter`] / [`IterArrayValue::iter_value`]
//! and hold the resulting state on the returned iterator. The per-element
//! cost should be a small constant — no decode work per tick.
//!
//! Two traits are needed because some array types (notably bit-packed
//! [`BoolArray`](crate::arrays::BoolArray)) cannot yield a `&bool` reference
//! into their underlying storage. Such arrays implement
//! [`IterArrayValue`], which yields `Option<Item>` by value. Arrays whose
//! values live as contiguous `T` or `[u8]` in a buffer implement
//! [`IterArray`], which yields `Option<&Item>`.
//!
//! # Example
//!
//! ```ignore
//! use vortex_array::iter_array::IterArray;
//! use vortex_array::arrays::PrimitiveArray;
//!
//! let arr: PrimitiveArray = (0i32..1024).collect();
//! let total: i64 = IterArray::<i32>::iter(&arr)
//!     .flatten()
//!     .map(|v| *v as i64)
//!     .sum();
//! ```

/// Element-wise iteration over an array whose values are accessible by
/// reference into the underlying buffer.
///
/// `Item` is the (possibly unsized) element type — `T` for
/// [`PrimitiveArray`](crate::arrays::PrimitiveArray) and
/// [`DecimalArray`](crate::arrays::DecimalArray), `[u8]` for
/// [`VarBinArray`](crate::arrays::VarBinArray) and
/// [`VarBinViewArray`](crate::arrays::VarBinViewArray).
pub trait IterArray<Item: ?Sized> {
    /// The concrete iterator type returned by [`Self::iter`].
    type Iter<'a>: Iterator<Item = Option<&'a Item>> + 'a
    where
        Self: 'a,
        Item: 'a;

    /// Construct an iterator that yields one `Option<&Item>` per element.
    ///
    /// Implementations should perform any one-time setup (decoding offsets,
    /// collecting buffer references, resolving the validity bitmap) eagerly
    /// inside this method and stash the result on the returned iterator.
    fn iter(&self) -> Self::Iter<'_>;
}

/// Element-wise iteration over an array whose values are not addressable as
/// references — e.g. bit-packed booleans where each value occupies one bit.
///
/// `Item` must be `Copy` since values are returned by value.
pub trait IterArrayValue<Item: Copy> {
    /// The concrete iterator type returned by [`Self::iter_value`].
    type Iter<'a>: Iterator<Item = Option<Item>> + 'a
    where
        Self: 'a;

    /// Construct an iterator that yields one `Option<Item>` per element.
    ///
    /// The same per-call caching contract as [`IterArray::iter`] applies.
    fn iter_value(&self) -> Self::Iter<'_>;
}
