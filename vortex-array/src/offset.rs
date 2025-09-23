// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::{AsPrimitive, PrimInt};
use vortex_dtype::NativePType;
use vortex_scalar::Scalar;

// TODO(connor)[ListView]: Replace the bottom `ListBuilder` link with `ListViewBuilder`
/// A trait bound for integer types that can represent offsets.
///
/// This is mainly used in the builders [`ListBuilder`] and [`ListViewBuilder`].
///
/// [`ListBuilder`]: crate::builders::ListBuilder
/// [`ListViewBuilder`]: crate::builders::ListBuilder
pub trait OffsetPType: NativePType + PrimInt + AsPrimitive<usize> + Into<Scalar> {
    /// Returns the maximum offset value that can be represented by this type.
    fn max_offset() -> u64 {
        Self::PTYPE.max_value_as_u64()
    }
}

/// Implements [`OffsetPType`] for all possible `T` that have the correct bounds.
impl<T> OffsetPType for T where T: NativePType + PrimInt + AsPrimitive<usize> + Into<Scalar> {}
