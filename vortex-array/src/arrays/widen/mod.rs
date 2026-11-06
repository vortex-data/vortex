// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An in-memory encoding that stores integer and decimal arrays at a narrower canonical width.
//!
//! A [`WidenArray`] keeps the logical dtype of an integer or decimal array, and stores its values
//! and their validity in a child with the same nullability and a strictly narrower canonical width.
//!
//! An integer child has a narrower integer type of the same signedness. For example, an `i64` array
//! whose values are all between 0 and 200 can store `i16` values, but not `u8` values.
//! [`WidenArray::encode`] picks the narrowest such type for a [`PrimitiveArray`].
//!
//! A decimal child has the logical scale and a lower precision, and **must** be valid for its dtype.
//! Comparisons, bounds checks, and fills use the child dtype to bound its valid values.
//! [`WidenArray::encode_decimal`] and [`WidenArray::from_stored_decimal`] choose child precision
//! from valid values or supplied bounds, at the logical scale. A widening decimal cast keeps the
//! source dtype on its child. Decimal execution copies child values directly into the logical
//! canonical width, since yielding a cast would produce another `Widen` instead of canonical values.
//!
//! Narrowing is a physical choice, so it **must not** change a result. Execution casts the values
//! to the logical dtype for integers, and an operation runs on the values only where it gives the
//! canonical result. A `Widen` never wraps another `Widen`.
//!
//! Take executes its gather instead of leaving a lazy [`Dict`] under a `Widen`, because code such
//! as the DuckDB exporter and the dict layout writer looks for a `Dict` at the root of an array.
//!
//! `Widen` has no wire form yet. The default [`ArraySession`] registers [`WidenPlugin`], which
//! writes a decimal `Widen` with a [`Decimal`](crate::arrays::Decimal) leaf using the legacy
//! `vortex.decimal` wire form. The leaf supplies the stored width and the containing format supplies
//! the parent dtype. Serialization keeps the parent statistics and never executes the array.
//! Integer arrays and non-decimal leaves require explicit preparation by the caller.
//!
//! [`ArraySession`]: crate::session::ArraySession
//! [`Dict`]: crate::arrays::Dict
//! [`PrimitiveArray`]: crate::arrays::PrimitiveArray

use vortex_session::VortexSession;

mod array;
pub use array::WidenArraySlotsExt;
pub use array::WidenSlots;
pub use array::WidenSlotsView;

pub(crate) mod compute;
mod encode;

mod vtable;
pub use vtable::Widen;
pub use vtable::WidenArray;
pub use vtable::WidenPlugin;

use crate::array::VTable;
use crate::arrays::Dict;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::widen::compute::numeric::WidenNumericKernel;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;
use crate::scalar_fn::fns::zip::Zip;
use crate::scalar_fn::fns::zip::ZipExecuteAdaptor;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Binary.id(), Widen, CompareExecuteAdaptor(Widen));
    kernels.register_execute_parent_kernel(Binary.id(), Widen, WidenNumericKernel);
    kernels.register_execute_parent_kernel(Dict.id(), Widen, TakeExecuteAdaptor(Widen));
    kernels.register_execute_parent_kernel(Zip.id(), Widen, ZipExecuteAdaptor(Widen));
}

#[cfg(test)]
mod tests;
