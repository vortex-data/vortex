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
//! A decimal child has the logical scale and a lower precision. Its dtype is a label that the
//! values cannot check: the child precision **must** hold every valid value.
//! [`WidenArray::encode_decimal`] picks the smallest precision that holds the valid values and the
//! scale. Decimal execution copies the child values into the logical canonical width.
//!
//! Narrowing is a physical choice, so it **must not** change a result. Execution casts the values
//! to the logical dtype for integers, and an operation runs on the values only where it gives the
//! canonical result. A `Widen` never wraps another `Widen`.
//!
//! Take executes its gather instead of leaving a lazy [`Dict`] under a `Widen`, because code such
//! as the DuckDB exporter and the dict layout writer looks for a `Dict` at the root of an array.
//!
//! `Widen` has no wire form yet. The default [`ArraySession`] registers [`WidenPlugin`], which
//! writes an integer `Widen` as its canonical array and a decimal `Widen` as the narrow decimal
//! node of its child. The decimal node stores its width in metadata and takes its precision from
//! the containing format. Serialization keeps the `Widen` statistics, and readers never see
//! `vortex.widen`.
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
