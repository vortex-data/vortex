// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! An in-memory encoding that stores an integer array in a narrower integer type.
//!
//! A [`WidenArray`] keeps the logical dtype of an integer array, and stores its values and their
//! validity in a child with a strictly narrower integer type of the same signedness and
//! nullability. For example, an `i64` array whose values are all between 0 and 200 can store `i16`
//! values, but not `u8` values. [`WidenArray::encode`] picks the narrowest such type for a
//! [`PrimitiveArray`].
//!
//! Narrowing is a physical choice, so it **must not** change a result. Execution casts the values
//! to the logical dtype, and an operation runs on the values only where it gives the canonical
//! result. A `Widen` never wraps another `Widen`.
//!
//! Take executes its gather instead of leaving a lazy [`Dict`] under a `Widen`, because code such
//! as the DuckDB exporter and the dict layout writer looks for a `Dict` at the root of an array.
//!
//! `Widen` has no wire form yet, so the default [`ArraySession`] does not register it, and an
//! array that holds a `Widen` cannot be serialized.
//!
//! [`ArraySession`]: crate::session::ArraySession
//! [`Dict`]: crate::arrays::Dict
//! [`PrimitiveArray`]: crate::arrays::PrimitiveArray

mod array;
pub use array::WidenArraySlotsExt;
pub use array::WidenSlots;
pub use array::WidenSlotsView;
pub use vtable::WidenArray;

pub(crate) mod compute;
mod encode;

mod vtable;
use vortex_session::VortexSession;
pub use vtable::Widen;

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
    compute::rules::register_parent_rules(session);
}

#[cfg(test)]
mod tests;
