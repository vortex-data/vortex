// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stores integer, floating-point, and decimal values in a narrower child array.
//!
//! A [`WidenArray`] exposes the original dtype. Its child stores the same values at a strictly
//! narrower canonical width and supplies their validity. The wrapper and child have the same
//! nullability.
//!
//! An integer child has a narrower integer type of the same signedness. For example, an `i64` array
//! whose values are all between 0 and 200 can store `i16` values, but not `u8` values.
//! [`WidenArray::encode`] picks the narrowest such type for a [`PrimitiveArray`]. An encoding that
//! fixes the dtype of an integer child and stores it narrower uses
//! [`WidenArray::from_stored_integer`] on read and [`WidenArray::to_stored_integer`] on write.
//!
//! A floating-point child has a strictly narrower IEEE-754 type (`f16 < f32 < f64`) and **must
//! not** contain signaling NaNs. [`WidenArray::encode_float`] selects the narrowest type that
//! reproduces every valid value bit-for-bit on widening back, preserving `-0.0` vs `+0.0` and
//! canonical quiet NaNs.
//!
//! A decimal child has the logical scale and a lower precision, and **must** be valid for its dtype.
//! Comparisons, bounds checks, and fills use the child dtype to bound its valid values.
//! [`WidenArray::encode_decimal`] and [`WidenArray::from_stored_decimal`] choose child precision
//! from valid values or supplied bounds, at the logical scale. A widening decimal cast keeps the
//! source dtype on its child. Decimal execution copies child values directly into the logical
//! canonical width, since yielding a cast would produce another `Widen` instead of canonical values.
//!
//! Narrowing is a physical choice, so it **must not** change a result. Execution casts the values
//! to the logical dtype for primitives, and an operation runs on the values only where it gives the
//! canonical result. A `Widen` never wraps another `Widen`.
//!
//! Take executes its gather instead of leaving a lazy [`Dict`] under a `Widen`, because code such
//! as the DuckDB exporter and the dict layout writer looks for a `Dict` at the root of an array.
//!
//! The default [`ArraySession`] registers [`WidenPlugin`]. For a [`Decimal`](crate::arrays::Decimal)
//! child, the plugin writes a legacy `vortex.decimal` node so older readers can read it. The node
//! stores the child's buffer width, and the enclosing format supplies the parent's logical dtype.
//! [`WidenArray::from_stored_decimal`] handles these nodes in both old and newly written files.
//! Integer children and decimals stored in another encoding use `vortex.widen`, while
//! floating-point children use `vortex.widen.v2`. When edition checks are enabled, the target
//! edition must allow the chosen wire ID.
//!
//! [`ArraySession`]: crate::session::ArraySession
//! [`Dict`]: crate::arrays::Dict
//! [`PrimitiveArray`]: crate::arrays::PrimitiveArray

use vortex_session::VortexSession;

mod array;
pub use array::WidenArraySlotsExt;
pub use array::WidenSlots;
pub use array::WidenSlotsView;

pub mod compute;
pub mod encode;
mod stored;

mod vtable;
pub use vtable::Widen;
pub use vtable::WidenArray;
pub use vtable::WidenPlugin;
pub use vtable::widen_v2_id;

use crate::array::VTable;
use crate::arrays::Dict;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::widen::compute::numeric::WidenNumericKernel;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::builtin::compare::Compare;
use crate::scalar_fn::builtin::compare::CompareExecuteAdaptor;
use crate::scalar_fn::builtin::numeric::Numeric;

pub(crate) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
    kernels.register_execute_parent_kernel(Compare.id(), Widen, CompareExecuteAdaptor(Widen));
    kernels.register_execute_parent_kernel(Numeric.id(), Widen, WidenNumericKernel);
    kernels.register_execute_parent_kernel(Dict.id(), Widen, TakeExecuteAdaptor(Widen));
    compute::rules::register_parent_rules(session);
}

#[cfg(test)]
mod tests;
