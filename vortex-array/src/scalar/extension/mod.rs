// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extension Scalar Values, and interfaces for working with them.
//!
//! We define normal [`Scalar`]s as the combination of a [`ScalarValue`] and a [`DType`].
//!
//! Similarly, we define an extension [`Scalar`] as the combination of an [`ExtScalarValueRef`] and
//! an [`ExtDTypeRef`].
//!
//! [`Scalar`]: crate::scalar::Scalar
//! [`ScalarValue`]: crate::scalar::ScalarValue
//! [`DType`]: crate::dtype::DType
//! [`ExtDTypeRef`]: crate::dtype::extension::ExtDTypeRef
//!
//! ## File layout convention
//!
//! Note that there is a single unified vtable for working with extension types at
//! [`vortex_array::dtype::extension::vtable`].
//!
//! Every other vtable-backed concept `FooScalarValue` follows this module structure:
//!
//! - `plugin.rs`: TODO
//! - `typed.rs`: TODO
//! - `erased.rs`: TODO
//! - `matcher.rs`: TODO

mod plugin;
// pub use plugin::ExtDTypePlugin;

mod typed;
// pub use typed::ExtDType;

mod erased;
// pub use erased::ExtDTypeRef;

mod matcher;
// pub use matcher::Matcher;

// /// Private module to seal [`typed::DynExtDType`].
// mod sealed {
//     use crate::dtype::extension::ExtVTable;
//     use crate::dtype::extension::typed::ExtDTypeInner;

//     /// Marker trait to prevent external implementations of [`super::typed::DynExtDType`].
//     pub(crate) trait Sealed {}

//     /// This can be the **only** implementor for [`super::typed::DynExtDType`].
//     impl<V: ExtVTable> Sealed for ExtDTypeInner<V> {}
// }
