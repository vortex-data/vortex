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
//! Note that there is a single unified vtable for working with extension types
//! [`ExtVTable`](crate::dtype::extension::ExtVTable).
//!
//! Every other vtable-backed concept `FooScalarValue` follows this module structure:
//!
//! - `plugin.rs`: TODO
//! - `typed.rs`: TODO
//! - `erased.rs`: TODO

// mod plugin;
// pub use plugin::ExtScalarValuePlugin;

mod typed;
pub use typed::ExtScalarValue;

mod erased;
pub use erased::ExtScalarValueRef;

/// Private module to seal [`DynExtScalarValue`].
mod sealed {
    use crate::dtype::extension::ExtVTable;
    use crate::scalar::extension::typed::ExtScalarValueInner;

    /// Marker trait to prevent external implementations of [`DynExtScalarValue`].
    pub(super) trait Sealed {}

    /// This can be the **only** implementor for [`super::typed::DynExtScalarValue`].
    impl<V: ExtVTable> Sealed for ExtScalarValueInner<V> {}
}
