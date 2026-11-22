// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extension dtypes and their typed, erased, and registered interfaces.
//!
//! An extension dtype gives semantic meaning to values stored in an ordinary Vortex storage dtype.
//! For example, a timestamp extension can store an integer while carrying unit/time-zone metadata.
//! Arrays with [`DType::Extension`](crate::dtype::DType::Extension) use
//! [`crate::arrays::ExtensionArray`] to pair the extension dtype with the storage array.
//!
//! Extension dtype implementations provide an [`ExtVTable`] for identity, metadata serialization,
//! the storage dtype declaration, scalar validation, and typed scalar unpacking. Register extension
//! plugins in a session before deserializing data that may reference them.
//!
//! [`ExtDType<V>`] provides typed access to a concrete [`ExtVTable`]. Its erased handle,
//! [`ExtDTypeRef`], stores different extension types behind one interface. [`Matcher`] retrieves
//! metadata for a known type, and [`ExtDTypePlugin`] reconstructs dtypes through the session
//! registry.
//!
//! ## File layout convention
//!
//! Each vtable-backed concept `Foo` separates these responsibilities:
//!
//! - `vtable.rs`: `FooVTable`, the non-object-safe trait users implement.
//! - `plugin.rs`: `FooPlugin`, the registry trait and its blanket implementation.
//! - `typed.rs`: `Foo<V>`, the typed wrapper, and the private `DynFoo` dispatch trait.
//! - `erased.rs`: `FooRef`, including display, debug, equality, and hashing.
//! - `matcher.rs`: `Matcher` and its implementation for `V: FooVTable`.

mod vtable;
pub use vtable::*;

mod plugin;
pub use plugin::*;

mod foreign;
pub(crate) use foreign::*;

mod typed;
pub use typed::*;

mod erased;
pub use erased::*;

mod matcher;
pub use matcher::*;
use vortex_session::registry::Id;

/// A unique identifier for an extension type
pub type ExtId = Id;

/// Private module to seal [`typed::DynExtDType`].
mod sealed {
    use crate::dtype::extension::ExtVTable;
    use crate::dtype::extension::typed::ExtDType;

    /// Marker trait to prevent external implementations of [`super::typed::DynExtDType`].
    pub(crate) trait Sealed {}

    /// This can be the **only** implementor for [`super::typed::DynExtDType`].
    impl<V: ExtVTable> Sealed for ExtDType<V> {}
}

#[cfg(test)]
mod tests;
