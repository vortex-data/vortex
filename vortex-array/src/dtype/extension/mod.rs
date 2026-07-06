// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extension DTypes, and interfaces for working with extension types.
//!
//! An extension dtype gives semantic meaning to values stored in an ordinary Vortex storage dtype.
//! For example, a timestamp extension can store an integer while carrying unit/time-zone metadata.
//! Arrays with [`DType::Extension`](crate::dtype::DType::Extension) use
//! [`crate::arrays::ExtensionArray`] to pair the extension dtype with the storage array.
//!
//! Extension dtype implementations provide an [`ExtVTable`] for identity, metadata
//! serialization, storage dtype validation, scalar validation, and typed scalar unpacking. Register
//! extension plugins in a session before deserializing data that may reference them.
//!
//! ## File layout convention
//!
//! Each vtable-backed concept `Foo` follows this module structure:
//!
//! - `vtable.rs`  — `FooVTable` (the non-object-safe trait users implement)
//! - `plugin.rs`  — `FooPlugin` (registry trait for deserialization + blanket impl)
//! - `typed.rs`   — `Foo<V>` (typed wrapper) + `FooInner<V>` + `DynFoo` (private)
//! - `erased.rs`  — `FooRef` (erased ref + Display/Debug/PartialEq/Hash impls)
//! - `matcher.rs` — `Matcher` trait + blanket impl for `V: FooVTable`

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
