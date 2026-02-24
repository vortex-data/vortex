// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extension DTypes, and interfaces for working with extension types (dtypes, scalars, and arrays).
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
pub use vtable::ExtVTable;

mod plugin;
pub use plugin::ExtDTypePlugin;

mod typed;
pub use typed::ExtDType;

mod erased;
pub use erased::ExtDTypeRef;

mod matcher;
pub use matcher::Matcher;

/// A unique identifier for an extension type
pub type ExtId = arcref::ArcRef<str>;

/// Private module to seal [`typed::DynExtDType`].
mod sealed {
    use crate::dtype::extension::ExtVTable;
    use crate::dtype::extension::typed::ExtDTypeInner;

    /// Marker trait to prevent external implementations of [`super::typed::DynExtDType`].
    pub(crate) trait Sealed {}

    /// This can be the **only** implementor for [`super::typed::DynExtDType`].
    impl<V: ExtVTable> Sealed for ExtDTypeInner<V> {}
}
