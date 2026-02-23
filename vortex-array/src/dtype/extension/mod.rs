// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Extension DTypes, and interfaces for working with extension types (dtypes, scalars, and arrays).

mod matcher;
pub use matcher::Matcher;

mod vtable;
pub use vtable::DynExtVTable;
pub use vtable::ExtVTable;

mod dtype;
pub use dtype::*;

/// A unique identifier for an extension type
pub type ExtId = arcref::ArcRef<str>;

/// Private module to seal [`ExtDTypeImpl`].
///
/// Note that this is not strictly necessary since [`ExtDTypeAdapter`] and [`ExtDTypeImpl`] are both
/// private to this module, this is just for hygiene.
mod sealed {
    use crate::dtype::extension::ExtDTypeAdapter;
    use crate::dtype::extension::ExtVTable;

    /// Marker trait to prevent external implementations of [`ExtDTypeImpl`].
    pub(crate) trait Sealed {}

    /// This can be the **only** implementor for [`ExtDTypeImpl`].
    impl<V: ExtVTable> Sealed for ExtDTypeAdapter<V> {}
}
