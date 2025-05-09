//! This module contains the VTable definitions for a Vortex encoding.

mod array;
mod decode;
mod encode;
mod operations;
mod serde;
mod validity;
mod visitor;

use std::fmt::Debug;
use std::ops::Deref;

use arcref::ArcRef;
pub use array::*;
pub use decode::*;
pub use encode::*;
pub use operations::*;
pub use serde::*;
pub use validity::*;
pub use visitor::*;

use crate::{Array, Encoding, EncodingRef};

/// The encoding [`VTable`] encapsulates logic for an Encoding type and associated Array type.
/// The logic is split across several "VTable" traits to enable easier code organization than
/// simply lumping everything into a single trait.
///
/// Some of these vtables are optional, such as the [`SerdeVTable`], which is only required if
/// the encoding supports serialization.
///
/// From this [`VTable`] trait, we derive implementations for the sealed [`Array`] and [`Encoding`]
/// traits via the [`crate::ArrayAdapter`] and [`crate::EncodingAdapter`] types respectively.
///
/// The functions defined in these vtable traits will typically document their pre- and
/// post-conditions. The pre-conditions are validated inside the [`Array`] and [`Encoding`]
/// implementations so do not need to be checked in the vtable implementations (for example, index
/// out of bounds). Post-conditions are validated after invocation of the vtable function and will
/// panic if violated.
pub trait VTable: 'static + Sized + Send + Sync + Debug {
    type Array: 'static + Send + Sync + Deref<Target = dyn Array>;
    type Encoding: 'static + Send + Sync + Deref<Target = dyn Encoding>;

    type ArrayVTable: ArrayVTable<Self>;
    type DecodeVTable: DecodeVTable<Self>;
    type OperationsVTable: OperationsVTable<Self>;
    type ValidityVTable: ValidityVTable<Self>;
    type VisitorVTable: VisitorVTable<Self>;

    /// Optionally enable the [`EncodeVTable`] for this encoding. This allows it to partake in
    /// compression.
    type EncodeVTable: EncodeVTable<Self> = ();
    /// Optionally enable serde for this encoding by implementing the [`SerdeVTable`] trait.
    type SerdeVTable: SerdeVTable<Self> = ();

    /// Returns the ID of the encoding.
    fn id(encoding: &Self::Encoding) -> ArcRef<str>;

    /// Returns the encoding for the array.
    fn encoding(array: &Self::Array) -> EncodingRef;
}

#[macro_export]
macro_rules! vtable {
    ($V:ident) => {
        $crate::aliases::paste::paste! {
            #[derive(Debug)]
            pub struct [<$V VTable>];

            impl AsRef<dyn $crate::Array> for [<$V Array>] {
                fn as_ref(&self) -> &dyn $crate::Array {
                    // We can unsafe cast ourselves to an ArrayAdapter.
                    unsafe { &*(self as *const [<$V Array>] as *const $crate::ArrayAdapter<[<$V VTable>]>) }
                }
            }

            impl std::ops::Deref for [<$V Array>] {
                type Target = dyn $crate::Array;

                fn deref(&self) -> &Self::Target {
                    // We can unsafe cast ourselves to an ArrayAdapter.
                    unsafe { &*(self as *const [<$V Array>] as *const $crate::ArrayAdapter<[<$V VTable>]>) }
                }
            }

            impl $crate::IntoArray for [<$V Array>] {
                fn into_array(self) -> $crate::ArrayRef {
                    std::sync::Arc::new(self)
                }
            }

            impl AsRef<dyn $crate::Encoding> for [<$V Encoding>] {
                fn as_ref(&self) -> &dyn $crate::Encoding {
                    // We can unsafe cast ourselves to an EncodingAdapter.
                    unsafe { &*(self as *const [<$V Encoding>] as *const $crate::EncodingAdapter<[<$V VTable>]>) }
                }
            }

            impl std::ops::Deref for [<$V Encoding>] {
                type Target = dyn $crate::Encoding;

                fn deref(&self) -> &Self::Target {
                    // We can unsafe cast ourselves to an EncodingAdapter.
                    unsafe { &*(self as *const [<$V Encoding>] as *const $crate::EncodingAdapter<[<$V VTable>]>) }
                }
            }
        }
    };
}
