// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module contains the VTable definitions for a Vortex encoding.

mod array;
mod compute;
mod decode;
mod encode;
mod operations;
mod serde;
mod validity;
mod visitor;

use std::fmt::Debug;
use std::ops::Deref;

pub use array::*;
pub use compute::*;
pub use decode::*;
pub use encode::*;
pub use operations::*;
pub use serde::*;
pub use validity::*;
pub use visitor::*;

use crate::{Array, Encoding, EncodingId, EncodingRef, IntoArray};

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
    /// The concrete array type for this encoding.
    type Array: 'static + Send + Sync + Clone + Debug + Deref<Target = dyn Array> + IntoArray;
    /// The concrete encoding type for this encoding.
    type Encoding: 'static + Send + Sync + Clone + Deref<Target = dyn Encoding>;

    /// VTable for basic array operations like length, dtype, and stats.
    type ArrayVTable: ArrayVTable<Self>;
    /// VTable for converting to canonical array format.
    type CanonicalVTable: CanonicalVTable<Self>;
    /// VTable for basic array operations like slicing and scalar access.
    type OperationsVTable: OperationsVTable<Self>;
    /// VTable for validity (null/non-null) operations.
    type ValidityVTable: ValidityVTable<Self>;
    /// VTable for visiting array structure and children.
    type VisitorVTable: VisitorVTable<Self>;

    /// Optionally enable implementing dynamic compute dispatch for this encoding.
    /// Can be disabled by assigning to the [`NotSupported`] type.
    type ComputeVTable: ComputeVTable<Self>;
    /// Optionally enable the [`EncodeVTable`] for this encoding. This allows it to partake in
    /// compression.
    /// Can be disabled by assigning to the [`NotSupported`] type.
    type EncodeVTable: EncodeVTable<Self>;
    /// Optionally enable serde for this encoding by implementing the [`SerdeVTable`] trait.
    /// Can be disabled by assigning to the [`NotSupported`] type.
    type SerdeVTable: SerdeVTable<Self>;

    /// Returns the ID of the encoding.
    fn id(encoding: &Self::Encoding) -> EncodingId;

    /// Returns the encoding for the array.
    fn encoding(array: &Self::Array) -> EncodingRef;
}

/// Placeholder type used to indicate when a particular vtable is not supported by the encoding.
///
/// This can be used as the type for optional vtables that an encoding doesn't implement.
pub struct NotSupported;

#[macro_export]
/// Macro to generate VTable boilerplate for an encoding.
///
/// This macro generates the VTable struct, array deref implementations,
/// and other required boilerplate for a Vortex encoding.
macro_rules! vtable {
    ($V:ident) => {
        $crate::aliases::paste::paste! {
            #[doc = concat!("VTable implementation for ", stringify!($V), " encoding.")]
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
                    // We can unsafe transmute ourselves to an ArrayAdapter.
                    std::sync::Arc::new(unsafe { std::mem::transmute::<[<$V Array>], $crate::ArrayAdapter::<[<$V VTable>]>>(self) })
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
