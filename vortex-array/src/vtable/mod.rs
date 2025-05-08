//! This module contains the VTable definitions for a Vortex encoding.

mod serde;
mod validity;
mod variants;

use std::fmt::Debug;
use std::ops::Deref;

use arcref::ArcRef;
pub use serde::*;
pub use validity::*;
pub use variants::*;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use crate::stats::StatsSetRef;
use crate::{Array, ArrayRef, Encoding};

/// The encoding [`VTable`] encapsulates _all_ logic for both an Array and an Encoding in a
/// single trait, giving users a single entry-point to implement their own arrays.
///
/// From this [`VTable`], we derive implementations for the [`Array`] and [`Encoding`] traits.
pub trait VTable: 'static + Sized + Send + Sync + Debug {
    type Array: 'static + Send + Sync + Deref<Target = dyn Array>;
    type Encoding: 'static + Send + Sync + Deref<Target = dyn Encoding>;

    /// Specify the validity implementation for this encoding.
    type ValidityVTable: ValidityVTable<Self>;

    /// Optionally enable serde for this encoding by implementing the [`SerdeVTable`] trait.
    type SerdeVTable: SerdeVTable<Self> = ();

    // Declare which dtypes this encoding supports by providing vtables for each dtype.
    // type BoolVTable: BoolVTable<Self> = ();

    // Encoding Functions

    /// Returns the ID of the encoding.
    fn id(encoding: &Self::Encoding) -> ArcRef<str>;

    // Array Functions

    fn len(array: &Self::Array) -> usize;

    fn dtype(array: &Self::Array) -> &DType;

    fn stats(array: &Self::Array) -> StatsSetRef<'_>;

    // TODO(ngates): remove the result from this function, since the bounds are already checked.
    fn slice(array: &Self::Array, start: usize, stop: usize) -> VortexResult<Self::Array>;

    // TODO(ngates): remove the result from this function, since the bounds are already checked.
    fn scalar_at(array: &Self::Array, index: usize) -> VortexResult<Scalar>;

    /// Replace the children of this array with the given arrays.
    ///
    /// ## Pre-conditions
    ///
    /// - The number of given children matches the current number of children of the array.
    fn with_children(array: &Self::Array, children: &[ArrayRef]) -> VortexResult<Self::Array>;
}

#[macro_export]
macro_rules! vtable {
    ($V:tt) => {
        $crate::aliases::paste::paste! {
            #[derive(Debug)]
            pub struct [<$V VTable>];

            impl AsRef<dyn $crate::Array> for [<$V Array>] {
                fn as_ref(&self) -> &dyn $crate::Array {
                    // We can unsafe cast ourselves to an ArrayAdapter.
                    unsafe { &*(self as *const [<$V Array>] as *const $crate::ArrayAdapter<$V>) }
                }
            }

            impl std::ops::Deref for [<$V Array>] {
                type Target = dyn Array;

                fn deref(&self) -> &Self::Target {
                    // We can unsafe cast ourselves to an ArrayAdapter.
                    unsafe { &*(self as *const [<$V Array>] as *const $crate::ArrayAdapter<$V>) }
                }
            }

            impl AsRef<dyn Encoding> for [<$V Encoding>] {
                fn as_ref(&self) -> &dyn Encoding {
                    // We can unsafe cast ourselves to an EncodingAdapter.
                    unsafe { &*(self as *const [<$V Encoding>] as *const $crate::EncodingAdapter<$V>) }
                }
            }

            impl std::ops::Deref for [<$V Encoding>] {
                type Target = dyn $crate::Encoding;

                fn deref(&self) -> &Self::Target {
                    // We can unsafe cast ourselves to an EncodingAdapter.
                    unsafe { &*(self as *const [<$V Encoding>] as *const $crate::EncodingAdapter<$V>) }
                }
            }
        }
    };
}
