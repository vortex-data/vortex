//! TypedPrimitiveArray is a typed wrapper around `PrimitiveArray`.
//!
//! It provides ergonomics for cases where you can guarantee at compile time that a particular `
//! PrimitiveArray` is of a certain type.
//!
//! Example:
//!
//! ```
//! use vortex_array::array::TypedPrimitiveArray;
//! use vortex_array::{IntoArray, IntoArrayVariant};
//! use vortex_buffer::buffer;
//!
//! // Create a new array of values
//! let values = buffer![1i32, 2, 3, 4];
//! let values = values.into_array().into_primitive().unwrap();
//! let typed_array = TypedPrimitiveArray::<i32>::try_from(values).unwrap();
//!
//! // Directly index the values like a normal slice.
//! fn sum(values: impl AsRef<[i32]>) -> i32 {
//!   values.as_ref().iter().sum()
//! }
//! assert_eq!(sum(&typed_array), 10i32);
//! ```
//!

use std::ops::Deref;

use vortex_dtype::NativePType;
use vortex_error::{vortex_bail, VortexResult};

use crate::array::PrimitiveArray;
use crate::variants::PrimitiveArrayTrait;

#[derive(Clone, Debug)]
pub struct TypedPrimitiveArray<T> {
    inner: PrimitiveArray,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: NativePType> TypedPrimitiveArray<T> {
    pub fn try_from(untyped: PrimitiveArray) -> VortexResult<Self> {
        if untyped.ptype() != T::PTYPE {
            vortex_bail!(
                "mismatched PTypes: expected {}, got {}",
                T::PTYPE,
                untyped.ptype()
            );
        }

        Ok(Self {
            inner: untyped,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<T> Deref for TypedPrimitiveArray<T> {
    type Target = PrimitiveArray;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

// Access the values in PrimitiveArray as a native slice of `T`.
impl<T: NativePType> AsRef<[T]> for TypedPrimitiveArray<T> {
    fn as_ref(&self) -> &[T] {
        self.inner.as_slice::<T>()
    }
}
