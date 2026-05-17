// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Concrete-iterator impl for [`DecimalArray`], parameterised by the
//! physical scalar type `T: NativeDecimalType`.

use std::mem::size_of;

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

#[expect(deprecated)]
use crate::ToCanonical as _;
use crate::arrays::DecimalArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::primitive::array::iter::ValidityCursor;
use crate::dtype::NativeDecimalType;
use crate::iter_array::IterArray;
use crate::validity::Validity;

/// Iterator over a [`DecimalArray`]'s elements as `&T`.
pub enum DecimalIter<'a, T: NativeDecimalType> {
    AllValid(std::slice::Iter<'a, T>),
    AllInvalid {
        remaining: usize,
    },
    WithValidity {
        values: std::slice::Iter<'a, T>,
        validity: ValidityCursor,
    },
}

impl<'a, T: NativeDecimalType> Iterator for DecimalIter<'a, T> {
    type Item = Option<&'a T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            DecimalIter::AllValid(it) => it.next().map(Some),
            DecimalIter::AllInvalid { remaining } => {
                if *remaining == 0 {
                    None
                } else {
                    *remaining -= 1;
                    Some(None)
                }
            }
            DecimalIter::WithValidity { values, validity } => {
                let v = values.next()?;
                let valid = validity.next_bit().unwrap_or(false);
                Some(valid.then_some(v))
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = match self {
            DecimalIter::AllValid(it) => it.len(),
            DecimalIter::AllInvalid { remaining } => *remaining,
            DecimalIter::WithValidity { values, .. } => values.len(),
        };
        (n, Some(n))
    }
}

impl<T: NativeDecimalType> ExactSizeIterator for DecimalIter<'_, T> {}

/// Borrow the decimal array's host bytes as a typed slice. Mirrors
/// [`PrimitiveData::as_slice`](crate::arrays::primitive::PrimitiveData::as_slice).
#[inline]
fn as_typed_slice<T: NativeDecimalType>(arr: &DecimalArray) -> &[T] {
    if arr.values_type() != T::DECIMAL_TYPE {
        vortex_panic!(
            "Attempted to get slice of type {} from decimal array of type {}",
            T::DECIMAL_TYPE,
            arr.values_type()
        );
    }
    let bb = arr.buffer_handle().as_host();
    // SAFETY: buffer alignment for T is checked at construction.
    unsafe { std::slice::from_raw_parts(bb.as_ptr().cast::<T>(), bb.len() / size_of::<T>()) }
}

impl<T: NativeDecimalType> IterArray<T> for DecimalArray {
    type Iter<'a> = DecimalIter<'a, T>;

    fn iter(&self) -> Self::Iter<'_> {
        let values = as_typed_slice::<T>(self).iter();
        let validity = self
            .validity()
            .vortex_expect("decimal validity should be derivable");
        match validity {
            Validity::NonNullable | Validity::AllValid => DecimalIter::AllValid(values),
            Validity::AllInvalid => DecimalIter::AllInvalid {
                remaining: self.len(),
            },
            Validity::Array(v) => {
                #[expect(deprecated)]
                let bits: BitBuffer = v.to_bool().into_bit_buffer();
                DecimalIter::WithValidity {
                    values,
                    validity: ValidityCursor::new(bits),
                }
            }
        }
    }
}
