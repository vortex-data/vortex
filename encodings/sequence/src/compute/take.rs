// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use num_traits::cast::NumCast;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::{TakeKernel, TakeKernelAdapter};
use vortex_array::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};
use vortex_dtype::{NativePType, match_each_native_ptype};
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{SequenceArray, SequenceVTable};

impl TakeKernel for SequenceVTable {
    #[allow(clippy::cast_possible_truncation)]
    fn take(&self, array: &SequenceArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let is_nullable = indices.dtype().is_nullable();
        let mask = indices.validity_mask();
        let indices = indices.to_primitive();
        let a = match_each_native_ptype!(indices.ptype(), |T| {
            let indices = indices.as_slice::<T>();
            match_each_native_ptype!(array.ptype(), |S| {
                let mul = array.multiplier().as_primitive::<S>();
                let base = array.base().as_primitive::<S>();
                take(mul, base, indices, is_nullable.then_some(mask))
            })
        });

        Ok(a.into_array())
    }
}

fn take<T: NativePType, S: NativePType>(
    mul: S,
    base: S,
    indices: &[T],
    is_valid: Option<Mask>,
) -> PrimitiveArray {
    match is_valid {
        Some(mask) => {
            PrimitiveArray::from_option_iter(indices.iter().enumerate().map(|(mask_index, i)| {
                mask.value(mask_index).then(|| {
                    let i = <S as NumCast>::from::<T>(*i).vortex_expect("all valid indices fit");
                    base + i * mul
                })
            }))
        }
        None => PrimitiveArray::from_iter(indices.iter().map(|i| {
            let i = <S as NumCast>::from::<T>(*i).vortex_expect("all indices fit");
            base + i * mul
        })),
    }
}

register_kernel!(TakeKernelAdapter(SequenceVTable).lift());
