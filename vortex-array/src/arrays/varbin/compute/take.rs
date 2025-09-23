// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter::Sum;

use num_traits::PrimInt;
use vortex_dtype::{DType, NativePType, match_each_integer_ptype};
use vortex_error::{VortexResult, vortex_err, vortex_panic};
use vortex_mask::Mask;

use crate::arrays::VarBinVTable;
use crate::arrays::varbin::VarBinArray;
use crate::arrays::varbin::builder::VarBinBuilder;
use crate::compute::{TakeKernel, TakeKernelAdapter};
use crate::{Array, ArrayRef, IntoArray, ToCanonical, register_kernel};

impl TakeKernel for VarBinVTable {
    fn take(&self, array: &VarBinArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        let offsets = array.offsets().to_primitive();
        let data = array.bytes();
        let indices = indices.to_primitive();
        match_each_integer_ptype!(offsets.ptype(), |O| {
            match_each_integer_ptype!(indices.ptype(), |I| {
                Ok(take(
                    array
                        .dtype()
                        .clone()
                        .union_nullability(indices.dtype().nullability()),
                    offsets.as_slice::<O>(),
                    data.as_slice(),
                    indices.as_slice::<I>(),
                    array.validity_mask(),
                    indices.validity_mask(),
                )?
                .into_array())
            })
        })
    }
}

register_kernel!(TakeKernelAdapter(VarBinVTable).lift());

fn take<I: NativePType, O: NativePType + PrimInt + Sum>(
    dtype: DType,
    offsets: &[O],
    data: &[u8],
    indices: &[I],
    validity_mask: Mask,
    indices_validity_mask: Mask,
) -> VortexResult<VarBinArray> {
    if !validity_mask.all_true() || !indices_validity_mask.all_true() {
        return Ok(take_nullable(
            dtype,
            offsets,
            data,
            indices,
            validity_mask,
            indices_validity_mask,
        ));
    }

    let mut builder = VarBinBuilder::<u32>::with_capacity(indices.len());
    for &idx in indices {
        let idx = idx
            .to_usize()
            .ok_or_else(|| vortex_err!("Failed to convert index to usize: {}", idx))?;
        let start = offsets[idx]
            .to_usize()
            .ok_or_else(|| vortex_err!("Failed to convert offset to usize: {}", offsets[idx]))?;
        let stop = offsets[idx + 1].to_usize().ok_or_else(|| {
            vortex_err!("Failed to convert offset to usize: {}", offsets[idx + 1])
        })?;
        builder.append_value(&data[start..stop]);
    }
    Ok(builder.finish(dtype))
}

fn take_nullable<I: NativePType, O: NativePType + PrimInt>(
    dtype: DType,
    offsets: &[O],
    data: &[u8],
    indices: &[I],
    data_validity: Mask,
    indices_validity: Mask,
) -> VarBinArray {
    let mut builder = VarBinBuilder::<u32>::with_capacity(indices.len());
    for (idx, data_idx) in indices.iter().enumerate() {
        if !indices_validity.value(idx) {
            builder.append_null();
            continue;
        }
        let data_idx = data_idx
            .to_usize()
            .unwrap_or_else(|| vortex_panic!("Failed to convert index to usize: {}", data_idx));
        if data_validity.value(data_idx) {
            let start = offsets[data_idx].to_usize().unwrap_or_else(|| {
                vortex_panic!("Failed to convert offset to usize: {}", offsets[data_idx])
            });
            let stop = offsets[data_idx + 1].to_usize().unwrap_or_else(|| {
                vortex_panic!(
                    "Failed to convert offset to usize: {}",
                    offsets[data_idx + 1]
                )
            });
            builder.append_value(&data[start..stop]);
        } else {
            builder.append_null();
        }
    }
    builder.finish(dtype)
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_dtype::{DType, Nullability};

    use crate::Array;
    use crate::arrays::{PrimitiveArray, VarBinArray};
    use crate::compute::conformance::take::test_take_conformance;
    use crate::compute::take;

    #[test]
    fn test_null_take() {
        let arr = VarBinArray::from_iter([Some("h")], DType::Utf8(Nullability::NonNullable));

        let idx1: PrimitiveArray = (0..1).collect();

        assert_eq!(
            take(arr.as_ref(), idx1.as_ref()).unwrap().dtype(),
            &DType::Utf8(Nullability::NonNullable)
        );

        let idx2: PrimitiveArray = PrimitiveArray::from_option_iter(vec![Some(0)]);

        assert_eq!(
            take(arr.as_ref(), idx2.as_ref()).unwrap().dtype(),
            &DType::Utf8(Nullability::Nullable)
        );
    }

    #[rstest]
    #[case(VarBinArray::from_iter(
        ["hello", "world", "test", "data", "array"].map(Some),
        DType::Utf8(Nullability::NonNullable),
    ))]
    #[case(VarBinArray::from_iter(
        [Some("hello"), None, Some("test"), Some("data"), None],
        DType::Utf8(Nullability::Nullable),
    ))]
    #[case(VarBinArray::from_iter(
        [b"hello".as_slice(), b"world", b"test", b"data", b"array"].map(Some),
        DType::Binary(Nullability::NonNullable),
    ))]
    #[case(VarBinArray::from_iter(["single"].map(Some), DType::Utf8(Nullability::NonNullable)))]
    fn test_take_varbin_conformance(#[case] array: VarBinArray) {
        test_take_conformance(array.as_ref());
    }
}
