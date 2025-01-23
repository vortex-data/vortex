use vortex_array::array::{ConstantArray, PrimitiveArray};
use vortex_array::compute::{compare, take, CompareFn, Operator};
use vortex_array::validity::Validity;
use vortex_array::{ArrayDType, ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_buffer::Buffer;
use vortex_dtype::{match_each_unsigned_integer_ptype, NativePType};
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::{DictArray, DictEncoding};

impl CompareFn<DictArray> for DictEncoding {
    fn compare(
        &self,
        lhs: &DictArray,
        rhs: &ArrayData,
        operator: Operator,
    ) -> VortexResult<Option<ArrayData>> {
        // If the RHS is constant, then we just need to compare against our encoded values.
        if let Some(const_scalar) = rhs.as_constant() {
            // Ensure the other is the same length as the dictionary
            let compare_result = compare(
                lhs.values(),
                ConstantArray::new(const_scalar, lhs.values().len()),
                operator,
            )?;
            return take(compare_result, lhs.codes()).map(Some);
        }

        // It's a little more complex, but we could perform a comparison against the dictionary
        // values in the future.
        Ok(None)
    }

    fn compare_with_selection(
        &self,
        lhs: &DictArray,
        rhs: &ArrayData,
        operator: Operator,
        selection: &Mask,
    ) -> VortexResult<Option<ArrayData>> {
        if selection.selectivity() < 0.2 {
            if let Some(const_scalar) = rhs.as_constant() {
                // Ensure the other is the same length as the dictionary
                // println!("lhs.values().len() = {}", lhs.values().len());
                let compare_result = compare(
                    lhs.values(),
                    ConstantArray::new(const_scalar, lhs.values().len()),
                    operator,
                )?;

                let codes = lhs.clone().into_primitive()?;
                let validity = codes.validity().take(
                    &PrimitiveArray::new::<u64>(
                        selection
                            .indices()
                            .iter()
                            .map(|s| *s as u64)
                            .collect::<Buffer<u64>>(),
                        Validity::NonNullable,
                    )
                    .into_array(),
                )?;
                let takes_idx = match_each_unsigned_integer_ptype!( PType::try_from(codes.dtype())?, |$T| {
                    let codes = codes.as_slice::<$T>();
                    take_codes_with_sel(codes, selection, validity)
                });

                return take(compare_result, takes_idx).map(Some);
            }
        }
        Ok(None)
    }
}

fn take_codes_with_sel<T: NativePType>(codes: &[T], selection: &Mask, val: Validity) -> ArrayData {
    PrimitiveArray::new::<T>(
        selection
            .indices()
            .iter()
            .map(|idx| codes[*idx])
            .collect::<Buffer<T>>(),
        val,
    )
    .into_array()
}
