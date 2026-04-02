// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::FoR;
impl OperationsVTable<FoR> for FoR {
    fn scalar_at(
        array: ArrayView<'_, FoR>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let encoded_pvalue = array.encoded().scalar_at(index)?;
        let encoded_pvalue = encoded_pvalue.as_primitive();
        let reference = array.reference_scalar();
        let reference = reference.as_primitive();

        Ok(match_each_integer_ptype!(array.ptype(), |P| {
            encoded_pvalue
                .typed_value::<P>()
                .map(|v| {
                    v.wrapping_add(
                        reference
                            .typed_value::<P>()
                            .vortex_expect("FoRArray Reference value cannot be null"),
                    )
                })
                .map(|v| Scalar::primitive::<P>(v, array.reference_scalar().dtype().nullability()))
                .unwrap_or_else(|| Scalar::null(array.reference_scalar().dtype().clone()))
        }))
    }
}

#[cfg(test)]
mod test {
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;

    use crate::FoRData;

    #[test]
    fn for_scalar_at() {
        let for_arr = FoRData::encode(PrimitiveArray::from_iter([-100, 1100, 1500, 1900])).unwrap();
        let expected = PrimitiveArray::from_iter([-100, 1100, 1500, 1900]);
        assert_arrays_eq!(for_arr, expected);
    }
}
