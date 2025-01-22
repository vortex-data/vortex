use arrow_buffer::BooleanBuffer;
use vortex_dtype::{match_each_native_ptype, NativePType};
use vortex_error::{VortexExpect, VortexResult};

use crate::array::{BoolArray, PrimitiveArray, PrimitiveEncoding};
use crate::compute::{compare, CompareFn, FilterMask, Operator, SelectionArray};
use crate::validity::Validity;
use crate::variants::PrimitiveArrayTrait;
use crate::{ArrayData, IntoArrayData};

const SELECTIVITIY_THRESHOLD: f64 = 0.2;

impl CompareFn<PrimitiveArray> for PrimitiveEncoding {
    fn compare(
        &self,
        _lhs: &PrimitiveArray,
        _rhs: &ArrayData,
        _operator: Operator,
    ) -> VortexResult<Option<ArrayData>> {
        // Explicitly fallback to arrow
        Ok(None)
    }

    fn compare_with_selection(
        &self,
        lhs: &PrimitiveArray,
        rhs: &ArrayData,
        operator: Operator,
        selection: &FilterMask,
    ) -> VortexResult<Option<ArrayData>> {
        if let Some(rhs_scalar) = rhs.as_constant() {
            if selection.selectivity() <= SELECTIVITIY_THRESHOLD {
                // If selectivity is low, save work by iterating the indices and applying the compare
                // to the selected elements only.
                match_each_native_ptype!(lhs.ptype(), |$T| {
                    let compare_fn: fn($T, $T) -> bool = match operator {
                        Operator::Eq => |lhs, rhs| lhs == rhs,
                        Operator::NotEq => |lhs, rhs| lhs != rhs,
                        Operator::Lt => |lhs, rhs| lhs < rhs,
                        Operator::Lte => |lhs, rhs| lhs <= rhs,
                        Operator::Gt => |lhs, rhs| lhs > rhs,
                        Operator::Gte => |lhs, rhs| lhs >= rhs,
                    };

                    // let num: $T = rhs.cast(&lhs.dtype())?.into();
                    let num: $T = rhs_scalar.as_primitive().typed_value::<$T>().vortex_expect("primitive scalar");
                    compare_primitive_constant_selection::<$T>(lhs.as_slice::<$T>(), num, compare_fn, lhs.validity(), selection)
                })
            } else {
                // Apply the comparison to the whole array, preserving the mask.
                let result = compare(lhs.as_ref(), rhs, operator)?;
                Ok(Some(
                    SelectionArray::new(result, selection.clone()).into_array(),
                ))
            }
        } else {
            Ok(None)
        }
    }
}

fn compare_primitive_constant_selection<T: NativePType>(
    lhs: &[T],
    rhs: T,
    cmp_fn: fn(T, T) -> bool,
    validity: Validity,
    selection: &FilterMask,
) -> VortexResult<Option<ArrayData>> {
    // iterate the mask
    let bools = selection
        .indices()
        .iter()
        .map(|idx| cmp_fn(lhs[*idx], rhs))
        .collect::<BooleanBuffer>();

    Ok(Some(
        BoolArray::try_new(bools, validity.filter(selection)?)?.into_array(),
    ))
}
