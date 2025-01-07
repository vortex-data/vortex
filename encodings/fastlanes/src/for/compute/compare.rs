use num_traits::{CheckedShr, WrappingSub};
use vortex_array::array::ConstantArray;
use vortex_array::compute::{compare, CompareFn, Operator};
use vortex_array::{ArrayDType, ArrayData, ArrayLen, IntoArrayData};
use vortex_dtype::{match_each_integer_ptype, NativePType};
use vortex_error::{vortex_err, VortexError, VortexResult};
use vortex_scalar::{PValue, PrimitiveScalar, Scalar};

use crate::{FoRArray, FoREncoding};

impl CompareFn<FoRArray> for FoREncoding {
    fn compare(
        &self,
        lhs: &FoRArray,
        rhs: &ArrayData,
        operator: Operator,
    ) -> VortexResult<Option<ArrayData>> {
        if let Some(constant) = rhs.as_constant() {
            if let Ok(constant) = PrimitiveScalar::try_from(&constant) {
                match_each_integer_ptype!(constant.ptype(), |$T| {
                    return compare_constant(lhs, constant.typed_value::<$T>(), operator);
                })
            }
        }

        Ok(None)
    }
}

fn compare_constant<T>(
    lhs: &FoRArray,
    rhs: Option<T>,
    operator: Operator,
) -> VortexResult<Option<ArrayData>>
where
    T: NativePType + WrappingSub + CheckedShr,
    T: TryFrom<PValue, Error = VortexError>,
    Scalar: From<Option<T>>,
{
    // For now, we only support equals and not equals. Comparisons are a little more fiddly to
    // get right regarding how to handle overflow and the wrapping subtraction.
    if !matches!(operator, Operator::Eq | Operator::NotEq) {
        return Ok(None);
    }

    let reference = lhs.reference_scalar();
    let reference = reference.as_primitive().typed_value::<T>();

    // We encode the RHS into the FoR domain.
    let rhs = rhs
        .map(|mut rhs| {
            if let Some(reference) = reference {
                rhs = rhs.wrapping_sub(&reference);
            }
            if lhs.shift() > 0 {
                rhs = rhs
                    .checked_shr(lhs.shift() as u32)
                    .ok_or_else(|| vortex_err!("Shift overflow"))?;
            }
            Ok::<_, VortexError>(rhs)
        })
        .transpose()?;

    // Wrap up the RHS into a scalar and cast to the encoded DType (this will be the equivalent
    // unsigned integer type).
    let rhs = Scalar::from(rhs).cast(lhs.encoded().dtype())?;

    compare(
        lhs.encoded(),
        ConstantArray::new(rhs, lhs.len()).into_array(),
        operator,
    )
    .map(Some)
}

#[cfg(test)]
mod tests {
    use arrow_buffer::BooleanBuffer;
    use vortex_array::array::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_array::IntoCanonical;
    use vortex_buffer::buffer;

    use super::*;

    #[test]
    fn test_compare_constant() {
        let reference = Scalar::from(10);
        // 10, 30, 12
        let lhs = FoRArray::try_new(
            PrimitiveArray::new(buffer!(0u32, 10, 1), Validity::AllValid).into_array(),
            reference,
            1,
        )
        .unwrap();

        assert_result(
            compare_constant(&lhs, Some(30i32), Operator::Eq),
            [false, true, false],
        );
        assert_result(
            compare_constant(&lhs, Some(12i32), Operator::NotEq),
            [true, true, false],
        );
        for op in [Operator::Lt, Operator::Lte, Operator::Gt, Operator::Gte] {
            assert!(compare_constant(&lhs, Some(30i32), op).unwrap().is_none());
        }
    }

    fn assert_result<T: IntoIterator<Item = bool>>(
        result: VortexResult<Option<ArrayData>>,
        expected: T,
    ) {
        let result = result
            .unwrap()
            .unwrap()
            .into_canonical()
            .unwrap()
            .into_bool()
            .unwrap();
        assert_eq!(result.boolean_buffer(), BooleanBuffer::from_iter(expected));
    }
}
