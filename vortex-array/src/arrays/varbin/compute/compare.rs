// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::BinaryArray;
use arrow_array::StringArray;
use arrow_ord::cmp;
use itertools::Itertools;
use vortex_buffer::BitBuffer;
use vortex_dtype::DType;
use vortex_dtype::IntegerPType;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::ToCanonical;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinVTable;
use crate::arrow::Datum;
use crate::arrow::from_arrow_array_with_len;
use crate::compute::CompareKernel;
use crate::compute::CompareKernelAdapter;
use crate::compute::Operator;
use crate::compute::compare;
use crate::compute::compare_lengths_to_empty;
use crate::register_kernel;
use crate::vtable::ValidityHelper;

// This implementation exists so we can have custom translation of RHS to arrow that's not the same as IntoCanonical
impl CompareKernel for VarBinVTable {
    fn compare(
        &self,
        lhs: &VarBinArray,
        rhs: &dyn Array,
        operator: Operator,
    ) -> VortexResult<Option<ArrayRef>> {
        if let Some(rhs_const) = rhs.as_constant() {
            let nullable = lhs.dtype().is_nullable() || rhs_const.dtype().is_nullable();
            let len = lhs.len();

            let rhs_is_empty = match rhs_const.dtype() {
                DType::Binary(_) => rhs_const
                    .as_binary()
                    .is_empty()
                    .vortex_expect("RHS should not be null"),
                DType::Utf8(_) => rhs_const
                    .as_utf8()
                    .is_empty()
                    .vortex_expect("RHS should not be null"),
                _ => vortex_bail!("VarBinArray can only have type of Binary or Utf8"),
            };

            if rhs_is_empty {
                let buffer = match operator {
                    Operator::Gte => BitBuffer::new_set(len), // Every possible value is >= ""
                    Operator::Lt => BitBuffer::new_unset(len), // No value is < ""
                    Operator::Eq | Operator::NotEq | Operator::Gt | Operator::Lte => {
                        let lhs_offsets = lhs.offsets().to_primitive();
                        match_each_integer_ptype!(lhs_offsets.ptype(), |P| {
                            compare_offsets_to_empty::<P>(lhs_offsets, operator)
                        })
                    }
                };

                return Ok(Some(
                    BoolArray::from_bit_buffer(
                        buffer,
                        lhs.validity()
                            .clone()
                            .union_nullability(rhs.dtype().nullability()),
                    )
                    .into_array(),
                ));
            }

            let lhs = Datum::try_new(lhs.as_ref())?;

            // Use StringViewArray/BinaryViewArray to match the Utf8View/BinaryView types
            // produced by Datum::try_new (which uses into_arrow_preferred())
            let arrow_rhs: &dyn arrow_array::Datum = match rhs_const.dtype() {
                DType::Utf8(_) => &rhs_const
                    .as_utf8()
                    .value()
                    .map(StringArray::new_scalar)
                    .unwrap_or_else(|| arrow_array::Scalar::new(StringArray::new_null(1))),
                DType::Binary(_) => &rhs_const
                    .as_binary()
                    .value()
                    .map(BinaryArray::new_scalar)
                    .unwrap_or_else(|| arrow_array::Scalar::new(BinaryArray::new_null(1))),
                _ => vortex_bail!(
                    "VarBin array RHS can only be Utf8 or Binary, given {}",
                    rhs_const.dtype()
                ),
            };

            let array = match operator {
                Operator::Eq => cmp::eq(&lhs, arrow_rhs),
                Operator::NotEq => cmp::neq(&lhs, arrow_rhs),
                Operator::Gt => cmp::gt(&lhs, arrow_rhs),
                Operator::Gte => cmp::gt_eq(&lhs, arrow_rhs),
                Operator::Lt => cmp::lt(&lhs, arrow_rhs),
                Operator::Lte => cmp::lt_eq(&lhs, arrow_rhs),
            }
            .map_err(|err| vortex_err!("Failed to compare VarBin array: {}", err))?;

            Ok(Some(from_arrow_array_with_len(&array, len, nullable)))
        } else if !rhs.is::<VarBinVTable>() {
            // NOTE: If the rhs is not a VarBin array it will be canonicalized to a VarBinView
            // Arrow doesn't support comparing VarBin to VarBinView arrays, so we convert ourselves
            // to VarBinView and re-invoke.
            return Ok(Some(compare(lhs.to_varbinview().as_ref(), rhs, operator)?));
        } else {
            Ok(None)
        }
    }
}

register_kernel!(CompareKernelAdapter(VarBinVTable).lift());

fn compare_offsets_to_empty<P: IntegerPType>(
    offsets: PrimitiveArray,
    operator: Operator,
) -> BitBuffer {
    let lengths_iter = offsets
        .as_slice::<P>()
        .iter()
        .tuple_windows()
        .map(|(&s, &e)| e - s);
    compare_lengths_to_empty(lengths_iter, operator)
}

#[cfg(test)]
mod test {
    use vortex_buffer::BitBuffer;
    use vortex_buffer::ByteBuffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::ToCanonical;
    use crate::arrays::ConstantArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::compute::Operator;
    use crate::compute::compare;

    #[test]
    fn test_binary_compare() {
        let array = VarBinArray::from_iter(
            [Some(b"abc".to_vec()), None, Some(b"def".to_vec())],
            DType::Binary(Nullability::Nullable),
        );
        let result = compare(
            array.as_ref(),
            ConstantArray::new(
                Scalar::binary(ByteBuffer::copy_from(b"abc"), Nullability::Nullable),
                3,
            )
            .as_ref(),
            Operator::Eq,
        )
        .unwrap()
        .to_bool();

        assert_eq!(
            &result.validity_mask().to_bit_buffer(),
            &BitBuffer::from_iter([true, false, true])
        );
        assert_eq!(
            result.bit_buffer(),
            &BitBuffer::from_iter([true, false, false])
        );
    }

    #[test]
    fn varbinview_compare() {
        let array = VarBinArray::from_iter(
            [Some(b"abc".to_vec()), None, Some(b"def".to_vec())],
            DType::Binary(Nullability::Nullable),
        );
        let vbv = VarBinViewArray::from_iter(
            [None, None, Some(b"def".to_vec())],
            DType::Binary(Nullability::Nullable),
        );
        let result = compare(array.as_ref(), vbv.as_ref(), Operator::Eq)
            .unwrap()
            .to_bool();

        assert_eq!(
            &result.validity_mask().to_bit_buffer(),
            &BitBuffer::from_iter([false, false, true])
        );
        assert_eq!(
            result.bit_buffer(),
            &BitBuffer::from_iter([false, true, true])
        );
    }
}

#[cfg(test)]
mod tests {
    use vortex_dtype::DType;
    use vortex_dtype::Nullability;
    use vortex_scalar::Scalar;

    use crate::Array;
    use crate::arrays::ConstantArray;
    use crate::arrays::VarBinArray;
    use crate::compute::Operator;
    use crate::compute::compare;

    #[test]
    fn test_null_compare() {
        let arr = VarBinArray::from_iter([Some("h")], DType::Utf8(Nullability::NonNullable));

        let const_ = ConstantArray::new(Scalar::utf8("", Nullability::Nullable), 1);

        assert_eq!(
            compare(arr.as_ref(), const_.as_ref(), Operator::Eq)
                .unwrap()
                .dtype(),
            &DType::Bool(Nullability::Nullable)
        );
    }
}
