// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBin;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbin::VarBinArrayExt;
use crate::arrow_hooks::arrow_compute;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::IntegerPType;
use crate::match_each_integer_ptype;
use crate::scalar_fn::fns::binary::CompareKernel;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;

// This implementation exists so we can have custom translation of RHS to arrow that's not the same as IntoCanonical
impl CompareKernel for VarBin {
    fn compare(
        lhs: ArrayView<'_, VarBin>,
        rhs: &ArrayRef,
        operator: CompareOperator,
        ctx: &mut ExecutionCtx,
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
                    CompareOperator::Gte => BitBuffer::new_set(len), // Every possible value is >= ""
                    CompareOperator::Lt => BitBuffer::new_unset(len), // No value is < ""
                    CompareOperator::Eq | CompareOperator::Lte => {
                        let lhs_offsets = lhs.offsets().clone().execute::<PrimitiveArray>(ctx)?;
                        match_each_integer_ptype!(lhs_offsets.ptype(), |P| {
                            compare_offsets_to_empty::<P>(lhs_offsets, true)
                        })
                    }
                    CompareOperator::NotEq | CompareOperator::Gt => {
                        let lhs_offsets = lhs.offsets().clone().execute::<PrimitiveArray>(ctx)?;
                        match_each_integer_ptype!(lhs_offsets.ptype(), |P| {
                            compare_offsets_to_empty::<P>(lhs_offsets, false)
                        })
                    }
                };

                return Ok(Some(
                    BoolArray::new(
                        buffer,
                        lhs.validity()?.union_nullability(rhs.dtype().nullability()),
                    )
                    .into_array(),
                ));
            }

            match rhs_const.dtype() {
                DType::Utf8(_) | DType::Binary(_) => {}
                _ => vortex_bail!(
                    "VarBin array RHS can only be Utf8 or Binary, given {}",
                    rhs_const.dtype()
                ),
            }

            let _ = (len, nullable); // still useful context for future work
            Ok(Some((arrow_compute()?.varbin_compare_with_const)(
                lhs.array(),
                &rhs_const,
                operator,
            )?))
        } else if !rhs.is::<VarBin>() {
            // NOTE: If the rhs is not a VarBin array it will be canonicalized to a VarBinView
            // Arrow doesn't support comparing VarBin to VarBinView arrays, so we convert ourselves
            // to VarBinView and re-invoke.
            Ok(Some(
                lhs.array()
                    .clone()
                    .execute::<VarBinViewArray>(ctx)?
                    .into_array()
                    .binary(rhs.clone(), Operator::from(operator))?,
            ))
        } else {
            Ok(None)
        }
    }
}

fn compare_offsets_to_empty<P: IntegerPType>(offsets: PrimitiveArray, eq: bool) -> BitBuffer {
    let fn_ = if eq { P::eq } else { P::ne };
    let offsets = offsets.as_slice::<P>();
    BitBuffer::collect_bool(offsets.len() - 1, |idx| {
        let left = unsafe { offsets.get_unchecked(idx) };
        let right = unsafe { offsets.get_unchecked(idx + 1) };
        fn_(left, right)
    })
}

#[cfg(test)]
mod test {
    use vortex_buffer::BitBuffer;
    use vortex_buffer::ByteBuffer;

    use crate::IntoArray;
    use crate::LEGACY_SESSION;
    #[expect(deprecated)]
    use crate::ToCanonical as _;
    use crate::VortexSessionExecute;
    use crate::arrays::ConstantArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::VarBinViewArray;
    use crate::arrays::bool::BoolArrayExt;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::operators::Operator;

    #[test]
    fn test_binary_compare() {
        let array = VarBinArray::from_iter(
            [Some(b"abc".to_vec()), None, Some(b"def".to_vec())],
            DType::Binary(Nullability::Nullable),
        );
        #[expect(deprecated)]
        let result = array
            .into_array()
            .binary(
                ConstantArray::new(
                    Scalar::binary(ByteBuffer::copy_from(b"abc"), Nullability::Nullable),
                    3,
                )
                .into_array(),
                Operator::Eq,
            )
            .unwrap()
            .to_bool();

        assert_eq!(
            &result
                .as_ref()
                .validity()
                .unwrap()
                .to_mask(
                    result.as_ref().len(),
                    &mut LEGACY_SESSION.create_execution_ctx()
                )
                .unwrap()
                .to_bit_buffer(),
            &BitBuffer::from_iter([true, false, true])
        );
        assert_eq!(
            result.to_bit_buffer(),
            BitBuffer::from_iter([true, false, false])
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
        #[expect(deprecated)]
        let result = array
            .into_array()
            .binary(vbv.into_array(), Operator::Eq)
            .unwrap()
            .to_bool();

        assert_eq!(
            result
                .as_ref()
                .validity()
                .unwrap()
                .to_mask(
                    result.as_ref().len(),
                    &mut LEGACY_SESSION.create_execution_ctx()
                )
                .unwrap()
                .to_bit_buffer(),
            BitBuffer::from_iter([false, false, true])
        );
        assert_eq!(
            result.to_bit_buffer(),
            BitBuffer::from_iter([false, true, true])
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::IntoArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::VarBinArray;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::operators::Operator;

    #[test]
    fn test_null_compare() {
        let arr = VarBinArray::from_iter([Some("h")], DType::Utf8(Nullability::NonNullable));

        let const_ = ConstantArray::new(Scalar::utf8("", Nullability::Nullable), 1);

        assert_eq!(
            arr.into_array()
                .binary(const_.into_array(), Operator::Eq)
                .unwrap()
                .dtype(),
            &DType::Bool(Nullability::Nullable)
        );
    }
}
