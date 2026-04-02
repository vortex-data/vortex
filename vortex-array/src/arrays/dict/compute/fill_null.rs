// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::Dict;
use super::DictArray;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::builtins::ArrayBuiltins;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;
use crate::scalar_fn::fns::fill_null::FillNullKernel;
use crate::scalar_fn::fns::operators::Operator;

impl FillNullKernel for Dict {
    fn fill_null(
        array: ArrayView<'_, Dict>,
        fill_value: &Scalar,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        // If the fill value already exists in the dictionary, we can simply rewrite the null codes
        // to point to the value.
        let found_fill_values = array
            .values()
            .clone()
            .binary(
                ConstantArray::new(fill_value.clone(), array.values().len()).into_array(),
                Operator::Eq,
            )?
            .execute::<BoolArray>(ctx)?;

        // We found the fill value already in the values at this given index.
        let Some(existing_fill_value_index) =
            found_fill_values.to_bit_buffer().set_indices().next()
        else {
            // No fill values found, so we must canonicalize and fill_null.
            return Ok(Some(
                array
                    .array()
                    .clone()
                    .execute::<Canonical>(ctx)?
                    .into_array()
                    .fill_null(fill_value.clone())?,
            ));
        };

        // Now we rewrite the nullable codes to point at the fill value.
        let codes = array.codes();

        // Cast the index to the correct unsigned integer type matching the codes' ptype.
        let codes_ptype = codes.dtype().as_ptype();

        #[expect(
            clippy::cast_possible_truncation,
            reason = "The existing index must be representable by the existing ptype"
        )]
        let fill_scalar_value = match_each_integer_ptype!(codes_ptype, |P| {
            ScalarValue::from(existing_fill_value_index as P)
        });

        // Fill nulls in both the codes and the values. Note that the precondition of this function
        // states that the fill value is non-null, so we do not have to worry about the nullability.
        let codes = codes.clone().fill_null(Scalar::try_new(
            codes.dtype().as_nonnullable(),
            Some(fill_scalar_value),
        )?)?;
        let values = array.values().clone().fill_null(fill_value.clone())?;

        // SAFETY: invariants are still satisfied after patching nulls.
        unsafe {
            Ok(Some(
                DictArray::new_unchecked(codes, values)
                    .set_all_values_referenced(array.has_all_values_referenced())
                    .into_array(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;

    use crate::IntoArray;
    use crate::ToCanonical;
    use crate::arrays::DictArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;
    use crate::dtype::Nullability;
    use crate::scalar::Scalar;
    use crate::validity::Validity;

    #[test]
    fn nullable_codes_fill_in_values() {
        let dict = DictArray::try_new(
            PrimitiveArray::new(
                buffer![0u32, 1, 2],
                Validity::from(BitBuffer::from(vec![true, false, true])),
            )
            .into_array(),
            PrimitiveArray::new(buffer![10, 20, 20], Validity::AllValid).into_array(),
        )
        .vortex_expect("operation should succeed in test");

        let filled = dict
            .into_array()
            .fill_null(Scalar::primitive(20, Nullability::NonNullable))
            .vortex_expect("operation should succeed in test");
        let filled_primitive = filled.to_primitive();
        assert_arrays_eq!(filled_primitive, PrimitiveArray::from_iter([10, 20, 20]));
        assert!(filled_primitive.all_valid().unwrap());
    }
}
