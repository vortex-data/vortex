// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::Dict;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::dict::DictData;
use crate::builtins::ArrayBuiltins;
use crate::dtype::Nullability;
use crate::scalar::Scalar;
use crate::validity::Validity;

impl ValidityVTable<Dict> for Dict {
    fn validity(array: ArrayView<'_, Dict>) -> VortexResult<Validity> {
        Ok(
            match (array.codes().validity()?, array.values().validity()?) {
                (
                    Validity::NonNullable | Validity::AllValid,
                    Validity::NonNullable | Validity::AllValid,
                ) => {
                    // Recall that we know the dictionary is nullable if we're in this function.
                    Validity::AllValid
                }
                (Validity::AllInvalid, _) | (_, Validity::AllInvalid) => Validity::AllInvalid,
                (Validity::Array(codes_validity), Validity::NonNullable | Validity::AllValid) => {
                    Validity::Array(codes_validity)
                }
                (Validity::AllValid | Validity::NonNullable, Validity::Array(values_validity)) => {
                    // We know codes are all valid, so the cast is free.
                    let codes = array.codes().cast(array.codes().dtype().as_nonnullable())?;
                    Validity::Array(
                        unsafe { DictData::new_unchecked(codes, values_validity) }.into_array(),
                    )
                }
                (Validity::Array(_codes_validity), Validity::Array(values_validity)) => {
                    // Create a mask representing "is the value at codes[i] valid?"
                    let values_valid_mask =
                        unsafe { DictData::new_unchecked(array.codes().clone(), values_validity) }
                            .into_array();
                    let values_valid_mask = values_valid_mask
                        .fill_null(Scalar::bool(false, Nullability::NonNullable))?;

                    Validity::Array(values_valid_mask)
                }
            },
        )
    }
}
