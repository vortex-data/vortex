// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::Nullability;
use vortex_error::VortexResult;
use vortex_scalar::Scalar;

use super::DictVTable;
use crate::Array;
use crate::IntoArray;
use crate::arrays::dict::DictArray;
use crate::compute::fill_null;
use crate::validity::Validity;
use crate::vtable::ValidityVTable;

impl ValidityVTable<DictVTable> for DictVTable {
    fn validity(array: &DictArray) -> VortexResult<Validity> {
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
                    Validity::Array(
                        unsafe { DictArray::new_unchecked(array.codes().clone(), values_validity) }
                            .into_array(),
                    )
                }
                (Validity::Array(_codes_validity), Validity::Array(values_validity)) => {
                    // Create a mask representing "is the value at codes[i] valid?"
                    let values_valid_mask =
                        unsafe { DictArray::new_unchecked(array.codes().clone(), values_validity) }
                            .into_array();
                    let values_valid_mask = fill_null(
                        &values_valid_mask,
                        &Scalar::bool(false, Nullability::NonNullable),
                    )?;

                    Validity::Array(values_valid_mask)
                }
            },
        )
    }
}
