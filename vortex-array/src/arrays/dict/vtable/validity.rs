// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_dtype::Nullability;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use super::DictVTable;
use crate::Array;
use crate::IntoArray;
use crate::ToCanonical;
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

    fn validity_mask(array: &DictArray) -> Mask {
        let codes_validity = array.codes().validity_mask();
        match codes_validity.bit_buffer() {
            AllOr::All => {
                let primitive_codes = array.codes().to_primitive();
                let values_mask = array.values().validity_mask();
                let is_valid_buffer = match_each_integer_ptype!(primitive_codes.ptype(), |P| {
                    let codes_slice = primitive_codes.as_slice::<P>();
                    BitBuffer::collect_bool(array.len(), |idx| {
                        #[allow(clippy::cast_possible_truncation)]
                        values_mask.value(codes_slice[idx] as usize)
                    })
                });
                Mask::from_buffer(is_valid_buffer)
            }
            AllOr::None => Mask::AllFalse(array.len()),
            AllOr::Some(validity_buff) => {
                let primitive_codes = array.codes().to_primitive();
                let values_mask = array.values().validity_mask();
                let is_valid_buffer = match_each_integer_ptype!(primitive_codes.ptype(), |P| {
                    let codes_slice = primitive_codes.as_slice::<P>();
                    #[allow(clippy::cast_possible_truncation)]
                    BitBuffer::collect_bool(array.len(), |idx| {
                        validity_buff.value(idx) && values_mask.value(codes_slice[idx] as usize)
                    })
                });
                Mask::from_buffer(is_valid_buffer)
            }
        }
    }
}
