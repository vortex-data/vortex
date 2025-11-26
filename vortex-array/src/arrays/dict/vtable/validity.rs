// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::BitBuffer;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexExpect;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use super::DictVTable;
use crate::Array;
use crate::ToCanonical;
use crate::arrays::dict::DictArray;
use crate::vtable::ValidityVTable;

impl ValidityVTable<DictVTable> for DictVTable {
    fn is_valid(array: &DictArray, index: usize) -> bool {
        let scalar = array.codes().scalar_at(index);

        if scalar.is_null() {
            return false;
        };
        let values_index: usize = scalar
            .as_ref()
            .try_into()
            .vortex_expect("Failed to convert dictionary code to usize");
        array.values().is_valid(values_index)
    }

    fn all_valid(array: &DictArray) -> bool {
        array.codes().all_valid() && array.values().all_valid()
    }

    fn all_invalid(array: &DictArray) -> bool {
        array.codes().all_invalid() || array.values().all_invalid()
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
