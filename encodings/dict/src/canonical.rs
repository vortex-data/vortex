// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Not;

use vortex_array::arrays::{BoolArray, ConstantArray};
use vortex_array::compute::{Operator, cast, compare, mask, take};
use vortex_array::validity::Validity;
use vortex_array::vtable::CanonicalVTable;
use vortex_array::{Array, ArrayRef, Canonical, IntoArray, ToCanonical};
use vortex_buffer::BitBuffer;
use vortex_dtype::{DType, Nullability};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::{AllOr, Mask};
use vortex_scalar::Scalar;

use crate::{DictArray, DictVTable};

impl CanonicalVTable<DictVTable> for DictVTable {
    fn canonicalize(array: &DictArray) -> Canonical {
        match array.dtype() {
            // NOTE: Utf8 and Binary will decompress into VarBinViewArray, which requires a full
            // decompression to construct the views child array.
            // For this case, it is *always* faster to decompress the values first and then create
            // copies of the view pointers.
            DType::Utf8(_) | DType::Binary(_) => {
                let canonical_values: ArrayRef = array.values().to_canonical().into_array();
                take(&canonical_values, array.codes())
                    .vortex_expect("taking codes from dictionary values shouldn't fail")
                    .to_canonical()
            }
            DType::Bool(_) => {
                dict_bool_take(array).vortex_expect("Canonicalizing dict bool array shouldn't fail")
            }
            _ => take(array.values(), array.codes())
                .vortex_expect("taking codes from dictionary values shouldn't fail")
                .to_canonical(),
        }
    }
}

fn dict_bool_take(dict_array: &DictArray) -> VortexResult<Canonical> {
    let values = dict_array.values();
    let codes = dict_array.codes();
    let result_nullability = dict_array.dtype().nullability();

    let bool_values = values.to_bool();
    let result_validity = bool_values.validity_mask();
    let bool_buffer = bool_values.bit_buffer();
    let (first_match, second_match) = match result_validity.bit_buffer() {
        AllOr::All => {
            let mut indices_iter = bool_buffer.set_indices();
            (indices_iter.next(), indices_iter.next())
        }
        AllOr::None => (None, None),
        AllOr::Some(v) => {
            let mut indices_iter = bool_buffer.set_indices().filter(|i| v.value(*i));
            (indices_iter.next(), indices_iter.next())
        }
    };

    Ok(match (first_match, second_match) {
        // Couldn't find a value match, so the result is all false
        (None, _) => match result_validity {
            Mask::AllTrue(_) => BoolArray::from_bit_buffer(
                BitBuffer::new_unset(codes.len()),
                Validity::copy_from_array(codes).union_nullability(result_nullability),
            )
            .to_canonical(),
            Mask::AllFalse(_) => ConstantArray::new(
                Scalar::null(DType::Bool(Nullability::Nullable)),
                codes.len(),
            )
            .to_canonical(),
            Mask::Values(_) => BoolArray::from_bit_buffer(
                BitBuffer::new_unset(codes.len()),
                Validity::from_mask(result_validity, result_nullability).take(codes)?,
            )
            .to_canonical(),
        },
        // We found a single matching value so we can compare the codes directly.
        (Some(code), None) => match result_validity {
            Mask::AllTrue(_) => cast(
                &compare(
                    codes,
                    &cast(
                        ConstantArray::new(code, codes.len()).as_ref(),
                        codes.dtype(),
                    )?,
                    Operator::Eq,
                )?,
                &DType::Bool(result_nullability),
            )?
            .to_canonical(),
            Mask::AllFalse(_) => ConstantArray::new(
                Scalar::null(DType::Bool(Nullability::Nullable)),
                codes.len(),
            )
            .to_canonical(),
            Mask::Values(rv) => mask(
                &compare(
                    codes,
                    &cast(
                        ConstantArray::new(code, codes.len()).as_ref(),
                        codes.dtype(),
                    )?,
                    Operator::Eq,
                )?,
                &Mask::from_buffer(
                    take(BoolArray::from(rv.bit_buffer().clone()).as_ref(), codes)?
                        .to_bool()
                        .bit_buffer()
                        .not(),
                ),
            )?
            .to_canonical(),
        },
        // more than one value matches
        _ => take(bool_values.as_ref(), codes)
            .vortex_expect("taking codes from dictionary values shouldn't fail")
            .to_canonical(),
    })
}
