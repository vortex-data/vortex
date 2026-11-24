// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::dict::DictArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;

use crate::list_contains::ListContains;
use crate::list_contains::ListContainsElementReduce;
use crate::list_contains::ListContainsOptions;
use crate::list_contains::PreparedSet;

/// Pushes `list_contains` of a prepared set into the values of a dictionary of needles.
impl ListContainsElementReduce for Dict {
    fn list_contains(
        list: &ArrayRef,
        needles: ArrayView<'_, Dict>,
        options: &ListContainsOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(set) = PreparedSet::set_of(list) else {
            return Ok(None);
        };
        let dtype = DType::Bool(options.result_nullability(list.dtype(), needles.dtype()));

        // Off SQL null semantics an empty list answers `false` for every needle, a null one
        // included, which the dictionary would answer `null` for from a null code.
        if set.is_empty() && !options.sql_null_semantics {
            return Ok(Some(
                ConstantArray::new(Scalar::bool(false, dtype.nullability()), needles.len())
                    .into_array(),
            ));
        }

        // A sliced dictionary can have more values than code rows. Do not increase the work in
        // that case.
        let values = needles.values();
        let codes = needles.codes();
        if values.len() > codes.len() {
            return Ok(None);
        }

        // A null code answers `null`, as a null needle does against a list with elements.
        // The optimizer reduces this lazy node when it visits the returned dictionary.
        let values =
            ListContains::try_new_opts(list.slice(0..values.len())?, values.clone(), *options)?
                .into_array();

        // SAFETY: the codes are unchanged, and the values have one result for each value.
        let dict = unsafe {
            DictArray::new_unchecked(codes.clone(), values)
                .set_all_values_referenced(needles.has_all_values_referenced())
        }
        .into_array();

        if dict.dtype() == &dtype {
            Ok(Some(dict))
        } else {
            dict.cast(dtype).map(Some)
        }
    }
}
