// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::Dict;
use crate::arrays::DictArray;
use crate::arrays::dict::DictArrayExt;
use crate::arrays::dict::DictArraySlotsExt;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::list_contains::ListContains;
use crate::scalar_fn::fns::list_contains::ListContainsElementReduce;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;
use crate::scalar_fn::fns::list_contains::PreparedSet;

/// Pushes `list_contains` of a prepared set into the values of a dictionary of needles.
///
/// The values get a slice of the prepared set, which shares the one set, so that the set is not
/// built again for them.
impl ListContainsElementReduce for Dict {
    fn list_contains(
        list: &ArrayRef,
        needles: ArrayView<'_, Dict>,
        options: &ListContainsOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(set) = list.as_opt::<PreparedSet>() else {
            return Ok(None);
        };
        let dtype = DType::Bool(options.result_nullability(list.dtype(), needles.dtype()));

        // Off SQL null semantics an empty list answers `false` for every needle, a null one
        // included, which the dictionary would answer `null` for from a null code.
        if set.data().is_empty() && !options.sql_null_semantics {
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
        let values =
            ListContains::try_new_opts(list.slice(0..values.len())?, values.clone(), *options)?
                .into_array()
                .optimize()?;

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
