// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::Dict;
use super::DictArray;
use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::ConstantArray;
use crate::arrays::scalar_fn::ScalarFnArrayExt;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::fns::like::Like;
use crate::scalar_fn::fns::like::LikeOptions;
use crate::scalar_fn::fns::like::LikeReduce;

impl LikeReduce for Dict {
    fn like(
        array: ArrayView<'_, Dict>,
        pattern: &ArrayRef,
        options: LikeOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        // If we have more values than codes, it is faster to canonicalize first.
        if array.values().len() > array.codes().len() {
            return Ok(None);
        }
        if let Some(pattern) = pattern.as_constant() {
            let pattern = ConstantArray::new(pattern, array.values().len()).into_array();

            let values = Like
                .try_new_array(pattern.len(), options, [array.values().clone(), pattern])?
                .optimize()?;

            // SAFETY: LIKE preserves the len of the values, so codes are still pointing at
            //  valid positions.
            // Preserve all_values_referenced since codes are unchanged.
            unsafe {
                Ok(Some(
                    DictArray::new_unchecked(array.codes().clone(), values)
                        .set_all_values_referenced(array.has_all_values_referenced())
                        .into_array(),
                ))
            }
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::IntoArray;
    use crate::arrays::BoolArray;
    use crate::arrays::DictArray;
    use crate::arrays::VarBinArray;
    use crate::arrays::dict::compute::like::ConstantArray;
    use crate::arrays::scalar_fn::ScalarFnArrayExt;
    use crate::assert_arrays_eq;
    use crate::optimizer::ArrayOptimizer;
    use crate::scalar_fn::fns::like::Like;
    use crate::scalar_fn::fns::like::LikeOptions;

    #[test]
    fn like_reduce_dict() -> VortexResult<()> {
        let dict = DictArray::try_new(
            buffer![0u8, 1, 0, 2].into_array(),
            VarBinArray::from(vec!["hello", "world", "help"]).into_array(),
        )?
        .into_array();

        let pattern = ConstantArray::new("hello%", 4).into_array();
        let result = Like
            .try_new_array(4, LikeOptions::default(), [dict, pattern])?
            .optimize()?;

        assert_arrays_eq!(result, BoolArray::from_iter([true, false, true, false]));
        Ok(())
    }
}
