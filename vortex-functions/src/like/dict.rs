// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::dict::DictArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_error::VortexResult;

use super::Dict;
use crate::like::Like;
use crate::like::LikeOptions;
use crate::like::LikeReduce;

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

            let values = Like::try_new(array.values().clone(), pattern, options)?.into_array();

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
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::assert_arrays_eq;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::like::Like;
    use crate::like::LikeOptions;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();

        crate::initialize(&session);

        session
    });

    #[test]
    fn like_reduce_dict() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let dict = DictArray::try_new(
            buffer![0u8, 1, 0, 2].into_array(),
            VarBinArray::from(vec!["hello", "world", "help"]).into_array(),
        )?
        .into_array();

        let pattern = ConstantArray::new("hello%", 4).into_array();
        let result = Like::try_new(dict, pattern, LikeOptions::default())?.into_array();

        assert_arrays_eq!(
            result,
            BoolArray::from_iter([true, false, true, false]),
            &mut ctx
        );
        Ok(())
    }
}
