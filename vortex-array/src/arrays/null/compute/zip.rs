// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::arrays::Null;
use crate::scalar_fn::fns::zip::ZipReduce;

impl ZipReduce for Null {
    fn zip(
        array: ArrayView<'_, Null>,
        _if_false: &ArrayRef,
        _mask: &ArrayRef,
    ) -> VortexResult<Option<ArrayRef>> {
        // Both branches have the null dtype, so every selected row is null.
        Ok(Some(array.array().clone()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Null;
    use crate::arrays::NullArray;
    use crate::assert_arrays_eq;
    use crate::builtins::ArrayBuiltins;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn zip_nulls() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let result = Mask::from_iter([true, false, true, false, false])
            .into_array()
            .zip(
                NullArray::new(5).into_array(),
                NullArray::new(5).into_array(),
            )?
            .execute::<ArrayRef>(&mut ctx)?;

        assert!(result.is::<Null>());
        assert_arrays_eq!(result, NullArray::new(5), &mut ctx);
        Ok(())
    }
}
