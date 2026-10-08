// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::arrays::slice::SliceReduce;
use vortex_error::VortexResult;

use crate::ALP;
use crate::ALPArrayExt;
use crate::ALPArraySlotsExt;

impl SliceReduce for ALP {
    fn slice(array: ArrayView<'_, Self>, range: Range<usize>) -> VortexResult<Option<ArrayRef>> {
        let patches = match array.patches() {
            None => None,
            Some(patches) if patches.is_cheap_to_slice() => patches.slice(range.clone())?,
            // Slicing these patches would read their buffers.
            Some(_) => return Ok(None),
        };
        Ok(Some(
            ALP::new(array.encoded().slice(range)?, array.exponents(), patches).into_array(),
        ))
    }
}

impl SliceKernel for ALP {
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let sliced_alp = ALP::new(
            array.encoded().slice(range.clone())?,
            array.exponents(),
            array
                .patches()
                .map(|p| p.slice(range))
                .transpose()?
                .flatten(),
        )
        .into_array();
        Ok(Some(sliced_alp))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::ALP;
    use crate::ALPArrayExt;
    use crate::alp_encode;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = vortex_array::array_session();
        crate::initialize(&session);
        session
    });

    #[test]
    fn slice_reduces_with_host_patches() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_iter((0..4096).map(|i| {
            if i % 97 == 0 {
                f64::NAN
            } else {
                f64::from(i) / 4.0
            }
        }));
        let alp = alp_encode(values.as_view(), None, &mut ctx)?;
        assert!(alp.patches().is_some(), "test setup expects patches");

        let sliced = alp.into_array().slice(700..3500)?;

        assert!(sliced.is::<ALP>());
        assert_arrays_eq!(sliced, values.into_array().slice(700..3500)?, &mut ctx);
        Ok(())
    }
}
