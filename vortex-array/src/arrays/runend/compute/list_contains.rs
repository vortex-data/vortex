// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::IntoArray;
use crate::arrays::ScalarFnArray;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::array::RunEndArrayExt;
use crate::arrays::runend::array::RunEndArraySlotsExt;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::list_contains::ListContains;
use crate::scalar_fn::fns::list_contains::ListContainsElementReduce;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;
use crate::scalar_fn::fns::list_contains::PreparedSet;

/// Pushes `list_contains` of a prepared set into the values of run-end encoded needles.
impl ListContainsElementReduce for RunEnd {
    fn list_contains(
        list: &ArrayRef,
        needles: ArrayView<'_, RunEnd>,
        options: &ListContainsOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(set) = PreparedSet::set_of(list) else {
            return Ok(None);
        };

        // The values get the shared set at their own length, which can exceed the rows of a
        // zero-length slice that kept its runs.
        let values = needles.values();
        let set = ScalarFnArray::try_new_with_len(PreparedSet.bind(set.clone()), [], values.len())?
            .into_array();
        let values = ListContains::try_new_opts(set, values.clone(), *options)?.into_array();

        // SAFETY: the ends are unchanged, and the values have one result for each run.
        Ok(Some(
            unsafe {
                RunEnd::new_unchecked(
                    needles.ends().clone(),
                    values,
                    needles.offset(),
                    needles.len(),
                )
            }
            .into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_buffer::buffer;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::BoolArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::runend::RunEnd;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::optimizer::ArrayOptimizer;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::list_contains::ListContains;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn constant_list_is_looked_up_once_per_run() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let needles = RunEnd::try_new(
            buffer![2u32, 5, 6].into_array(),
            buffer![1i32, 2, 3].into_array(),
            &mut ctx,
        )?
        .into_array();
        let element = DType::Primitive(PType::I32, Nullability::NonNullable);
        let list = Scalar::list(
            element,
            vec![
                Scalar::primitive(2i32, Nullability::NonNullable),
                Scalar::primitive(3i32, Nullability::NonNullable),
            ],
            Nullability::NonNullable,
        );
        let list = ConstantArray::new(list, needles.len()).into_array();

        let array = ListContains::try_new(list, needles)?
            .into_array()
            .optimize()?;
        assert!(array.is::<RunEnd>());

        assert_arrays_eq!(
            array,
            BoolArray::from_iter([false, false, true, true, true, true]),
            &mut ctx
        );
        Ok(())
    }
}
