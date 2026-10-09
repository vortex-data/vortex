// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::IntoArray;
use vortex_array::scalar_fn::fns::list_contains::ListContains;
use vortex_array::scalar_fn::fns::list_contains::ListContainsElementReduce;
use vortex_array::scalar_fn::fns::list_contains::ListContainsOptions;
use vortex_array::scalar_fn::fns::list_contains::PreparedSet;
use vortex_array::scalar_fn::fns::list_contains::PreparedSetArray;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::array::RunEndArrayExt;
use crate::array::RunEndArraySlotsExt;

/// Pushes `list_contains` of a prepared set into the values of run-end encoded needles.
///
/// The values get the prepared set at their own length, sharing the one set, so that the set is
/// not built again for them.
impl ListContainsElementReduce for RunEnd {
    fn list_contains(
        list: &ArrayRef,
        needles: ArrayView<'_, RunEnd>,
        options: &ListContainsOptions,
    ) -> VortexResult<Option<ArrayRef>> {
        let Some(set) = list.as_opt::<PreparedSet>() else {
            return Ok(None);
        };

        // The values get the shared set at their own length, which can exceed the rows of a
        // zero-length slice that kept its runs.
        let values = needles.values();
        let set = PreparedSetArray::new(set.data().clone(), values.len()).into_array();
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
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::optimizer::ArrayOptimizer;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::fns::list_contains::ListContains;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::RunEnd;

    #[test]
    fn constant_list_is_probed_once_per_run() -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
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
