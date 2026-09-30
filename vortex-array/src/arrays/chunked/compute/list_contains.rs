// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::dtype::DType;
use crate::scalar_fn::fns::list_contains::ListContains;
use crate::scalar_fn::fns::list_contains::ListContainsElementKernel;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;
use crate::scalar_fn::fns::list_contains::PreparedSet;

/// Probes each chunk of the needles against one prepared set.
///
/// A constant list is prepared first, by the execution of [`ListContains`]. Each chunk then gets a
/// lazy [`ListContains`] over a slice of the prepared set, which shares the one probe, so that the
/// kernels of the chunk's own encoding can probe it.
impl ListContainsElementKernel for Chunked {
    fn list_contains(
        list: &ArrayRef,
        needles: ArrayView<'_, Chunked>,
        options: &ListContainsOptions,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        if !list.is::<PreparedSet>() {
            return Ok(None);
        }

        let mut offset = 0;
        let chunks = needles
            .iter_chunks()
            .map(|chunk| {
                let set = list.slice(offset..offset + chunk.len())?;
                offset += chunk.len();
                Ok(ListContains::try_new_opts(set, chunk.clone(), *options)?.into_array())
            })
            .collect::<VortexResult<Vec<_>>>()?;

        let dtype = DType::Bool(options.result_nullability(list.dtype(), needles.dtype()));

        // SAFETY: each chunk is `list_contains` of the prepared set and a needle chunk of one dtype,
        // so every chunk has the dtype of the whole result.
        Ok(Some(
            unsafe { ChunkedArray::new_unchecked(chunks, dtype) }.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::array_session;
    use crate::arrays::BoolArray;
    use crate::arrays::Chunked;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::PrimitiveArray;
    use crate::assert_arrays_eq;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::optimizer::ArrayOptimizer;
    use crate::scalar::Scalar;
    use crate::scalar_fn::fns::list_contains::ListContains;
    use crate::scalar_fn::fns::list_contains::ListContainsOptions;

    /// The set `{2, null}` of nullable `i32`.
    fn set_with_null() -> Scalar {
        let element = DType::Primitive(PType::I32, Nullability::Nullable);
        Scalar::list(
            element.clone(),
            vec![
                Scalar::primitive(2i32, Nullability::Nullable),
                Scalar::null(element),
            ],
            Nullability::NonNullable,
        )
    }

    fn chunked_needles() -> VortexResult<ArrayRef> {
        Ok(ChunkedArray::try_new(
            vec![
                PrimitiveArray::from_option_iter([Some(1i32), Some(2)]).into_array(),
                PrimitiveArray::from_option_iter::<i32, _>([]).into_array(),
                PrimitiveArray::from_option_iter([None, Some(3), Some(2)]).into_array(),
            ],
            DType::Primitive(PType::I32, Nullability::Nullable),
        )?
        .into_array())
    }

    #[rstest]
    #[case::default(
        ListContainsOptions::default(),
        [Some(false), Some(true), None, Some(false), Some(true)],
    )]
    #[case::sql(
        ListContainsOptions { sql_null_semantics: true },
        [None, Some(true), None, None, Some(true)],
    )]
    fn test_constant_list_over_chunked_needles(
        #[case] options: ListContainsOptions,
        #[case] expected: [Option<bool>; 5],
    ) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let needles = chunked_needles()?;
        let list = ConstantArray::new(set_with_null(), needles.len()).into_array();

        // The node stays whole, so that execution prepares the list once for every chunk.
        let array = ListContains::try_new_opts(list, needles, options)?
            .into_array()
            .optimize()?;
        assert!(!array.is::<Chunked>());

        assert_arrays_eq!(array, BoolArray::from_iter(expected), &mut ctx);
        Ok(())
    }
}
