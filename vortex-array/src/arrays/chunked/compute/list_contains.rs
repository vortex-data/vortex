// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Chunked;
use crate::arrays::ChunkedArray;
use crate::arrays::chunked::ChunkedArrayExt;
use crate::dtype::DType;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fn::fns::list_contains::ListContains;
use crate::scalar_fn::fns::list_contains::ListContainsElementReduce;
use crate::scalar_fn::fns::list_contains::ListContainsOptions;
use crate::scalar_fn::fns::list_contains::PreparedSet;

/// Pushes `list_contains` of a prepared set into each chunk of the needles.
///
/// Each chunk gets a slice of the prepared set, which shares the one set, so that the set is built
/// once for every chunk.
impl ListContainsElementReduce for Chunked {
    fn list_contains(
        list: &ArrayRef,
        needles: ArrayView<'_, Chunked>,
        options: &ListContainsOptions,
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
                ListContains::try_new_opts(set, chunk.clone(), *options)?
                    .into_array()
                    .optimize()
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
