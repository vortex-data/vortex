// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use futures::future::BoxFuture;
use futures::future::try_join_all;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::expr::ScalarFn;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::v2::reader::Reader;
use crate::v2::reader::ReaderRef;
use crate::v2::reader::ReaderStream;
use crate::v2::reader::ReaderStreamRef;

/// A [`Reader] for applying a scalar function to another layout.
pub struct ScalarFnReader {
    scalar_fn: ScalarFn,
    dtype: DType,
    row_count: u64,
    children: Vec<ReaderRef>,
}

impl Reader for ScalarFnReader {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

    fn execute(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
        let input_streams = self
            .children
            .iter()
            .map(|child| child.execute(row_range.clone()))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Box::new(ScalarFnArrayStream {
            dtype: self.dtype.clone(),
            scalar_fn: self.scalar_fn.clone(),
            input_streams,
        }))
    }
}

struct ScalarFnArrayStream {
    dtype: DType,
    scalar_fn: ScalarFn,
    input_streams: Vec<ReaderStreamRef>,
}

impl ReaderStream for ScalarFnArrayStream {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn next_chunk_len(&self) -> Option<usize> {
        self.input_streams
            .iter()
            .map(|s| s.next_chunk_len())
            .min()
            .flatten()
    }

    fn next_chunk(
        &mut self,
        selection: &Mask,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        let scalar_fn = self.scalar_fn.clone();
        let len = selection.true_count();
        let futs = self
            .input_streams
            .iter_mut()
            .map(|s| s.next_chunk(selection))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Box::pin(async move {
            let input_arrays = try_join_all(futs).await?;
            let array = ScalarFnArray::try_new(scalar_fn, input_arrays, len)?.into_array();
            let array = array.optimize()?;
            Ok(array)
        }))
    }
}
