// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use futures::future::try_join_all;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::MaskFuture;
use vortex_array::arrays::ScalarFnArray;
use vortex_array::dtype::DType;
use vortex_array::optimizer::ArrayOptimizer;
use vortex_array::scalar_fn::ScalarFnRef;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_error::VortexResult;

use crate::v2::reader::MaskStreamRef;
use crate::v2::reader::Reader;
use crate::v2::reader::ReaderRef;
use crate::v2::reader::ReaderStream;
use crate::v2::reader::ReaderStreamRef;

/// A [`Reader`] for applying a scalar function to another layout.
pub struct ScalarFnReader {
    scalar_fn: ScalarFnRef,
    dtype: DType,
    row_count: u64,
    children: Vec<ReaderRef>,
}

impl ScalarFnReader {
    pub fn try_new(
        scalar_fn: ScalarFnRef,
        children: Vec<ReaderRef>,
        row_count: u64,
    ) -> VortexResult<Self> {
        let dtype = scalar_fn.return_dtype(
            &children
                .iter()
                .map(|c| c.dtype().clone())
                .collect::<Vec<DType>>(),
        )?;

        Ok(Self {
            scalar_fn,
            dtype,
            row_count,
            children,
        })
    }

    pub fn scalar_fn(&self) -> &ScalarFnRef {
        &self.scalar_fn
    }
}

impl Reader for ScalarFnReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

    fn project(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
        let input_streams = self
            .children
            .iter()
            .map(|child| child.project(row_range.clone()))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Box::new(ScalarFnArrayStream {
            dtype: self.dtype.clone(),
            scalar_fn: self.scalar_fn.clone(),
            input_streams,
        }))
    }

    fn filter(&self, _row_range: Range<u64>) -> VortexResult<MaskStreamRef> {
        todo!("ScalarFnReader::filter")
    }
}

struct ScalarFnArrayStream {
    dtype: DType,
    scalar_fn: ScalarFnRef,
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
        mask: &MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        let scalar_fn = self.scalar_fn.clone();
        let futs = self
            .input_streams
            .iter_mut()
            .map(|s| s.next_chunk(mask))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Box::pin(async move {
            let input_arrays = try_join_all(futs).await?;
            let len = input_arrays.first().map_or(0, |a| a.len());
            let array = ScalarFnArray::try_new(scalar_fn, input_arrays, len)?.into_array();
            let array = array.optimize()?;
            Ok(array)
        }))
    }
}

pub trait ScalarFnReaderExt: ScalarFnVTable {
    /// Creates a [`ScalarFnReader`] applying this scalar function to the given children.
    fn new_reader(
        &self,
        options: Self::Options,
        children: Vec<ReaderRef>,
        row_count: u64,
    ) -> VortexResult<ReaderRef> {
        Ok(Arc::new(ScalarFnReader::try_new(
            self.bind(options),
            children,
            row_count,
        )?))
    }
}
impl<V: ScalarFnVTable> ScalarFnReaderExt for V {}
