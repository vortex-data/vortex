// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use futures::future::BoxFuture;
use moka::future::FutureExt;
use vortex_array::ArrayRef;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::layouts::SharedArrayFuture;
use crate::v2::reader::LayoutReader2;
use crate::v2::reader::LayoutReader2Ref;
use crate::v2::stream::LayoutReaderStream;
use crate::v2::stream::SendableLayoutReaderStream;

pub struct FlatReader2 {
    len: usize,
    dtype: DType,
    array_fut: SharedArrayFuture,
}

impl LayoutReader2 for FlatReader2 {
    fn row_count(&self) -> u64 {
        self.len as u64
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn nchildren(&self) -> usize {
        0
    }

    fn child(&self, _idx: usize) -> &LayoutReader2Ref {
        unreachable!()
    }

    fn execute(&self, row_range: Range<u64>) -> VortexResult<SendableLayoutReaderStream> {
        // We need to share the same array future
        let array_fut = self.array_fut.clone();

        let start = usize::try_from(row_range.start).map_err(|_| {
            vortex_err!("Row range start {} is too large for usize", row_range.start)
        })?;
        let end = usize::try_from(row_range.end)
            .map_err(|_| vortex_err!("Row range end {} is too large for usize", row_range.end))?;

        if start > self.len || end > self.len || start > end {
            vortex_bail!(
                "Row range {:?} is out of bounds for array of length {}",
                row_range,
                self.len
            );
        }

        Ok(Box::new(FlatLayoutReaderStream {
            array_fut,
            offset: start,
            remaining: end - start,
        }))
    }
}

struct FlatLayoutReaderStream {
    array_fut: SharedArrayFuture,
    offset: usize,
    remaining: usize,
}

impl LayoutReaderStream for FlatLayoutReaderStream {
    fn next_chunk_len(&self) -> Option<usize> {
        if self.remaining == 0 {
            None
        } else {
            Some(self.remaining)
        }
    }

    fn next_chunk(
        &mut self,
        selection: &Mask,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        if selection.len() > self.remaining {
            vortex_bail!(
                "Selection mask length {} exceeds remaining rows {}",
                selection.len(),
                self.remaining
            );
        }

        let array_fut = self.array_fut.clone();
        let offset = self.offset;
        let selection = selection.clone();

        self.offset += selection.len();
        self.remaining -= selection.len();

        Ok(async move {
            let array = array_fut.await?;
            let sliced_array = array.slice(offset..offset + selection.len());
            let selected_array = sliced_array.filter(selection.clone())?;
            Ok(selected_array)
        }
        .boxed())
    }
}
