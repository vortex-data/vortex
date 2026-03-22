// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;

use futures::future::BoxFuture;
use moka::future::FutureExt;
use vortex_array::ArrayRef;
use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::layouts::SharedArrayFuture;
use crate::v2::reader::MaskStreamRef;
use crate::v2::reader::Reader;
use crate::v2::reader::ReaderStream;
use crate::v2::reader::ReaderStreamRef;

pub struct FlatReader {
    len: usize,
    dtype: DType,
    array_fut: SharedArrayFuture,
}

impl Reader for FlatReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.len as u64
    }

    fn project(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
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
            dtype: self.dtype.clone(),
            array_fut,
            offset: start,
            remaining: end - start,
        }))
    }

    fn filter(&self, _row_range: Range<u64>) -> VortexResult<MaskStreamRef> {
        todo!("FlatReader::filter")
    }
}

struct FlatLayoutReaderStream {
    dtype: DType,
    array_fut: SharedArrayFuture,
    offset: usize,
    remaining: usize,
}

impl ReaderStream for FlatLayoutReaderStream {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn next_chunk_len(&self) -> Option<usize> {
        if self.remaining == 0 {
            None
        } else {
            Some(self.remaining)
        }
    }

    fn next_chunk(
        &mut self,
        mask: &MaskFuture,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        let mask_len = mask.len();
        if mask_len > self.remaining {
            vortex_bail!(
                "Mask length {} exceeds remaining rows {}",
                mask_len,
                self.remaining
            );
        }

        let array_fut = self.array_fut.clone();
        let offset = self.offset;
        let mask = mask.clone();

        self.offset += mask_len;
        self.remaining -= mask_len;

        Ok(async move {
            let array = array_fut.await?;
            let sliced_array = array.slice(offset..offset + mask.len())?;
            let selection = mask.await?;
            let selected_array = sliced_array.filter(selection)?;
            Ok(selected_array)
        }
        .boxed())
    }
}
