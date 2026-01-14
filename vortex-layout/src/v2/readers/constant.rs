// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;

use futures::future::BoxFuture;
use moka::future::FutureExt;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::v2::reader::Reader;
use crate::v2::reader::ReaderStream;
use crate::v2::reader::ReaderStreamRef;

pub struct ConstantReader {
    scalar: Scalar,
    row_count: u64,
}

impl ConstantReader {
    pub fn new(scalar: Scalar, row_count: u64) -> Self {
        Self { scalar, row_count }
    }
}

impl Reader for ConstantReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        self.scalar.dtype()
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

    fn execute(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
        let remaining = row_range.end.saturating_sub(row_range.start);
        Ok(Box::new(ConstantReaderStream {
            scalar: self.scalar.clone(),
            remaining,
        }))
    }
}

struct ConstantReaderStream {
    scalar: Scalar,
    remaining: u64,
}

impl ReaderStream for ConstantReaderStream {
    fn dtype(&self) -> &DType {
        self.scalar.dtype()
    }

    fn next_chunk_len(&self) -> Option<usize> {
        if self.remaining == 0 {
            None
        } else {
            Some(usize::try_from(self.remaining).unwrap_or(usize::MAX))
        }
    }

    fn next_chunk(
        &mut self,
        mask: &Mask,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        let array = ConstantArray::new(self.scalar.clone(), mask.true_count()).into_array();
        Ok(async move { Ok(array) }.boxed())
    }
}
