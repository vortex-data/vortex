// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::ops::Range;

use futures::future::BoxFuture;
use futures::future::try_join_all;
use moka::future::FutureExt;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::validity::Validity;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::v2::reader::Reader;
use crate::v2::reader::ReaderRef;
use crate::v2::reader::ReaderStream;
use crate::v2::reader::ReaderStreamRef;

pub struct StructReader {
    row_count: u64,
    dtype: DType,
    // TODO(ngates): we should make this lazy?
    fields: Vec<ReaderRef>,
}

impl Reader for StructReader {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn row_count(&self) -> u64 {
        self.row_count
    }

    fn execute(&self, row_range: Range<u64>) -> VortexResult<ReaderStreamRef> {
        let field_streams = self
            .fields
            .iter()
            .map(|field| field.execute(row_range.clone()))
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Box::new(StructReaderStream {
            dtype: self.dtype.clone(),
            fields: field_streams,
        }))
    }
}

struct StructReaderStream {
    dtype: DType,
    fields: Vec<ReaderStreamRef>,
}

impl ReaderStream for StructReaderStream {
    fn dtype(&self) -> &DType {
        &self.dtype
    }

    fn next_chunk_len(&self) -> Option<usize> {
        self.fields
            .iter()
            .map(|s| s.next_chunk_len())
            .min()
            .flatten()
    }

    fn next_chunk(
        &mut self,
        selection: &Mask,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ArrayRef>>> {
        let struct_fields = self.dtype.as_struct_fields().clone();
        let validity: Validity = self.dtype.nullability().into();
        let fields = self
            .fields
            .iter_mut()
            .map(|s| s.next_chunk(selection))
            .collect::<VortexResult<Vec<_>>>()?;
        let len = selection.true_count();

        Ok(async move {
            let fields = try_join_all(fields).await?;
            Ok(
                StructArray::try_new_with_dtype(fields, struct_fields, len, validity.clone())?
                    .into_array(),
            )
        }
        .boxed())
    }
}
