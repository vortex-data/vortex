// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RecordBatchReader;
use arrow_array::ffi_stream;
use arrow_schema::SchemaRef;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::iter::ArrayIterator;
use vortex_error::VortexError;
use vortex_error::VortexResult;

use crate::ArrowSession;

/// An adapter for converting an `ArrowArrayStreamReader` into a Vortex `ArrayIterator`.
///
/// The stream's schema and batches are imported through the [`ArrowSession`], so Arrow extension
/// types are routed to their registered import plugins instead of being flattened into their
/// storage types.
pub struct ArrowArrayStreamAdapter {
    stream: ffi_stream::ArrowArrayStreamReader,
    session: ArrowSession,
    schema: SchemaRef,
    dtype: DType,
}

impl ArrowArrayStreamAdapter {
    /// Adapt `stream`, importing its schema and each of its batches through `session`.
    ///
    /// The adapter holds a clone of `session`, which shares the plugin registries with it, so
    /// plugins registered after construction are still observed.
    ///
    /// The schema declared by the stream is the authoritative schema for the import: every batch
    /// is converted against it, so extension metadata survives even if an individual batch has
    /// lost it.
    pub fn try_new(
        session: &ArrowSession,
        stream: ffi_stream::ArrowArrayStreamReader,
    ) -> VortexResult<Self> {
        let schema = stream.schema();
        let dtype = session.from_arrow_schema(schema.as_ref())?;
        Ok(Self {
            stream,
            session: session.clone(),
            schema,
            dtype,
        })
    }
}

impl ArrayIterator for ArrowArrayStreamAdapter {
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl Iterator for ArrowArrayStreamAdapter {
    type Item = VortexResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.stream.next()?;

        Some(batch.map_err(VortexError::from).and_then(|b| {
            self.session
                .from_arrow_record_batch(b, self.schema.as_ref())
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow_array::FixedSizeBinaryArray;
    use arrow_array::RecordBatch;
    use arrow_array::RecordBatchIterator;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use arrow_schema::extension::Uuid as ArrowUuid;
    use vortex_array::array_session;
    use vortex_array::arrays::Struct;
    use vortex_array::arrays::struct_::StructArrayExt;
    use vortex_array::extension::uuid::Uuid;
    use vortex_error::VortexExpect;

    use super::*;
    use crate::ArrowSessionExt;

    /// The adapter imports through the [`ArrowSession`], so a UUID column arrives as a Vortex
    /// extension array rather than its `FixedSizeList` storage.
    #[test]
    fn stream_preserves_extension_types() -> VortexResult<()> {
        let mut field = Field::new("id", DataType::FixedSizeBinary(16), false);
        field.try_with_extension_type(ArrowUuid)?;
        let schema = Arc::new(Schema::new(vec![field]));
        let ids = FixedSizeBinaryArray::try_from_iter(
            [*b"0123456789abcdef", *b"fedcba9876543210"].into_iter(),
        )?;
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(ids)])?;

        let reader = RecordBatchIterator::new([Ok(batch)], schema);
        let stream = ffi_stream::ArrowArrayStreamReader::try_new(
            ffi_stream::FFI_ArrowArrayStream::new(Box::new(reader)),
        )?;

        let vortex_session = array_session();
        let mut adapter = ArrowArrayStreamAdapter::try_new(&vortex_session.arrow(), stream)?;

        let DType::Struct(fields, _) = adapter.dtype().clone() else {
            panic!("expected a struct dtype, got {}", adapter.dtype());
        };
        assert!(
            fields
                .field_by_index(0)
                .vortex_expect("one field")
                .as_extension()
                .is::<Uuid>()
        );

        let array = adapter.next().vortex_expect("one batch")?;
        assert_eq!(array.dtype(), adapter.dtype());
        assert!(
            array
                .as_::<Struct>()
                .unmasked_field(0)
                .dtype()
                .as_extension()
                .is::<Uuid>()
        );
        assert!(adapter.next().is_none());

        Ok(())
    }
}
