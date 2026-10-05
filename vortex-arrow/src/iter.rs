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
