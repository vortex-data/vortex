// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A writer strategy for extension-typed arrays.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::layouts::extension::ExtensionLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Writes extension-typed arrays into an [`ExtensionLayout`] over a layout of their storage.
///
/// Every chunk is replaced by its storage, and the storage stream is written through `storage`.
/// Like [`StructStrategy`](crate::layouts::struct_::StructStrategy), it is a structural writer
/// that does not decide which extension columns to decompose. That is the job of
/// [`TableStrategy`](crate::layouts::table::TableStrategy).
pub struct ExtensionStrategy {
    storage: Arc<dyn LayoutStrategy>,
}

impl ExtensionStrategy {
    /// Create a writer that writes the extension storage through `storage`.
    pub fn new(storage: Arc<dyn LayoutStrategy>) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl LayoutStrategy for ExtensionStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let ext_dtype = stream
            .dtype()
            .as_extension_opt()
            .ok_or_else(|| {
                vortex_err!(
                    "ExtensionStrategy can only write extension-typed streams, got {}",
                    stream.dtype()
                )
            })?
            .clone();
        let storage_dtype = ext_dtype.storage_dtype().clone();

        let storage_session = session.clone();
        let storage_stream = stream.map(move |chunk| {
            let (sequence_id, chunk) = chunk?;
            Ok((sequence_id, extension_storage(chunk, &storage_session)?))
        });
        let storage = self
            .storage
            .write_stream(
                ctx,
                segment_sink,
                SequentialStreamAdapter::new(storage_dtype, storage_stream).sendable(),
                eof,
                session,
            )
            .await?;
        Ok(ExtensionLayout::try_new(ext_dtype, storage)?.into_layout())
    }
}

/// Returns the storage of an extension-typed `chunk`, whatever its encoding.
///
/// A constant chunk keeps a constant storage. Any other chunk that is not already an extension
/// array is executed into one.
fn extension_storage(chunk: ArrayRef, session: &VortexSession) -> VortexResult<ArrayRef> {
    if let Some(scalar) = chunk.as_constant() {
        let storage = scalar.as_extension().to_storage_scalar();
        return Ok(ConstantArray::new(storage, chunk.len()).into_array());
    }
    let extension = chunk.execute::<ExtensionArray>(&mut session.create_execution_ctx())?;
    Ok(extension.storage_array().clone())
}
