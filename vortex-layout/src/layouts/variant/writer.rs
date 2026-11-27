// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Writes a shredded Variant stream as a struct layout of the Variant.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::variant::VariantArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_functions::variant_get::VariantGet;
use vortex_functions::variant_get::VariantGetOptions;
use vortex_functions::variant_get::VariantPath;
use vortex_session::VortexSession;

use super::CORE;
use super::SHREDDED;
use super::layout_id;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
use crate::layouts::struct_::StructLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Writes a shredded Variant stream as the struct of its core storage and its shredded child.
///
/// Each chunk keeps its own shredding schema. Unshredded chunks and writes whose editions
/// exclude the Variant layout use the fallback strategy.
///
/// **Note**: this writes an unstable and experimental layout that is expected to change. Using
/// it may lead to unreadable files in the future.
#[derive(Clone)]
pub struct VariantStrategy {
    /// Writes the struct of the core storage and the shredded child.
    storage: Arc<dyn LayoutStrategy>,
    /// Writes a Variant stream that is not shredded.
    fallback: Arc<dyn LayoutStrategy>,
}

impl VariantStrategy {
    /// Creates a Variant writer that writes the struct of a shredded Variant through `storage`,
    /// and a Variant that is not shredded through `fallback`.
    pub fn new(storage: Arc<dyn LayoutStrategy>, fallback: Arc<dyn LayoutStrategy>) -> Self {
        Self { storage, fallback }
    }
}

#[async_trait]
impl LayoutStrategy for VariantStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        if !stream.dtype().is_variant() {
            vortex_bail!(
                "VariantStrategy can only write Variant streams, got {}",
                stream.dtype()
            );
        }
        if !ctx.allows_layout(&layout_id()) {
            return self
                .fallback
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }
        ChunkedLayoutStrategy::new(VariantChunk(self.clone()))
            .write_stream(ctx, segment_sink, stream, eof, session)
            .await
    }
}

/// Adapts a single Variant chunk to the ordinary struct writer.
struct VariantChunk(VariantStrategy);

#[async_trait]
impl LayoutStrategy for VariantChunk {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        mut stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let (id, chunk) = stream
            .next()
            .await
            .vortex_expect("chunked writer supplies one chunk")?;
        let dtype = chunk.dtype().clone();
        let mut exec = session.create_execution_ctx();
        let variant = chunk.clone().execute::<VariantArray>(&mut exec)?;
        let Some(shredded) = variant.shredded() else {
            let stream = SequentialStreamAdapter::new(dtype, stream::iter([Ok((id, chunk))]));
            return self
                .0
                .fallback
                .write_stream(ctx, segment_sink, stream.sendable(), eof, session)
                .await;
        };
        let core = variant.core_storage();
        let complete = is_complete(core, shredded, &mut exec)?;
        let storage =
            StructArray::from_fields(&[(CORE, core.clone()), (SHREDDED, shredded.clone())])?
                .into_array();
        let stream = SequentialStreamAdapter::new(
            storage.dtype().clone(),
            stream::iter([Ok((id, storage))]),
        );
        let layout = self
            .0
            .storage
            .write_stream(ctx, segment_sink, stream.sendable(), eof, session)
            .await?;
        Ok(
            StructLayout::try_new_variant(dtype, shredded.dtype().clone(), &layout, complete)?
                .into_layout(),
        )
    }
}

/// Prove that a direct typed read cannot miss a residual value. Nested containers may
/// have residual children even when the container itself is valid.
fn is_complete(core: &ArrayRef, shredded: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    let valid = shredded.validity()?.execute_mask(shredded.len(), ctx)?;
    // Any residual on a missing root could supply the requested value, including an
    // object whose fields are all missing. Avoid a speculative cast to the shredded dtype.
    if !valid.all_true() && !residual_absent(&core.filter(!&valid)?, VariantPath::root(), ctx)? {
        return Ok(false);
    }
    if shredded.dtype().is_struct() {
        let object = shredded.clone().execute::<StructArray>(ctx)?;
        for (name, field) in object.names().iter().zip(object.iter_unmasked_fields()) {
            let present = field.validity()?.execute_mask(field.len(), ctx)?;
            let covered = &valid & &present;
            let probe = match field.dtype() {
                DType::Struct(..) | DType::List(..) | DType::FixedSizeList(..) => core.clone(),
                _ if covered.all_true() => continue,
                _ => core.filter(!&covered)?,
            };
            if !residual_absent(&probe, VariantPath::field(name.clone()), ctx)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }
    Ok(!matches!(
        shredded.dtype(),
        DType::List(..) | DType::FixedSizeList(..)
    ))
}

fn residual_absent(
    core: &ArrayRef,
    path: VariantPath,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let values = VariantGet::try_new(core.clone(), VariantGetOptions::new(path, None))?
        .into_array()
        .execute::<ArrayRef>(ctx)?;
    Ok(values
        .validity()?
        .execute_mask(values.len(), ctx)?
        .all_false())
}
