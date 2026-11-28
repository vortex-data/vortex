// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Writes shredded Variant chunks using the ordinary struct writer for their columns.

use std::sync::Arc;

use async_channel::bounded;
use async_trait::async_trait;
use futures::TryStreamExt;
use futures::future::try_join_all;
use futures::stream;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VariantArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::variant::VariantArraySlotsExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_functions::variant_get::VariantGet;
use vortex_functions::variant_get::VariantGetOptions;
use vortex_functions::variant_get::VariantPath;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_set::HashSet;

use super::CORE;
use super::SHREDDED;
use super::Variant;
use super::VariantLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::VTable;
use crate::layouts::chunked::ChunkedLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Writes a shredded Variant stream as the struct of its core storage and its shredded child.
///
/// Consecutive chunks with the same shredding schema and completeness share a writer.
/// Unshredded runs and writes whose editions exclude the Variant layout use the fallback.
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
        mut stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        if !stream.dtype().is_variant() {
            vortex_bail!(
                "VariantStrategy can only write Variant streams, got {}",
                stream.dtype()
            );
        }
        if !ctx.allows_layout(&Variant.id()) {
            return self
                .fallback
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }
        let dtype = stream.dtype().clone();
        let mut writers = Vec::new();
        let mut active = None;
        let mut all_sent = true;
        while let Some((id, chunk)) = stream.try_next().await? {
            let (shredding, array) = storage_chunk(chunk, &mut session.create_execution_ctx())?;
            let sender = match &active {
                Some((previous, sender)) if previous == &shredding => sender,
                _ => {
                    let (sender, receiver) = bounded(1);
                    let input =
                        SequentialStreamAdapter::new(array.dtype().clone(), receiver).sendable();
                    let writer = Arc::clone(if shredding.is_some() {
                        &self.storage
                    } else {
                        &self.fallback
                    });
                    let ctx = ctx.clone();
                    let sink = Arc::clone(&segment_sink);
                    let run_eof = eof.split_off();
                    let session = session.clone();
                    let dtype = dtype.clone();
                    let schema = shredding.clone();
                    // A run may write statistics at EOF. Keep it running while later runs consume
                    // their input sequence IDs so an earlier writer can finish.
                    writers.push(session.handle().spawn_nested(move |_| async move {
                        let layout = writer
                            .write_stream(ctx, sink, input, run_eof, &session)
                            .await?;
                        match schema {
                            Some(Shredding {
                                dtype: shredded,
                                complete,
                            }) => Ok(VariantLayout::from_storage(
                                dtype, shredded, &layout, complete,
                            )?
                            .into_layout()),
                            None => VortexResult::Ok(layout),
                        }
                    }));
                    &active.insert((shredding, sender)).1
                }
            };
            if sender.send(Ok((id, array))).await.is_err() {
                all_sent = false;
                break;
            }
        }
        drop(active);
        drop(stream);
        if writers.is_empty() {
            let empty = SequentialStreamAdapter::new(dtype, stream::empty()).sendable();
            return self
                .fallback
                .write_stream(ctx, segment_sink, empty, eof, session)
                .await;
        }
        let layouts = try_join_all(writers).await?;
        // Preserve a child writer's error if it closed its receiver early.
        vortex_ensure!(
            all_sent,
            "Variant writer finished before all chunks were sent"
        );
        Ok(ChunkedLayout::from_chunks(dtype, layouts))
    }
}

#[derive(Clone, PartialEq, Eq)]
struct Shredding {
    dtype: DType,
    complete: bool,
}

fn storage_chunk(
    chunk: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Option<Shredding>, ArrayRef)> {
    let variant = chunk.clone().execute::<VariantArray>(ctx)?;
    let Some(shredded) = variant.shredded() else {
        return Ok((None, chunk));
    };
    let core = variant.core_storage();
    let complete = is_complete(core, shredded, ctx)?;
    let shredded = if complete && core.dtype().is_nullable() {
        shredded
            .clone()
            .mask(core.validity()?.to_array(core.len()))?
    } else {
        shredded.clone()
    };
    let schema = Shredding {
        dtype: shredded.dtype().clone(),
        complete,
    };
    let storage = StructArray::from_fields(&[(CORE, core.clone()), (SHREDDED, shredded)])?;
    Ok((Some(schema), storage.into_array()))
}

/// Prove that a direct typed read cannot miss a residual value. Nested containers may
/// have residual children even when the container itself is valid.
fn is_complete(core: &ArrayRef, shredded: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    let valid = shredded.validity()?.execute_mask(shredded.len(), ctx)?;
    // The root path exists on every SQL-valid row, including one containing Variant null.
    if !valid.all_true() {
        let core_valid = core.validity()?.execute_mask(core.len(), ctx)?;
        if !core_valid.bitand_not(&valid).all_false() {
            return Ok(false);
        }
    }
    if shredded.dtype().is_struct() {
        let object = shredded.clone().execute::<StructArray>(ctx)?;
        let mut nested = Vec::new();
        for (name, field) in object.names().iter().zip(object.iter_unmasked_fields()) {
            if matches!(
                field.dtype(),
                DType::Struct(..) | DType::List(..) | DType::FixedSizeList(..)
            ) {
                nested.push(name);
                continue;
            }
            let present = field.validity()?.execute_mask(field.len(), ctx)?;
            let missing = valid.clone().bitand_not(&present);
            if !missing.all_false() && !residual_absent(&core.filter(missing)?, name, ctx)? {
                return Ok(false);
            }
        }
        return match nested.as_slice() {
            [] => Ok(true),
            [name] => residual_absent(core, name, ctx),
            names => nested_residuals_absent(core, names, ctx),
        };
    }
    Ok(!matches!(
        shredded.dtype(),
        DType::List(..) | DType::FixedSizeList(..)
    ))
}

fn residual_absent(
    core: &ArrayRef,
    name: &FieldName,
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let values = VariantGet::try_new(
        core.clone(),
        VariantGetOptions::new(VariantPath::field(name.clone()), None),
    )?;
    Ok(values
        .validity()?
        .execute_mask(values.len(), ctx)?
        .all_false())
}

/// Inspect each residual object once when several shredded containers could overlap it.
fn nested_residuals_absent(
    core: &ArrayRef,
    names: &[&FieldName],
    ctx: &mut ExecutionCtx,
) -> VortexResult<bool> {
    let names: HashSet<&str> = names.iter().map(|name| name.as_ref()).collect();
    let mut probe = core.repeated_probe();
    for row in 0..core.len() {
        let scalar = probe.execute_scalar(row, ctx)?;
        let mut payload = Some(&scalar);
        while let Some(variant) = payload.and_then(|s| s.as_variant_opt()) {
            payload = variant.value();
        }
        if let Some(object) = payload.and_then(|s| s.as_struct_opt())
            && !object.is_null()
            && object
                .names()
                .iter()
                .any(|name| names.contains(name.as_ref()))
        {
            return Ok(false);
        }
    }
    Ok(true)
}
