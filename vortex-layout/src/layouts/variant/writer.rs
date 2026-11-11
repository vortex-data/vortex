// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Writes a shredded Variant stream as a [`VariantLayout`].

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use futures::StreamExt;
use futures::future;
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
use vortex_array::dtype::FieldNames;
use vortex_array::scalar_fn::fns::variant_get::VariantGet;
use vortex_array::scalar_fn::fns::variant_get::VariantGetOptions;
use vortex_array::scalar_fn::fns::variant_get::VariantPath;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use super::CORE;
use super::SHREDDED;
use super::VALIDITY;
use super::VariantLayout;
use super::splits_validity;
use super::storage_dtype;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Shreds every chunk of one Variant stream alike, keeping the values each chunk holds.
pub type VariantChunkShredder =
    Arc<dyn Fn(ArrayRef, &mut ExecutionCtx) -> VortexResult<ArrayRef> + Send + Sync>;

/// Chooses how to shred a Variant stream that is not shredded, from its first chunk. Returns
/// `None` to leave the stream as it is.
pub type VariantShredder = Arc<
    dyn Fn(&ArrayRef, &mut ExecutionCtx) -> VortexResult<Option<VariantChunkShredder>>
        + Send
        + Sync,
>;

/// Writes a shredded Variant stream as the struct of its core storage and its shredded child.
///
/// A stream whose first chunk is not shredded is shredded by the [`VariantShredder`], if there is
/// one and it finds something to shred. Otherwise the stream has nothing to split, and is
/// written by the fallback strategy as it is. Every chunk of a shredded stream must be shredded
/// alike.
///
/// **Note**: this writes an unstable and experimental layout that is expected to change. Using
/// it may lead to unreadable files in the future.
#[derive(Clone)]
pub struct VariantStrategy {
    /// Writes the struct of the core storage and the shredded child.
    storage: Arc<dyn LayoutStrategy>,
    /// Writes a Variant stream that is not shredded.
    fallback: Arc<dyn LayoutStrategy>,
    /// Shreds a stream that is not shredded.
    shredder: Option<VariantShredder>,
}

impl VariantStrategy {
    /// Creates a Variant writer that writes the struct of a shredded Variant through `storage`,
    /// and a Variant that is not shredded through `fallback`.
    pub fn new(storage: Arc<dyn LayoutStrategy>, fallback: Arc<dyn LayoutStrategy>) -> Self {
        Self {
            storage,
            fallback,
            shredder: None,
        }
    }

    /// Shreds a stream that is not shredded with `shredder`, instead of writing it through the
    /// fallback.
    pub fn with_shredder(mut self, shredder: VariantShredder) -> Self {
        self.shredder = Some(shredder);
        self
    }
}

#[async_trait]
impl LayoutStrategy for VariantStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        mut stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        if !dtype.is_variant() {
            vortex_bail!("VariantStrategy can only write Variant streams, got {dtype}");
        }

        // The dtype of a Variant does not say how it is shredded, so the first chunk decides.
        let first = stream.next().await.transpose()?;
        let shredded_dtype = |chunk: &ArrayRef| -> VortexResult<Option<DType>> {
            let variant = chunk
                .clone()
                .execute::<VariantArray>(&mut session.create_execution_ctx())?;
            Ok(variant.shredded().map(|shredded| shredded.dtype().clone()))
        };
        let shredder = match (&first, &self.shredder) {
            (Some((_, chunk)), Some(shredder)) if shredded_dtype(chunk)?.is_none() => {
                shredder(chunk, &mut session.create_execution_ctx())?
            }
            _ => None,
        };
        let shred = {
            let session = session.clone();
            move |chunk: VortexResult<(SequenceId, ArrayRef)>| {
                let Some(shredder) = &shredder else {
                    return chunk;
                };
                let (sequence_id, chunk) = chunk?;
                Ok((
                    sequence_id,
                    shredder(chunk, &mut session.create_execution_ctx())?,
                ))
            }
        };
        let first = first.map(|chunk| shred(Ok(chunk))).transpose()?;
        let shredded_dtype = first
            .as_ref()
            .map(|(_, chunk)| shredded_dtype(chunk))
            .transpose()?
            .flatten();
        let stream = stream::iter(first.map(Ok)).chain(stream.map(shred));

        let Some(shredded_dtype) = shredded_dtype else {
            let stream = SequentialStreamAdapter::new(dtype, stream).sendable();
            return self
                .fallback
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        };

        let complete = Arc::new(AtomicBool::new(true));
        let stream = {
            let complete = Arc::clone(&complete);
            let shredded_dtype = shredded_dtype.clone();
            let session = session.clone();
            stream.then(move |chunk| {
                let result = chunk.and_then(|(sequence_id, chunk)| {
                    let mut ctx = session.create_execution_ctx();
                    let (storage, chunk_complete) = split(chunk, &shredded_dtype, &mut ctx)?;
                    if !chunk_complete {
                        complete.store(false, Ordering::Relaxed);
                    }
                    Ok((sequence_id, storage))
                });
                future::ready(result)
            })
        };
        let storage_dtype = storage_dtype(&dtype, shredded_dtype.clone());
        let storage = self
            .storage
            .write_stream(
                ctx,
                segment_sink,
                SequentialStreamAdapter::new(storage_dtype, stream).sendable(),
                eof,
                session,
            )
            .await?;

        // The stream is exhausted, so every chunk has reported whether it is complete.
        Ok(VariantLayout::try_new(
            dtype,
            shredded_dtype,
            storage,
            complete.load(Ordering::Relaxed),
        )?
        .into_layout())
    }
}

/// Splits a Variant `chunk` into the struct of its core storage and its shredded child, of
/// `shredded_dtype`, and reports whether the shredded child is complete.
fn split(
    chunk: ArrayRef,
    shredded_dtype: &DType,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(ArrayRef, bool)> {
    let len = chunk.len();
    let variant = chunk.execute::<VariantArray>(ctx)?;
    let shredded = variant
        .shredded()
        .ok_or_else(|| vortex_err!("only some chunks of a Variant stream are shredded"))?
        .clone();
    let complete = is_complete(variant.core_storage(), &shredded, ctx)?;

    let shredded = if shredded.dtype() == shredded_dtype {
        shredded
    } else if shredded.dtype().eq_ignore_nullability(shredded_dtype) {
        shredded.cast(shredded_dtype.clone())?
    } else {
        vortex_bail!(
            "chunks of a Variant stream are shredded as {} and as {shredded_dtype}",
            shredded.dtype()
        );
    };
    // The core storage is written in its own encoding, not as a canonical Variant around it:
    // the writers below slice their input, and a slice reaches the storage encoding directly.
    let mut names = vec![CORE, SHREDDED];
    let mut fields = vec![variant.core_storage().clone()];
    if splits_validity(shredded_dtype) {
        let shredded = shredded.execute::<StructArray>(ctx)?;
        let validity = shredded.validity()?.execute_mask(len, ctx)?;
        fields.push(
            StructArray::try_new(
                shredded.names().clone(),
                shredded.iter_unmasked_fields().cloned().collect::<Vec<_>>(),
                len,
                Validity::NonNullable,
            )?
            .into_array(),
        );
        names.push(VALIDITY);
        fields.push(validity.into_array());
    } else {
        fields.push(shredded);
    }
    let storage =
        StructArray::try_new(FieldNames::from(names), fields, len, Validity::NonNullable)?;
    Ok((storage.into_array(), complete))
}

/// Whether `shredded` holds every value it could hold, so that `core` has no value for a path
/// that `shredded` represents; see
/// [`VariantPackOptions::complete`](vortex_array::scalar_fn::fns::variant_pack::VariantPackOptions::complete).
///
/// Only the rows where `shredded` is null can have left their value in `core`, so only those
/// rows of `core` are read.
fn is_complete(core: &ArrayRef, shredded: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<bool> {
    let valid = shredded.validity()?.execute_mask(shredded.len(), ctx)?;
    if valid.all_true() {
        return Ok(true);
    }
    let core = core.filter(!&valid)?;
    let mut has_no_value = |path: VariantPath, dtype: Option<DType>| -> VortexResult<bool> {
        let values = VariantGet::try_new(core.clone(), VariantGetOptions::new(path, dtype))?
            .into_array()
            .execute::<ArrayRef>(ctx)?;
        Ok(values
            .validity()?
            .execute_mask(values.len(), ctx)?
            .all_false())
    };

    match shredded.dtype() {
        // A row that is not in a shredded struct must not be an object with a shredded field.
        DType::Struct(fields, _) => {
            for name in fields.names().iter() {
                if !has_no_value(VariantPath::field(name.clone()), None)? {
                    return Ok(false);
                }
            }
            // The struct is stored without its validity, so a field read must find a null in
            // the rows where the struct was.
            let fields = shredded.clone().execute::<StructArray>(ctx)?;
            for field in fields.iter_unmasked_fields() {
                let field = field.filter(!&valid)?;
                if !field
                    .validity()?
                    .execute_mask(field.len(), ctx)?
                    .all_false()
                {
                    return Ok(false);
                }
            }
            Ok(true)
        }
        DType::Variant(_) | DType::List(..) | DType::FixedSizeList(..) => Ok(false),
        // A row that is not in a shredded leaf must not hold a value of the leaf's dtype.
        leaf => has_no_value(VariantPath::root(), Some(leaf.clone())),
    }
}
