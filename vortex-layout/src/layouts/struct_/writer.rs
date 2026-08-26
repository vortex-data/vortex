// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A writer strategy for struct-typed arrays.
//!
//! [`StructStrategy`] transposes a stream of struct chunks into one ordered stream per field
//! (plus a validity stream when the struct is nullable) and writes each through a configurable
//! child strategy, producing a single [`StructLayout`]. It is a *structural* writer: it does not
//! inspect child dtypes or resolve field-path overrides itself. Dispatching a child to the right
//! layout kind is the job of the caller (see [`TableStrategy`]).
//!
//! [`TableStrategy`]: crate::layouts::table::TableStrategy

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::try_join;
use futures::future::try_join_all;
use futures::pin_mut;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::kanal_ext::KanalExt;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriterContext;
use crate::layouts::struct_::StructLayout;
use crate::layouts::struct_::StructLayoutExt;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// Writes struct-typed arrays into a [`StructLayout`], one child layout per field.
///
/// Each field is written through a strategy resolved by direct field name: an entry in
/// `field_writers` if present, otherwise `default`. When the struct is nullable, its validity
/// bitmap is written through `validity`.
///
/// `StructStrategy` is intentionally unaware of nested dtypes and field-path overrides. To write
/// arbitrarily nested struct trees with per-path overrides, drive it from
/// [`TableStrategy`][crate::layouts::table::TableStrategy], which dispatches on dtype and resolves
/// the per-field child strategies before handing them here.
#[derive(Clone)]
pub struct StructStrategy {
    /// Per-field child strategies, keyed by direct field name. Fields without an entry use
    /// `default`.
    field_writers: HashMap<FieldName, Arc<dyn LayoutStrategy>>,
    /// Strategy for fields that have no entry in `field_writers`.
    default: Arc<dyn LayoutStrategy>,
    /// Strategy for the struct's own validity bitmap, used only when the struct is nullable.
    validity: Arc<dyn LayoutStrategy>,
}

impl StructStrategy {
    /// Create a new struct writer that writes every field through `default` and the validity
    /// bitmap (when present) through `validity`.
    pub fn new(validity: Arc<dyn LayoutStrategy>, default: Arc<dyn LayoutStrategy>) -> Self {
        Self {
            field_writers: HashMap::default(),
            default,
            validity,
        }
    }

    /// Override the strategy for a single field by name.
    pub fn with_field_writer(
        mut self,
        name: impl Into<FieldName>,
        writer: Arc<dyn LayoutStrategy>,
    ) -> Self {
        self.field_writers.insert(name.into(), writer);
        self
    }

    /// Override the strategy for several fields by name at once.
    pub fn with_field_writers(
        mut self,
        writers: impl IntoIterator<Item = (FieldName, Arc<dyn LayoutStrategy>)>,
    ) -> Self {
        self.field_writers.extend(writers);
        self
    }
}

#[async_trait]
impl LayoutStrategy for StructStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();

        let Some(struct_dtype) = dtype.as_struct_fields_opt() else {
            vortex_bail!("StructStrategy can only write struct-typed streams, got {dtype}");
        };

        // Check for unique field names at write time.
        if HashSet::<_, DefaultHashBuilder>::from_iter(struct_dtype.names().iter()).len()
            != struct_dtype.names().len()
        {
            vortex_bail!("StructLayout must have unique field names");
        }
        let is_nullable = dtype.is_nullable();

        // Optimization: when there are no fields, don't spawn any work and just write a trivial
        // StructLayout.
        if struct_dtype.nfields() == 0 && !is_nullable {
            let row_count = stream
                .try_fold(
                    0u64,
                    |acc, (_, arr)| async move { Ok(acc + arr.len() as u64) },
                )
                .await?;
            return Ok(StructLayout::new(row_count, dtype, vec![]).into_layout());
        }

        // stream<struct_chunk> -> stream<vec<column_chunk>>
        let columns_session = session.clone();
        let columns_vec_stream = stream.map(move |chunk| {
            let (sequence_id, chunk) = chunk?;
            let mut sequence_pointer = sequence_id.descend();
            let mut ctx = columns_session.create_execution_ctx();
            let struct_chunk = chunk.clone().execute::<StructArray>(&mut ctx)?;
            let mut columns: Vec<(SequenceId, ArrayRef)> = Vec::new();
            if is_nullable {
                columns.push((
                    sequence_pointer.advance(),
                    chunk
                        .validity()?
                        .execute_mask(chunk.len(), &mut ctx)?
                        .into_array(),
                ));
            }

            columns.extend(
                struct_chunk
                    .iter_unmasked_fields()
                    .map(|field| (sequence_pointer.advance(), field.clone())),
            );

            Ok(columns)
        });

        let mut stream_count = struct_dtype.nfields();
        if is_nullable {
            stream_count += 1;
        }

        let (column_streams_tx, column_streams_rx): (Vec<_>, Vec<_>) =
            (0..stream_count).map(|_| kanal::bounded_async(1)).unzip();

        // Fan out column chunks to their respective transposed streams. Keep this future joined
        // with the column writers so producer panics/errors cannot be hidden as channel EOF.
        let handle = session.handle();
        let fanout_fut = async move {
            pin_mut!(columns_vec_stream);
            while let Some(result) = columns_vec_stream.next().await {
                match result {
                    Ok(columns) => {
                        for (tx, column) in column_streams_tx.iter().zip_eq(columns) {
                            if tx.send(Ok(column)).await.is_err() {
                                vortex_bail!(
                                    "struct column writer finished before all chunks were sent"
                                );
                            }
                        }
                    }
                    Err(e) => {
                        let e: Arc<VortexError> = Arc::new(e);
                        for tx in column_streams_tx.iter() {
                            let _ = tx.send(Err(VortexError::from(Arc::clone(&e)))).await;
                        }
                        return Err(VortexError::from(e));
                    }
                }
            }
            Ok(())
        };

        // First child column is the validity, subsequent children are the individual struct fields
        let column_dtypes: Vec<DType> = if is_nullable {
            std::iter::once(DType::Bool(Nullability::NonNullable))
                .chain(struct_dtype.fields())
                .collect()
        } else {
            struct_dtype.fields().collect()
        };

        let column_names: Vec<FieldName> = if is_nullable {
            std::iter::once(FieldName::from("__validity"))
                .chain(struct_dtype.names().iter().cloned())
                .collect()
        } else {
            struct_dtype.names().iter().cloned().collect()
        };

        let layout_futures: Vec<_> = column_dtypes
            .into_iter()
            .zip_eq(column_streams_rx)
            .zip_eq(column_names)
            .enumerate()
            .map(move |(index, ((dtype, recv), name))| {
                let column_stream =
                    SequentialStreamAdapter::new(dtype, recv.into_stream().boxed()).sendable();
                let child_eof = eof.split_off();
                let session = session.clone();
                let ctx = ctx.clone();
                let segment_sink = Arc::clone(&segment_sink);
                handle.spawn_nested(move |_| {
                    // Validity is written through the validity strategy; every other field
                    // resolves to its named override or the default strategy.
                    let writer = if index == 0 && is_nullable {
                        Arc::clone(&self.validity)
                    } else {
                        self.field_writers
                            .get(&name)
                            .cloned()
                            .unwrap_or_else(|| Arc::clone(&self.default))
                    };

                    async move {
                        writer
                            .write_stream(ctx, segment_sink, column_stream, child_eof, &session)
                            .await
                    }
                })
            })
            .collect();

        let (_success, column_layouts) = try_join(fanout_fut, try_join_all(layout_futures)).await?;
        // TODO(os): transposed stream could count row counts as well,
        // This must hold though, all columns must have the same row count of the struct layout
        let row_count = column_layouts.first().map(|l| l.row_count()).unwrap_or(0);
        Ok(StructLayout::new(row_count, dtype, column_layouts).into_layout())
    }
}
