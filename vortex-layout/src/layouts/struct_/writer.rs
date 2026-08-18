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
use futures::future::try_join_all;
use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriter;
use crate::LayoutWriterContext;
use crate::layouts::struct_::StructLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SequenceId;
use crate::strategy::LayoutWriterActor;

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

impl LayoutStrategy for StructStrategy {
    fn new_writer(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        dtype: DType,
        session: &VortexSession,
    ) -> VortexResult<Box<dyn LayoutWriter>> {
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

        let buffered_bytes = ctx.buffered_bytes_tracker().clone();
        let handle = session.handle();
        let children = column_dtypes
            .into_iter()
            .zip_eq(column_names)
            .enumerate()
            .map(|(index, (dtype, name))| {
                let strategy = if index == 0 && is_nullable {
                    Arc::clone(&self.validity)
                } else {
                    self.field_writers
                        .get(&name)
                        .cloned()
                        .unwrap_or_else(|| Arc::clone(&self.default))
                };
                let writer =
                    strategy.new_writer(ctx.clone(), Arc::clone(&segment_sink), dtype, session)?;
                Ok(LayoutWriterActor::spawn(
                    writer,
                    buffered_bytes.clone(),
                    &handle,
                ))
            })
            .collect::<VortexResult<Vec<_>>>()?;

        Ok(Box::new(StructLayoutWriter {
            dtype,
            is_nullable,
            children,
            exec_ctx: session.create_execution_ctx(),
            row_count: 0,
        }))
    }
}

struct StructLayoutWriter {
    dtype: DType,
    is_nullable: bool,
    children: Vec<LayoutWriterActor>,
    exec_ctx: vortex_array::ExecutionCtx,
    row_count: u64,
}

#[async_trait]
impl LayoutWriter for StructLayoutWriter {
    async fn write(&mut self, sequence_id: SequenceId, chunk: ArrayRef) -> VortexResult<()> {
        self.row_count += chunk.len() as u64;
        if self.children.is_empty() {
            return Ok(());
        }

        let struct_chunk = chunk.clone().execute::<StructArray>(&mut self.exec_ctx)?;
        let mut columns = Vec::with_capacity(self.children.len());
        if self.is_nullable {
            columns.push(
                chunk
                    .validity()?
                    .execute_mask(chunk.len(), &mut self.exec_ctx)?
                    .into_array(),
            );
        }
        columns.extend(struct_chunk.iter_unmasked_fields().cloned());

        let mut sequence = sequence_id.descend();
        let child_sequences = (0..self.children.len())
            .map(|_| sequence.advance())
            .collect::<Vec<_>>();
        try_join_all(
            self.children
                .iter_mut()
                .zip_eq(columns)
                .zip(child_sequences)
                .map(|((writer, column), sequence_id)| writer.write(sequence_id, column)),
        )
        .await?;
        Ok(())
    }

    async fn finish(&mut self, sequence_id: SequenceId) -> VortexResult<()> {
        let mut sequence = sequence_id.descend();
        let child_sequences = (0..self.children.len())
            .map(|_| sequence.advance())
            .collect::<Vec<_>>();
        try_join_all(
            self.children
                .iter_mut()
                .zip(child_sequences)
                .map(|(child, sequence_id)| child.finish(sequence_id)),
        )
        .await?;
        Ok(())
    }

    async fn close(self: Box<Self>) -> VortexResult<LayoutRef> {
        let Self {
            dtype,
            children,
            row_count,
            ..
        } = *self;
        let mut layouts = Vec::with_capacity(children.len());
        for mut writer in children {
            layouts.push(writer.take_layout()?);
        }
        Ok(StructLayout::new(row_count, dtype, layouts).into_layout())
    }
}
