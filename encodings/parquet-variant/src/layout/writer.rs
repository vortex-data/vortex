// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream;
use parking_lot::Mutex;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::StructFields;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_layout::LayoutRef;
use vortex_layout::LayoutStrategy;
use vortex_layout::LayoutWriterContext;
use vortex_layout::layouts::table::TableStrategy;
use vortex_layout::segments::SegmentSinkRef;
use vortex_layout::sequence::SendableSequentialStream;
use vortex_layout::sequence::SequencePointer;
use vortex_layout::sequence::SequentialStreamAdapter;
use vortex_layout::sequence::SequentialStreamExt;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;

use super::ShreddedPath;
use super::expr::is_wrapper;
use super::new_parquet_variant_layout;
use crate::ParquetVariantArrayExt;
use crate::ParquetVariantArraySlotsExt;
use crate::arrow::parquet_variant_for_export;

/// Writes Variant columns as a [`ParquetVariantLayout`], decomposing each chunk into its Parquet
/// Variant storage struct and writing that through a table writer, so every shredded path lands in
/// its own column.
///
/// Residual `value` columns, which hold whatever was not shredded, go to the table's
/// [`variant_residual_strategy`][TableStrategy::variant_residual_strategy] when it has one.
///
/// Every chunk must decompose to the same storage dtype, i.e. use the same shredding schema.
pub struct ParquetVariantLayoutStrategy {
    storage: TableStrategy,
}

impl ParquetVariantLayoutStrategy {
    /// Creates a writer whose storage struct is written with `storage`.
    pub fn new(storage: TableStrategy) -> Self {
        Self { storage }
    }
}

#[async_trait]
impl LayoutStrategy for ParquetVariantLayoutStrategy {
    async fn write_stream(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        mut stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        vortex_ensure!(
            dtype.is_variant(),
            "ParquetVariantLayoutStrategy can only write Variant streams, got {dtype}"
        );

        // The storage dtype is only known once the first chunk is decomposed.
        let residuals = Arc::new(Mutex::new(ResidualTracker::default()));
        let first = match stream.next().await {
            Some(chunk) => {
                let (sequence_id, chunk) = chunk?;
                let mut exec_ctx = session.create_execution_ctx();
                let storage = storage_struct(chunk, &mut exec_ctx)?;
                residuals.lock().observe(&storage, &mut exec_ctx)?;
                Some((sequence_id, storage))
            }
            None => None,
        };
        let storage_dtype = match &first {
            Some((_, storage)) => storage.dtype().clone(),
            None => empty_storage_dtype(&dtype),
        };

        let rest_residuals = Arc::clone(&residuals);
        let rest_dtype = storage_dtype.clone();
        let rest_session = session.clone();
        let rest = stream.map(move |chunk| {
            let (sequence_id, chunk) = chunk?;
            let mut exec_ctx = rest_session.create_execution_ctx();
            let storage = storage_struct(chunk, &mut exec_ctx)?;
            if storage.dtype() != &rest_dtype {
                vortex_bail!(
                    "Variant chunks must share a storage dtype to be written as a \
                     ParquetVariantLayout: expected {rest_dtype}, found {}",
                    storage.dtype()
                );
            }
            rest_residuals.lock().observe(&storage, &mut exec_ctx)?;
            Ok((sequence_id, storage))
        });
        let storage_stream = SequentialStreamAdapter::new(
            storage_dtype.clone(),
            stream::iter(first.map(Ok)).chain(rest).boxed(),
        )
        .sendable();

        let mut storage_writer = self.storage.clone();
        if let Some(residual) = self.storage.variant_residual_strategy() {
            let residual = Arc::clone(residual);
            storage_writer = storage_writer.with_field_writers(
                residual_paths(&storage_dtype)
                    .into_iter()
                    .map(|path| (path, Arc::clone(&residual))),
            );
        }
        let storage = storage_writer
            .write_stream(ctx, segment_sink, storage_stream, eof, session)
            .await?;
        let typed_paths = residuals.lock().typed_paths();
        Ok(new_parquet_variant_layout(dtype, storage, typed_paths)?.into_layout())
    }
}

/// Decomposes a Variant chunk into its Parquet Variant storage struct.
fn storage_struct(chunk: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
    let parquet = parquet_variant_for_export(chunk, ctx)?;
    let parquet = parquet.as_::<crate::ParquetVariant>();

    let mut names = vec![FieldName::from("metadata")];
    let mut fields = vec![parquet.metadata().clone()];
    if let Some(value) = parquet.value() {
        names.push(FieldName::from("value"));
        fields.push(value.clone());
    }
    if let Some(typed_value) = parquet.typed_value() {
        names.push(FieldName::from("typed_value"));
        fields.push(typed_value.clone());
    }
    Ok(StructArray::try_new(
        FieldNames::from_iter(names),
        fields,
        parquet.len(),
        parquet.parquet_variant_validity(),
    )?
    .into_array())
}

/// The paths of every residual `value` column in a Parquet Variant storage struct.
fn residual_paths(storage_dtype: &DType) -> Vec<FieldPath> {
    fn visit(dtype: &DType, path: FieldPath, out: &mut Vec<FieldPath>) {
        let Some(fields) = dtype.as_struct_fields_opt() else {
            return;
        };
        if fields.field("value").is_some_and(|value| value.is_binary()) {
            out.push(path.clone().push("value"));
        }
        let Some(typed_value) = fields.field("typed_value") else {
            return;
        };
        let Some(typed_fields) = typed_value.as_struct_fields_opt() else {
            return;
        };
        for (name, field) in typed_fields.names().iter().zip(typed_fields.fields()) {
            if is_wrapper(&field) {
                visit(
                    &field,
                    path.clone().push("typed_value").push(name.clone()),
                    out,
                );
            }
        }
    }

    let mut paths = Vec::new();
    visit(storage_dtype, FieldPath::root(), &mut paths);
    paths
}

fn empty_storage_dtype(dtype: &DType) -> DType {
    DType::Struct(
        StructFields::from_iter([
            ("metadata", DType::Binary(Nullability::NonNullable)),
            ("value", DType::Binary(Nullability::Nullable)),
        ]),
        dtype.nullability(),
    )
}

/// Tracks, per shredded object path, whether its residual `value` has been null in every row.
#[derive(Default)]
struct ResidualTracker {
    all_null: HashMap<ShreddedPath, bool>,
}

impl ResidualTracker {
    fn observe(&mut self, storage: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<()> {
        let storage = storage.clone().execute::<StructArray>(ctx)?;
        if let Some(typed_value) = storage.unmasked_field_by_name_opt("typed_value") {
            let mut path = Vec::new();
            self.observe_typed(typed_value.clone(), &mut path, ctx)?;
        }
        Ok(())
    }

    fn observe_typed(
        &mut self,
        typed_value: ArrayRef,
        path: &mut ShreddedPath,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if !typed_value.dtype().is_struct() {
            return Ok(());
        }
        let typed_value = typed_value.execute::<StructArray>(ctx)?;
        for (name, wrapper) in typed_value
            .names()
            .iter()
            .zip(typed_value.iter_unmasked_fields())
        {
            if !is_wrapper(wrapper.dtype()) {
                continue;
            }
            path.push(name.clone());
            let wrapper = wrapper.clone().execute::<StructArray>(ctx)?;
            let residual_null = match wrapper.unmasked_field_by_name_opt("value") {
                Some(value) => value
                    .validity()?
                    .execute_mask(value.len(), ctx)?
                    .all_false(),
                None => true,
            };
            *self.all_null.entry(path.clone()).or_insert(true) &= residual_null;
            if let Some(typed) = wrapper.unmasked_field_by_name_opt("typed_value") {
                self.observe_typed(typed.clone(), path, ctx)?;
            }
            path.pop();
        }
        Ok(())
    }

    fn typed_paths(&self) -> Vec<ShreddedPath> {
        let mut paths: Vec<_> = self
            .all_null
            .iter()
            .filter(|(_, all_null)| **all_null)
            .map(|(path, _)| path.clone())
            .collect();
        paths.sort();
        paths
    }
}
