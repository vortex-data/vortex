// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A configurable writer strategy for tabular data.
//!
//! Allows the caller to override specific leaf fields with custom layout strategies.

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::future::try_join_all;
use futures::pin_mut;
use itertools::Itertools;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldPath;
use vortex_array::dtype::Nullability;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::kanal_ext::KanalExt;
use vortex_io::runtime::Handle;
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::struct_::StructLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequenceId;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt;

/// A configurable strategy for writing tables with nested field columns, allowing
/// overrides for specific leaf columns.
pub struct TableStrategy {
    /// A set of leaf field overrides, e.g. to force one column to be compact-compressed.
    leaf_writers: HashMap<FieldPath, Arc<dyn LayoutStrategy>>,
    /// The writer for any validity arrays that may be present
    validity: Arc<dyn LayoutStrategy>,
    /// The fallback writer for any fields that do not have an explicit writer set in `leaf_writers`
    fallback: Arc<dyn LayoutStrategy>,
}

impl TableStrategy {
    /// Create a new writer with the specified write strategies for validity, and for all leaf
    /// fields, with no overrides.
    ///
    /// Additional overrides can be configured using the `with_leaf_strategy` method.
    ///
    /// ## Example
    ///
    /// ```ignore
    /// # use std::sync::Arc;
    /// # use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
    /// # use vortex_layout::layouts::table::TableStrategy;
    ///
    /// // Build a write strategy that does not compress validity or any leaf fields.
    /// let flat = Arc::new(FlatLayoutStrategy::default());
    ///
    /// let strategy = TableStrategy::new(Arc::<FlatLayoutStrategy>::clone(&flat), Arc::<FlatLayoutStrategy>::clone(&flat));
    /// ```
    pub fn new(validity: Arc<dyn LayoutStrategy>, fallback: Arc<dyn LayoutStrategy>) -> Self {
        Self {
            leaf_writers: Default::default(),
            validity,
            fallback,
        }
    }

    /// Add a custom write strategy for the given leaf field.
    ///
    /// ## Example
    ///
    /// ```ignore
    /// # use std::sync::Arc;
    /// # use vortex_array::dtype::{field_path, Field, FieldPath};
    /// # use vortex_btrblocks::BtrBlocksCompressor;
    /// # use vortex_layout::layouts::compressed::CompressingStrategy;
    /// # use vortex_layout::layouts::flat::writer::FlatLayoutStrategy;
    /// # use vortex_layout::layouts::table::TableStrategy;
    ///
    /// // A strategy for compressing data using the balanced BtrBlocks compressor.
    /// let compress =
    ///     CompressingStrategy::new(FlatLayoutStrategy::default(), BtrBlocksCompressor::default());
    ///
    /// // Our combined strategy uses no compression for validity buffers, BtrBlocks compression
    /// // for most columns, and stores a nested binary column uncompressed (flat) because it
    /// // is pre-compressed or never filtered on.
    /// let strategy = TableStrategy::new(
    ///         Arc::new(FlatLayoutStrategy::default()),
    ///         Arc::new(compress),
    ///     )
    ///     .with_field_writer(
    ///         field_path!(request.body.bytes),
    ///         Arc::new(FlatLayoutStrategy::default()),
    ///     );
    /// ```
    pub fn with_field_writer(
        mut self,
        field_path: impl Into<FieldPath>,
        writer: Arc<dyn LayoutStrategy>,
    ) -> Self {
        self.leaf_writers
            .insert(self.validate_path(field_path.into()), writer);
        self
    }

    /// Set writers for several fields at once.
    ///
    /// See also: [`with_field_writer`][Self::with_field_writer].
    pub fn with_field_writers(
        mut self,
        writers: impl IntoIterator<Item = (FieldPath, Arc<dyn LayoutStrategy>)>,
    ) -> Self {
        for (field_path, strategy) in writers {
            self.leaf_writers
                .insert(self.validate_path(field_path), strategy);
        }
        self
    }

    /// Override the default strategy for leaf columns that don't have overrides.
    pub fn with_default_strategy(mut self, default: Arc<dyn LayoutStrategy>) -> Self {
        self.fallback = default;
        self
    }

    /// Override the strategy for compressing struct validity at all levels of the schema tree.
    pub fn with_validity_strategy(mut self, validity: Arc<dyn LayoutStrategy>) -> Self {
        self.validity = validity;
        self
    }
}

impl TableStrategy {
    /// Descend into a subfield for the writer.
    fn descend(&self, field: &Field) -> Self {
        // Start with the existing set of overrides, then only retain the ones that contain
        // the current field
        let mut new_writers = HashMap::with_capacity(self.leaf_writers.len());

        for (field_path, strategy) in &self.leaf_writers {
            if field_path.starts_with_field(field)
                && let Some(subpath) = field_path.clone().step_into()
            {
                new_writers.insert(subpath, Arc::clone(strategy));
            }
        }

        Self {
            leaf_writers: new_writers,
            validity: Arc::clone(&self.validity),
            fallback: Arc::clone(&self.fallback),
        }
    }

    fn validate_path(&self, path: FieldPath) -> FieldPath {
        assert!(
            !path.is_root(),
            "Do not set override as a root strategy, instead set the default strategy"
        );

        // Validate that the field path does not conflict with any overrides
        // that we've added by overlapping.
        for field_path in self.leaf_writers.keys() {
            assert!(
                !path.overlap(field_path),
                "Override for field_path {path} conflicts with existing override for {field_path}"
            );
        }

        path
    }
}

/// Specialized strategy for when we exactly know the input schema.
#[async_trait]
impl LayoutStrategy for TableStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();

        // Fallback: if the array is not a struct, fallback to writing a single array.
        if !dtype.is_struct() {
            return self
                .fallback
                .write_stream(ctx, segment_sink, stream, eof, handle)
                .await;
        }

        let struct_dtype = dtype.as_struct_fields();

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
        let columns_vec_stream = stream.map(move |chunk| {
            let (sequence_id, chunk) = chunk?;
            let mut sequence_pointer = sequence_id.descend();
            let struct_chunk = chunk.to_struct();
            let mut columns: Vec<(SequenceId, ArrayRef)> = Vec::new();
            if is_nullable {
                columns.push((
                    sequence_pointer.advance(),
                    chunk.validity_mask()?.into_array(),
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

        // Spawn a task to fan out column chunks to their respective transposed streams
        handle
            .spawn(async move {
                pin_mut!(columns_vec_stream);
                while let Some(result) = columns_vec_stream.next().await {
                    match result {
                        Ok(columns) => {
                            for (tx, column) in column_streams_tx.iter().zip_eq(columns.into_iter())
                            {
                                let _ = tx.send(Ok(column)).await;
                            }
                        }
                        Err(e) => {
                            let e: Arc<VortexError> = Arc::new(e);
                            for tx in column_streams_tx.iter() {
                                let _ = tx.send(Err(VortexError::from(Arc::clone(&e)))).await;
                            }
                            break;
                        }
                    }
                }
            })
            .detach();

        // First child column is the validity, subsequence children are the individual struct fields
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
                    SequentialStreamAdapter::new(dtype.clone(), recv.into_stream().boxed())
                        .sendable();
                let child_eof = eof.split_off();
                let field = Field::Name(name.clone());
                handle.spawn_nested(|h| {
                    let validity = Arc::clone(&self.validity);
                    // descend further and try with new fields
                    let writer = self
                        .leaf_writers
                        .get(&FieldPath::from_name(name))
                        .cloned()
                        .unwrap_or_else(|| {
                            if dtype.is_struct() {
                                // Step into the field path for struct columns
                                Arc::new(self.descend(&field))
                            } else {
                                // Use fallback for leaf columns
                                Arc::clone(&self.fallback)
                            }
                        });
                    let ctx = ctx.clone();
                    let segment_sink = Arc::clone(&segment_sink);

                    async move {
                        // If we have a matching writer, we use it.
                        // Otherwise, we descend into a new modified one.
                        // Write validity stream
                        if index == 0 && is_nullable {
                            validity
                                .write_stream(ctx, segment_sink, column_stream, child_eof, h)
                                .await
                        } else {
                            // Use the underlying writer, otherwise use the fallback writer.
                            writer
                                .write_stream(ctx, segment_sink, column_stream, child_eof, h)
                                .await
                        }
                    }
                })
            })
            .collect();

        let column_layouts = try_join_all(layout_futures).await?;
        // TODO(os): transposed stream could count row counts as well,
        // This must hold though, all columns must have the same row count of the struct layout
        let row_count = column_layouts.first().map(|l| l.row_count()).unwrap_or(0);
        Ok(StructLayout::new(row_count, dtype, column_layouts).into_layout())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use vortex_array::dtype::FieldPath;
    use vortex_array::field_path;

    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::table::TableStrategy;

    #[test]
    #[should_panic(
        expected = "Override for field_path $a.$b conflicts with existing override for $a.$b.$c"
    )]
    fn test_overlapping_paths_fail() {
        let flat = Arc::new(FlatLayoutStrategy::default());

        // Success
        let path = TableStrategy::new(
            Arc::<FlatLayoutStrategy>::clone(&flat),
            Arc::<FlatLayoutStrategy>::clone(&flat),
        )
        .with_field_writer(field_path!(a.b.c), Arc::<FlatLayoutStrategy>::clone(&flat));

        // Should panic right here.
        let _path = path.with_field_writer(field_path!(a.b), flat);
    }

    #[test]
    #[should_panic(
        expected = "Do not set override as a root strategy, instead set the default strategy"
    )]
    fn test_root_override() {
        let flat = Arc::new(FlatLayoutStrategy::default());
        let _strategy = TableStrategy::new(
            Arc::<FlatLayoutStrategy>::clone(&flat),
            Arc::<FlatLayoutStrategy>::clone(&flat),
        )
        .with_field_writer(FieldPath::root(), flat);
    }
}
