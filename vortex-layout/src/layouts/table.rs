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
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::StructArray;
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
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;
use vortex_utils::aliases::DefaultHashBuilder;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::list::writer::ListLayoutStrategy;
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
    /// Whether list arrays should be recursively shredded with [`ListLayoutStrategy`].
    list_layout: bool,
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
            list_layout: false,
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

    /// Recursively write list-typed fields using [`ListLayoutStrategy`].
    ///
    /// List elements are written with a nested [`TableStrategy`], so list elements that are
    /// themselves lists or structs are recursively shredded. Offsets use the fallback strategy and
    /// list validity uses the validity strategy.
    pub fn with_list_layout(mut self) -> Self {
        self.list_layout = true;
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
            if field_path.parts().first() == Some(field)
                && let Some(subpath) = field_path.clone().step_into()
            {
                new_writers.insert(subpath, Arc::clone(strategy));
            }
        }

        Self {
            leaf_writers: new_writers,
            validity: Arc::clone(&self.validity),
            fallback: Arc::clone(&self.fallback),
            list_layout: self.list_layout,
        }
    }

    fn list_elements_writer(&self) -> Arc<dyn LayoutStrategy> {
        self.leaf_writers
            .get(&FieldPath::from(Field::ElementType))
            .cloned()
            .unwrap_or_else(|| Arc::new(self.descend(&Field::ElementType)))
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
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();

        if self.list_layout && dtype.is_list() {
            let writer = ListLayoutStrategy::default()
                .with_elements(self.list_elements_writer())
                .with_offsets(Arc::clone(&self.fallback))
                .with_validity(Arc::clone(&self.validity))
                .with_fallback(Arc::clone(&self.fallback));

            return writer
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        // Fallback: if the array is not a struct, fallback to writing a single array.
        if !dtype.is_struct() {
            return self
                .fallback
                .write_stream(ctx, segment_sink, stream, eof, session)
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

        // Spawn a task to fan out column chunks to their respective transposed streams
        let handle = session.handle();
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
                let session = session.clone();
                let ctx = ctx.clone();
                let segment_sink = Arc::clone(&segment_sink);
                handle.spawn_nested(move |h| {
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
                            } else if self.list_layout && dtype.is_list() {
                                // Step into list-typed fields so list elements can recurse.
                                Arc::new(self.descend(&field))
                            } else {
                                // Use fallback for leaf columns
                                Arc::clone(&self.fallback)
                            }
                        });
                    let session = session.with_handle(h);

                    async move {
                        // If we have a matching writer, we use it.
                        // Otherwise, we descend into a new modified one.
                        // Write validity stream
                        if index == 0 && is_nullable {
                            validity
                                .write_stream(ctx, segment_sink, column_stream, child_eof, &session)
                                .await
                        } else {
                            // Use the underlying writer, otherwise use the fallback writer.
                            writer
                                .write_stream(ctx, segment_sink, column_stream, child_eof, &session)
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

    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::FieldPath;
    use vortex_array::field_path;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::table::TableStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::test::SESSION;

    fn flat_strategy() -> Arc<dyn LayoutStrategy> {
        Arc::new(FlatLayoutStrategy::default())
    }

    async fn write<S: LayoutStrategy>(strategy: &S, array: ArrayRef) -> VortexResult<LayoutRef> {
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        strategy
            .write_stream(ArrayContext::empty(), segments, stream, eof, &SESSION)
            .await
    }

    fn basic_list() -> VortexResult<ArrayRef> {
        Ok(ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5].into_array(),
            buffer![0u32, 2, 5, 5].into_array(),
            Validity::NonNullable,
        )?
        .into_array())
    }

    fn nested_list() -> VortexResult<ArrayRef> {
        let inner_list = ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5, 6].into_array(),
            buffer![0u32, 2, 5, 5, 6].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        Ok(ListArray::try_new(
            inner_list,
            buffer![0u32, 2, 4].into_array(),
            Validity::NonNullable,
        )?
        .into_array())
    }

    fn list_of_struct() -> VortexResult<ArrayRef> {
        let struct_array = StructArray::from_fields(
            [
                ("a", buffer![1i32, 2, 3, 4, 5].into_array()),
                ("b", buffer![10i32, 20, 30, 40, 50].into_array()),
            ]
            .as_slice(),
        )?
        .into_array();

        Ok(ListArray::try_new(
            struct_array,
            buffer![0u32, 2, 5, 5].into_array(),
            Validity::NonNullable,
        )?
        .into_array())
    }

    #[tokio::test]
    async fn list_layout_disabled_uses_fallback() -> VortexResult<()> {
        let flat = flat_strategy();
        let strategy = TableStrategy::new(Arc::clone(&flat), flat);

        let layout = write(&strategy, basic_list()?).await?;
        insta::assert_snapshot!(layout.display_tree(), @"vortex.flat, dtype: list(i32), segment: 0");
        Ok(())
    }

    #[tokio::test]
    async fn with_list_layout_shreds_list() -> VortexResult<()> {
        let flat = flat_strategy();
        let strategy = TableStrategy::new(Arc::clone(&flat), flat).with_list_layout();

        let layout = write(&strategy, basic_list()?).await?;
        insta::assert_snapshot!(layout.display_tree(), @"
        vortex.list, dtype: list(i32), children: 2
        ├── elements: vortex.flat, dtype: i32, segment: 0
        └── offsets: vortex.flat, dtype: u32, segment: 1
        ");
        Ok(())
    }

    #[tokio::test]
    async fn with_list_layout_recurses_into_nested_lists() -> VortexResult<()> {
        let flat = flat_strategy();
        let strategy = TableStrategy::new(Arc::clone(&flat), flat).with_list_layout();

        let layout = write(&strategy, nested_list()?).await?;
        insta::assert_snapshot!(layout.display_tree(), @"
        vortex.list, dtype: list(list(i32)), children: 2
        ├── elements: vortex.list, dtype: list(i32), children: 2
        │   ├── elements: vortex.flat, dtype: i32, segment: 1
        │   └── offsets: vortex.flat, dtype: u32, segment: 2
        └── offsets: vortex.flat, dtype: u32, segment: 0
        ");
        Ok(())
    }

    #[tokio::test]
    async fn with_list_layout_recurses_into_list_struct_elements() -> VortexResult<()> {
        let flat = flat_strategy();
        let strategy = TableStrategy::new(Arc::clone(&flat), flat).with_list_layout();

        let layout = write(&strategy, list_of_struct()?).await?;
        insta::assert_snapshot!(layout.display_tree(), @"
        vortex.list, dtype: list({a=i32, b=i32}), children: 2
        ├── elements: vortex.struct, dtype: {a=i32, b=i32}, children: 2
        │   ├── a: vortex.flat, dtype: i32, segment: 1
        │   └── b: vortex.flat, dtype: i32, segment: 2
        └── offsets: vortex.flat, dtype: u32, segment: 0
        ");
        Ok(())
    }

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
