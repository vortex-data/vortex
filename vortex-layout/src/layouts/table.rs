// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A configurable writer strategy for tabular data.
//!
//! [`TableStrategy`] is a *dispatcher*: it inspects the dtype of the stream it is handed and
//! routes struct columns to [`StructStrategy`], list columns to [`ListLayoutStrategy`], and
//! everything else to the configured leaf strategy. Because it hands *itself* (suitably descended)
//! to those structural writers as the strategy for their children, arbitrarily nested struct/list
//! trees are written with no manual wiring.
//!
//! The dispatcher also owns field-path overrides, letting callers force a specific leaf field —
//! at any depth — onto a custom strategy.

use std::env;
use std::sync::Arc;
use std::sync::LazyLock;

use async_trait::async_trait;
use vortex_array::ArrayContext;
use vortex_array::dtype::Field;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::FieldPath;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::layouts::list::writer::ListLayoutStrategy;
use crate::layouts::struct_::StructStrategy;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;

/// Whether [`TableStrategy`] writes list fields using a [`ListLayoutStrategy`] by
/// default. Disabled unless the environment variable `VORTEX_EXPERIMENTAL_LIST_LAYOUT`
/// is set to `1`.
///
/// [`ListLayoutStrategy`]: crate::layouts::list::writer::ListLayoutStrategy
pub fn use_experimental_list_layout() -> bool {
    static USE_EXPERIMENTAL_LIST_LAYOUT: LazyLock<bool> =
        LazyLock::new(|| env::var("VORTEX_EXPERIMENTAL_LIST_LAYOUT").is_ok_and(|v| v == "1"));
    *USE_EXPERIMENTAL_LIST_LAYOUT
}

type ListLayoutFactory = Arc<dyn Fn(ListLayoutStrategy) -> Arc<dyn LayoutStrategy> + Send + Sync>;

/// A configurable strategy for writing nested tabular data, dispatching each (sub)stream to the
/// structural writer for its dtype.
///
/// Dispatch rules, applied to the dtype of the stream handed to [`write_stream`]:
/// - **struct** → [`StructStrategy`], with each field written by its override (if any) or by a
///   descended copy of this dispatcher.
/// - **list** → [`ListLayoutStrategy`], with `elements` written by a descended copy of this
///   dispatcher (so nested structs/lists recurse) and `offsets`/`validity` by the leaf/validity
///   strategies. Gated: only when list decomposition is enabled via
///   [`with_list_layout`][Self::with_list_layout] (off by default); otherwise a list falls through
///   to the leaf strategy.
/// - **anything else** → the leaf strategy.
///
/// [`write_stream`]: LayoutStrategy::write_stream
pub struct TableStrategy {
    /// A set of field-path overrides, e.g. to force one column to be compact-compressed. Keys are
    /// paths relative to the level this dispatcher sits at.
    leaf_writers: HashMap<FieldPath, Arc<dyn LayoutStrategy>>,
    /// The writer for any validity arrays that may be present, at any level of the tree.
    validity: Arc<dyn LayoutStrategy>,
    /// The writer for leaf fields, i.e. anything that is not a struct.
    leaf: Arc<dyn LayoutStrategy>,
    /// Optional factory applied to each dynamically constructed [`ListLayoutStrategy`].
    /// Its presence also enables list decomposition.
    ///
    /// [`ListLayoutStrategy`]: crate::layouts::list::writer::ListLayoutStrategy
    list_layout_factory: Option<ListLayoutFactory>,
}

impl TableStrategy {
    /// Create a new dispatcher with the given `validity` strategy and `fallback` leaf strategy and
    /// no overrides.
    ///
    /// Additional per-field overrides can be configured with
    /// [`with_field_writer`][Self::with_field_writer].
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
            leaf: fallback,
            list_layout_factory: None,
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
        self.leaf = default;
        self
    }

    /// Override the strategy for compressing struct validity at all levels of the schema tree.
    pub fn with_validity_strategy(mut self, validity: Arc<dyn LayoutStrategy>) -> Self {
        self.validity = validity;
        self
    }

    /// Enable writing list fields with [`ListLayoutStrategy`].
    ///
    /// **Note**: this is an unstable and experimental layout that is expected to change.
    /// Using it may lead to unreadable files in the future.
    pub fn with_list_layout(self) -> Self {
        self.with_list_layout_factory(|strategy| Arc::new(strategy))
    }

    /// Enable writing list fields with [`ListLayoutStrategy`] and wrap each list writer. This
    /// allows repartitioning or zoning to operate in the list's outer-row space before shredding.
    ///
    /// **Note**: this is an unstable and experimental layout that is expected to change.
    /// Using it may lead to unreadable files in the future.
    pub fn with_list_layout_factory(
        mut self,
        factory: impl Fn(ListLayoutStrategy) -> Arc<dyn LayoutStrategy> + Send + Sync + 'static,
    ) -> Self {
        self.list_layout_factory = Some(Arc::new(factory));
        self
    }
}

impl TableStrategy {
    /// Build the [`StructStrategy`] used to write a struct-typed stream at this level.
    ///
    /// Each field that has an override (or a deeper override beneath it) is resolved up front;
    /// every other field falls through to a clean descended dispatcher.
    fn struct_strategy(&self) -> StructStrategy {
        let mut field_writers: HashMap<FieldName, Arc<dyn LayoutStrategy>> = HashMap::default();

        // The distinct named first-segments of our override paths are the only fields that need
        // anything other than the default dispatcher.
        let mut named_first: HashSet<FieldName> = HashSet::default();
        for path in self.leaf_writers.keys() {
            if let Some(Field::Name(name)) = path.parts().first() {
                named_first.insert(name.clone());
            }
        }

        for name in named_first {
            // `validate_path` forbids overlapping overrides, so a name has *either* an exact
            // single-segment override *or* deeper overrides, never both.
            let writer = match self.leaf_writers.get(&FieldPath::from_name(name.clone())) {
                Some(exact) => Arc::clone(exact),
                None => {
                    Arc::new(self.descend(&Field::Name(name.clone()))) as Arc<dyn LayoutStrategy>
                }
            };
            field_writers.insert(name, writer);
        }

        StructStrategy::new(Arc::clone(&self.validity), Arc::new(self.descend_clean()))
            .with_field_writers(field_writers)
    }

    /// Build the [`ListLayoutStrategy`] used to write a list field stream at this level.
    ///
    /// The `elements` sub-column is routed back through a clean descended dispatcher so nested
    /// structs/lists recurse; `offsets` go straight to the leaf (they are always a primitive
    /// column); and `validity` uses the shared validity strategy.
    fn list_strategy(&self) -> Option<Arc<dyn LayoutStrategy>> {
        let factory = self.list_layout_factory.as_ref()?;
        let list_layout = ListLayoutStrategy::default()
            .with_elements(Arc::new(self.descend_clean()))
            .with_offsets(Arc::clone(&self.leaf))
            .with_validity(Arc::clone(&self.validity))
            .with_fallback(Arc::clone(&self.leaf));
        Some(factory(list_layout))
    }

    /// Descend into a subfield, retaining only the overrides that apply beneath it (rebased to be
    /// relative to the child).
    fn descend(&self, field: &Field) -> Self {
        let mut new_writers = HashMap::with_capacity(self.leaf_writers.len());

        for (field_path, strategy) in &self.leaf_writers {
            if field_path.parts().first() == Some(field)
                && let Some(subpath) = field_path.clone().step_into()
                && !subpath.is_root()
            {
                new_writers.insert(subpath, Arc::clone(strategy));
            }
        }

        Self {
            leaf_writers: new_writers,
            validity: Arc::clone(&self.validity),
            leaf: Arc::clone(&self.leaf),
            list_layout_factory: self.list_layout_factory.clone(),
        }
    }

    /// A copy of this dispatcher with no overrides, used as the default child strategy for fields
    /// that carry no override.
    fn descend_clean(&self) -> Self {
        Self {
            leaf_writers: HashMap::default(),
            validity: Arc::clone(&self.validity),
            leaf: Arc::clone(&self.leaf),
            list_layout_factory: self.list_layout_factory.clone(),
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

/// Dispatches each stream to the structural writer for its dtype.
#[async_trait]
impl LayoutStrategy for TableStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();

        if dtype.is_struct() {
            return self
                .struct_strategy()
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        if dtype.is_list()
            && let Some(list_strategy) = self.list_strategy()
        {
            return list_strategy
                .write_stream(ctx, segment_sink, stream, eof, session)
                .await;
        }

        // Leaf: hand off to the leaf strategy.
        self.leaf
            .write_stream(ctx, segment_sink, stream, eof, session)
            .await
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;
    use std::sync::Arc;
    use std::task::Poll;

    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::ChunkedArray;
    use vortex_array::arrays::ListArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldPath;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::field_path;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::LayoutRef;
    use crate::LayoutStrategy;
    use crate::layouts::chunked::writer::ChunkedLayoutStrategy;
    use crate::layouts::flat::writer::FlatLayoutStrategy;
    use crate::layouts::list::List;
    use crate::layouts::repartition::RepartitionStrategy;
    use crate::layouts::repartition::RepartitionWriterOptions;
    use crate::layouts::table::TableStrategy;
    use crate::layouts::zoned::Zoned;
    use crate::layouts::zoned::writer::ZonedLayoutOptions;
    use crate::layouts::zoned::writer::ZonedStrategy;
    use crate::segments::TestSegments;
    use crate::sequence::SequenceId;
    use crate::sequence::SequentialArrayStreamExt;
    use crate::sequence::SequentialStreamAdapter;
    use crate::sequence::SequentialStreamExt;
    use crate::test::SESSION;

    async fn write<S: LayoutStrategy>(strategy: &S, array: ArrayRef) -> VortexResult<LayoutRef> {
        let segments = Arc::new(TestSegments::default());
        let (ptr, eof) = SequenceId::root().split();
        let stream = array.to_array_stream().sequenced(ptr);
        strategy
            .write_stream(ArrayContext::empty(), segments, stream, eof, &SESSION)
            .await
    }

    /// A plain table dispatcher with no overrides. `flat` here is both the validity and leaf
    /// strategy.
    fn flat_table() -> TableStrategy {
        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        TableStrategy::new(Arc::clone(&flat), flat)
    }

    /// The dispatcher shreds a top-level struct into one child per field.
    #[tokio::test]
    async fn dispatches_struct() -> VortexResult<()> {
        let struct_array = StructArray::from_fields(
            [
                ("a", buffer![1i32, 2, 3].into_array()),
                ("b", buffer![10i32, 20, 30].into_array()),
            ]
            .as_slice(),
        )?
        .into_array();

        let layout = write(&flat_table(), struct_array).await?;
        insta::assert_snapshot!(layout.display_tree(), @r"
        vortex.struct, dtype: {a=i32, b=i32}, children: 2
        ├── a: vortex.flat, dtype: i32, segment: 0
        └── b: vortex.flat, dtype: i32, segment: 1
        ");
        Ok(())
    }

    /// A `list<list<i32>>` column: the dispatcher recurses into itself so the outer list's
    /// `elements` are decomposed as a nested `ListLayout`.
    #[tokio::test]
    async fn dispatches_nested_list() -> VortexResult<()> {
        let inner = ListArray::try_new(
            buffer![1i32, 2, 3, 4, 5, 6].into_array(),
            buffer![0u32, 2, 5, 5, 6].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let outer = ListArray::try_new(
            inner,
            buffer![0u32, 2, 4].into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let layout = write(&flat_table().with_list_layout(), outer).await?;
        insta::assert_snapshot!(layout.display_tree(), @r"
        vortex.list, dtype: list(list(i32)), children: 2
        ├── elements: vortex.list, dtype: list(i32), children: 2
        │   ├── elements: vortex.flat, dtype: i32, segment: 1
        │   └── offsets: vortex.flat, dtype: u64, segment: 2
        └── offsets: vortex.flat, dtype: u64, segment: 0
        ");
        Ok(())
    }

    /// A `struct<{ items: list<struct<{a,b}>>? }>` column: list decomposition recurses into struct
    /// decomposition for the elements, and a nullable list writes a validity child.
    #[tokio::test]
    async fn dispatches_struct_list_struct() -> VortexResult<()> {
        let inner_struct = StructArray::from_fields(
            [
                ("a", buffer![1i32, 2, 3, 4, 5].into_array()),
                ("b", buffer![10i32, 20, 30, 40, 50].into_array()),
            ]
            .as_slice(),
        )?
        .into_array();
        let items = ListArray::try_new(
            inner_struct,
            buffer![0u32, 2, 5, 5].into_array(),
            Validity::Array(BoolArray::from_iter([true, false, true]).into_array()),
        )?
        .into_array();
        let st = StructArray::from_fields([("items", items)].as_slice())?.into_array();

        let layout = write(&flat_table().with_list_layout(), st).await?;
        insta::assert_snapshot!(layout.display_tree(), @r"
        vortex.struct, dtype: {items=list({a=i32, b=i32})?}, children: 1
        └── items: vortex.list, dtype: list({a=i32, b=i32})?, children: 3
            ├── elements: vortex.struct, dtype: {a=i32, b=i32}, children: 2
            │   ├── a: vortex.flat, dtype: i32, segment: 2
            │   └── b: vortex.flat, dtype: i32, segment: 3
            ├── offsets: vortex.flat, dtype: u64, segment: 0
            └── validity: vortex.flat, dtype: bool, segment: 1
        ");
        Ok(())
    }

    /// A multi-chunk `list<i32>` written with a chunked leaf: each sub-column (`elements`,
    /// `offsets`) becomes its own `ChunkedLayout`, so elements are chunked independently of rows.
    /// This is the "list-of-chunkeds" topology top-level decomposition unlocks.
    #[tokio::test]
    async fn dispatches_chunked_list() -> VortexResult<()> {
        let chunk0 = ListArray::try_new(
            buffer![1i32, 2, 3].into_array(),
            buffer![0u32, 2, 3].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let chunk1 = ListArray::try_new(
            buffer![4i32, 5, 6, 7].into_array(),
            buffer![0u32, 1, 4].into_array(),
            Validity::NonNullable,
        )?
        .into_array();
        let dtype = chunk0.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![chunk0, chunk1], dtype)?.into_array();

        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        let dispatcher = TableStrategy::new(
            Arc::clone(&flat),
            Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default())),
        )
        .with_list_layout();
        let layout = write(&dispatcher, chunked).await?;
        insta::assert_snapshot!(layout.display_tree(), @r"
        vortex.list, dtype: list(i32), children: 2
        ├── elements: vortex.chunked, dtype: i32, children: 2
        │   ├── [0]: vortex.flat, dtype: i32, segment: 0
        │   └── [1]: vortex.flat, dtype: i32, segment: 1
        └── offsets: vortex.chunked, dtype: u64, children: 2
            ├── [0]: vortex.flat, dtype: u64, segment: 2
            └── [1]: vortex.flat, dtype: u64, segment: 3
        ");
        Ok(())
    }

    /// A wrapper can repartition and zone lists in outer-row space before decomposition.
    #[tokio::test]
    async fn wraps_list_strategy_before_decomposition() -> VortexResult<()> {
        let list = ListArray::try_new(
            PrimitiveArray::from_iter(0..9_i32).into_array(),
            PrimitiveArray::from_iter(0..=9_u32).into_array(),
            Validity::NonNullable,
        )?
        .into_array();

        let row_block_size = NonZeroUsize::new(4).vortex_expect("4 is non-zero");
        let flat: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        let stats = Arc::clone(&flat);
        let chunked: Arc<dyn LayoutStrategy> =
            Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()));
        let dispatcher = TableStrategy::new(Arc::clone(&flat), chunked).with_list_layout_factory(
            move |list_layout| {
                let zoned = ZonedStrategy::new(
                    list_layout,
                    Arc::clone(&stats),
                    ZonedLayoutOptions {
                        block_size: row_block_size,
                        ..Default::default()
                    },
                );
                Arc::new(RepartitionStrategy::new(
                    zoned,
                    RepartitionWriterOptions {
                        block_size_minimum: 0,
                        block_len_multiple: row_block_size.get(),
                        block_size_target: None,
                        canonicalize: false,
                    },
                )) as Arc<dyn LayoutStrategy>
            },
        );

        let layout = write(&dispatcher, list).await?;
        let zoned = layout.as_::<Zoned>();
        assert_eq!(zoned.zone_len(), 4);
        assert_eq!(zoned.nzones(), 3);

        let data = layout
            .slot(0)?
            .vortex_expect("ZonedLayout always has a data child");
        assert!(data.is::<List>());
        assert_eq!(data.row_count(), 9);
        Ok(())
    }

    /// A non-struct stream is not shredded; it is handed straight to the leaf strategy.
    #[tokio::test]
    async fn non_struct_input_uses_leaf() -> VortexResult<()> {
        let primitive = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();
        let layout = write(&flat_table(), primitive).await?;
        insta::assert_snapshot!(layout.display_tree(), @"vortex.flat, dtype: i32, segment: 0");
        Ok(())
    }

    /// A multi-chunk struct is transposed per field; each column is written by a chunked leaf.
    #[tokio::test]
    async fn chunked_struct() -> VortexResult<()> {
        let validity: Arc<dyn LayoutStrategy> = Arc::new(FlatLayoutStrategy::default());
        let chunked_flat: Arc<dyn LayoutStrategy> =
            Arc::new(ChunkedLayoutStrategy::new(FlatLayoutStrategy::default()));
        let dispatcher = TableStrategy::new(validity, chunked_flat);

        let c0 = StructArray::from_fields(
            [
                ("a", buffer![1i32, 2].into_array()),
                ("b", buffer![10i32, 20].into_array()),
            ]
            .as_slice(),
        )?
        .into_array();
        let c1 = StructArray::from_fields(
            [
                ("a", buffer![3i32].into_array()),
                ("b", buffer![30i32].into_array()),
            ]
            .as_slice(),
        )?
        .into_array();
        let dtype = c0.dtype().clone();
        let chunked = ChunkedArray::try_new(vec![c0, c1], dtype)?.into_array();

        let layout = write(&dispatcher, chunked).await?;
        insta::assert_snapshot!(layout.display_tree(), @r"
        vortex.struct, dtype: {a=i32, b=i32}, children: 2
        ├── a: vortex.chunked, dtype: i32, children: 2
        │   ├── [0]: vortex.flat, dtype: i32, segment: 0
        │   └── [1]: vortex.flat, dtype: i32, segment: 1
        └── b: vortex.chunked, dtype: i32, children: 2
            ├── [0]: vortex.flat, dtype: i32, segment: 2
            └── [1]: vortex.flat, dtype: i32, segment: 3
        ");
        Ok(())
    }

    /// A field override on a struct field is honored ahead of the default leaf strategy.
    #[tokio::test]
    async fn field_override_is_used() -> VortexResult<()> {
        let struct_array = StructArray::from_fields(
            [
                ("a", buffer![1i32, 2, 3].into_array()),
                ("b", buffer![10i32, 20, 30].into_array()),
            ]
            .as_slice(),
        )?
        .into_array();

        let strategy =
            flat_table().with_field_writer(field_path!(a), Arc::new(FlatLayoutStrategy::default()));
        let layout = write(&strategy, struct_array).await?;
        insta::assert_snapshot!(layout.display_tree(), @r"
        vortex.struct, dtype: {a=i32, b=i32}, children: 2
        ├── a: vortex.flat, dtype: i32, segment: 0
        └── b: vortex.flat, dtype: i32, segment: 1
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

    #[tokio::test]
    #[should_panic(expected = "panic while transposing table stream")]
    async fn table_fanout_panic_propagates() {
        let ctx = ArrayContext::empty();
        let segments = Arc::new(TestSegments::default());
        let (_, eof) = SequenceId::root().split();
        let dtype = DType::Struct(
            StructFields::from_iter([(
                "a",
                DType::Primitive(PType::I32, Nullability::NonNullable),
            )]),
            Nullability::NonNullable,
        );
        let stream =
            futures::stream::poll_fn(|_| -> Poll<Option<VortexResult<(SequenceId, ArrayRef)>>> {
                panic!("panic while transposing table stream");
            });
        let strategy = TableStrategy::new(
            Arc::new(FlatLayoutStrategy::default()),
            Arc::new(FlatLayoutStrategy::default()),
        );

        strategy
            .write_stream(
                ctx,
                segments,
                SequentialStreamAdapter::new(dtype, stream).sendable(),
                eof,
                &SESSION,
            )
            .await
            .unwrap();
    }
}
