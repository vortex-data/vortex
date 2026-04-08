// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::sync::Arc;

use arcref::ArcRef;
use itertools::Itertools;
use vortex_array::SerializeMetadata;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::LayoutEncodingId;
use crate::LayoutEncodingRef;
use crate::LayoutReaderRef;
use crate::VTable;
use crate::display::DisplayLayoutTree;
use crate::display::display_tree_with_segment_sizes;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

pub type LayoutId = ArcRef<str>;

pub type LayoutRef = Arc<dyn Layout>;

pub trait Layout: 'static + Send + Sync + Debug + private::Sealed {
    fn as_any(&self) -> &dyn Any;

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync>;

    fn to_layout(&self) -> LayoutRef;

    /// Returns the [`crate::LayoutEncoding`] for this layout.
    fn encoding(&self) -> LayoutEncodingRef;

    /// The number of rows in this layout.
    fn row_count(&self) -> u64;

    /// The dtype of this layout when projected with the root scope.
    fn dtype(&self) -> &DType;

    /// The number of children in this layout.
    fn nchildren(&self) -> usize;

    /// Get the child at the given index.
    fn child(&self, idx: usize) -> VortexResult<LayoutRef>;

    /// Get the relative row offset of the child at the given index, returning `None` for
    /// any auxiliary children, e.g. dictionary values, zone maps, etc.
    fn child_type(&self, idx: usize) -> LayoutChildType;

    /// Get the metadata for this layout.
    fn metadata(&self) -> Vec<u8>;

    /// Get the segment IDs for this layout.
    fn segment_ids(&self) -> Vec<SegmentId>;

    fn new_reader(
        &self,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
    ) -> VortexResult<LayoutReaderRef>;
}

pub trait IntoLayout {
    /// Converts this type into a [`LayoutRef`].
    fn into_layout(self) -> LayoutRef;
}

/// A type that allows us to identify how a layout child relates to its parent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LayoutChildType {
    /// A layout child that retains the same schema and row offset position in the dataset.
    Transparent(Arc<str>),
    /// A layout child that provides auxiliary data, e.g. dictionary values, zone maps, etc.
    /// Contains a human-readable name of the child.
    Auxiliary(Arc<str>),
    /// A layout child that represents a row-based chunk of data.
    /// Contains the chunk index and relative row offset of the child.
    Chunk((usize, u64)),
    /// A layout child that represents a single field of data.
    /// Contains the field name of the child.
    Field(FieldName),
}

impl LayoutChildType {
    /// Returns the name of this child.
    pub fn name(&self) -> Arc<str> {
        match self {
            LayoutChildType::Chunk((idx, _offset)) => format!("[{idx}]").into(),
            LayoutChildType::Auxiliary(name) => Arc::clone(name),
            LayoutChildType::Transparent(name) => Arc::clone(name),
            LayoutChildType::Field(name) => name.clone().into(),
        }
    }

    /// Returns the relative row offset of this child.
    /// For auxiliary children, this is `None`.
    pub fn row_offset(&self) -> Option<u64> {
        match self {
            LayoutChildType::Chunk((_idx, offset)) => Some(*offset),
            LayoutChildType::Auxiliary(_) => None,
            LayoutChildType::Transparent(_) => Some(0),
            LayoutChildType::Field(_) => Some(0),
        }
    }
}

impl dyn Layout + '_ {
    /// The ID of the encoding for this layout.
    pub fn encoding_id(&self) -> LayoutEncodingId {
        self.encoding().id()
    }

    /// The children of this layout.
    pub fn children(&self) -> VortexResult<Vec<LayoutRef>> {
        (0..self.nchildren()).map(|i| self.child(i)).try_collect()
    }

    /// The child types of this layout.
    pub fn child_types(&self) -> impl Iterator<Item = LayoutChildType> {
        (0..self.nchildren()).map(|i| self.child_type(i))
    }

    /// The names of the children of this layout.
    pub fn child_names(&self) -> impl Iterator<Item = Arc<str>> {
        self.child_types().map(|child| child.name())
    }

    /// The row offsets of the children of this layout, where `None` indicates an auxiliary child.
    pub fn child_row_offsets(&self) -> impl Iterator<Item = Option<u64>> {
        self.child_types().map(|child| child.row_offset())
    }

    pub fn is<V: VTable>(&self) -> bool {
        self.as_opt::<V>().is_some()
    }

    /// Downcast a layout to a specific type.
    pub fn as_<V: VTable>(&self) -> &V::Layout {
        self.as_opt::<V>().vortex_expect("Failed to downcast")
    }

    /// Downcast a layout to a specific type.
    pub fn as_opt<V: VTable>(&self) -> Option<&V::Layout> {
        self.as_any()
            .downcast_ref::<LayoutAdapter<V>>()
            .map(|adapter| &adapter.0)
    }

    /// Downcast a layout to a specific type.
    pub fn into<V: VTable>(self: Arc<Self>) -> Arc<V::Layout> {
        let layout_adapter = self
            .as_any_arc()
            .downcast::<LayoutAdapter<V>>()
            .map_err(|_| vortex_err!("Invalid layout type"))
            .vortex_expect("Invalid layout type");

        // SAFETY: LayoutAdapter<V> is #[repr(transparent)] (see line 192) which guarantees
        // it has the same memory layout as V::Layout. The downcast above ensures we have
        // the correct type. This transmute is safe because both Arc types point to data
        // with identical layout and alignment.
        unsafe { std::mem::transmute::<Arc<LayoutAdapter<V>>, Arc<V::Layout>>(layout_adapter) }
    }

    /// Depth-first traversal of the layout and its children.
    pub fn depth_first_traversal(&self) -> impl Iterator<Item = VortexResult<LayoutRef>> {
        /// A depth-first pre-order iterator over a layout.
        struct ChildrenIterator {
            stack: Vec<LayoutRef>,
        }

        impl Iterator for ChildrenIterator {
            type Item = VortexResult<LayoutRef>;

            fn next(&mut self) -> Option<Self::Item> {
                let next = self.stack.pop()?;
                let Ok(children) = next.children() else {
                    return Some(Ok(next));
                };
                for child in children.into_iter().rev() {
                    self.stack.push(child);
                }
                Some(Ok(next))
            }
        }

        ChildrenIterator {
            stack: vec![self.to_layout()],
        }
    }

    /// Display the layout as a tree structure.
    pub fn display_tree(&self) -> DisplayLayoutTree {
        DisplayLayoutTree::new(self.to_layout(), false)
    }

    /// Display the layout as a tree structure with optional verbose metadata.
    pub fn display_tree_verbose(&self, verbose: bool) -> DisplayLayoutTree {
        DisplayLayoutTree::new(self.to_layout(), verbose)
    }

    /// Display the layout as a tree structure, fetching segment buffer sizes from the segment source.
    ///
    /// # Warning
    ///
    /// This function performs IO to fetch each segment's buffer. For layouts with
    /// many segments, this may result in significant IO overhead.
    pub async fn display_tree_with_segments(
        &self,
        segment_source: Arc<dyn SegmentSource>,
    ) -> VortexResult<DisplayLayoutTree> {
        display_tree_with_segment_sizes(self.to_layout(), segment_source).await
    }
}

/// Display the encoding, dtype, row count, and segment IDs of this layout.
impl Display for dyn Layout + '_ {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let segment_ids = self.segment_ids();
        if segment_ids.is_empty() {
            write!(
                f,
                "{}({}, rows={})",
                self.encoding_id(),
                self.dtype(),
                self.row_count()
            )
        } else {
            write!(
                f,
                "{}({}, rows={}, segments=[{}])",
                self.encoding_id(),
                self.dtype(),
                self.row_count(),
                segment_ids
                    .iter()
                    .map(|s| format!("{}", **s))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}

#[repr(transparent)]
pub struct LayoutAdapter<V: VTable>(V::Layout);

impl<V: VTable> Debug for LayoutAdapter<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl<V: VTable> Layout for LayoutAdapter<V> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_arc(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn to_layout(&self) -> LayoutRef {
        Arc::new(LayoutAdapter::<V>(self.0.clone()))
    }

    fn encoding(&self) -> LayoutEncodingRef {
        V::encoding(&self.0)
    }

    fn row_count(&self) -> u64 {
        V::row_count(&self.0)
    }

    fn dtype(&self) -> &DType {
        V::dtype(&self.0)
    }

    fn nchildren(&self) -> usize {
        V::nchildren(&self.0)
    }

    fn child(&self, idx: usize) -> VortexResult<LayoutRef> {
        V::child(&self.0, idx)
    }

    fn child_type(&self, idx: usize) -> LayoutChildType {
        V::child_type(&self.0, idx)
    }

    fn metadata(&self) -> Vec<u8> {
        V::metadata(&self.0).serialize()
    }

    fn segment_ids(&self) -> Vec<SegmentId> {
        V::segment_ids(&self.0)
    }

    fn new_reader(
        &self,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
    ) -> VortexResult<LayoutReaderRef> {
        V::new_reader(&self.0, name, segment_source, session)
    }
}

mod private {
    use super::*;

    pub trait Sealed {}

    impl<V: VTable> Sealed for LayoutAdapter<V> {}
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_session::registry::ReadContext;

    use super::*;

    #[test]
    fn test_layout_child_type_name() {
        // Test Chunk variant
        let chunk = LayoutChildType::Chunk((5, 100));
        assert_eq!(chunk.name().as_ref(), "[5]");

        // Test Field variant
        let field = LayoutChildType::Field(FieldName::from("customer_id"));
        assert_eq!(field.name().as_ref(), "customer_id");

        // Test Auxiliary variant
        let aux = LayoutChildType::Auxiliary(Arc::from("zone_map"));
        assert_eq!(aux.name().as_ref(), "zone_map");

        // Test Transparent variant
        let transparent = LayoutChildType::Transparent(Arc::from("compressed"));
        assert_eq!(transparent.name().as_ref(), "compressed");
    }

    #[test]
    fn test_layout_child_type_row_offset() {
        // Chunk should return the offset
        let chunk = LayoutChildType::Chunk((0, 42));
        assert_eq!(chunk.row_offset(), Some(42));

        // Field should return 0
        let field = LayoutChildType::Field(FieldName::from("field1"));
        assert_eq!(field.row_offset(), Some(0));

        // Auxiliary should return None
        let aux = LayoutChildType::Auxiliary(Arc::from("metadata"));
        assert_eq!(aux.row_offset(), None);

        // Transparent should return 0
        let transparent = LayoutChildType::Transparent(Arc::from("wrapper"));
        assert_eq!(transparent.row_offset(), Some(0));
    }

    #[test]
    fn test_layout_child_type_equality() {
        // Test Chunk equality
        let chunk1 = LayoutChildType::Chunk((1, 100));
        let chunk2 = LayoutChildType::Chunk((1, 100));
        let chunk3 = LayoutChildType::Chunk((2, 100));
        let chunk4 = LayoutChildType::Chunk((1, 200));

        assert_eq!(chunk1, chunk2);
        assert_ne!(chunk1, chunk3);
        assert_ne!(chunk1, chunk4);

        // Test Field equality
        let field1 = LayoutChildType::Field(FieldName::from("name"));
        let field2 = LayoutChildType::Field(FieldName::from("name"));
        let field3 = LayoutChildType::Field(FieldName::from("age"));

        assert_eq!(field1, field2);
        assert_ne!(field1, field3);

        // Test Auxiliary equality
        let aux1 = LayoutChildType::Auxiliary(Arc::from("stats"));
        let aux2 = LayoutChildType::Auxiliary(Arc::from("stats"));
        let aux3 = LayoutChildType::Auxiliary(Arc::from("index"));

        assert_eq!(aux1, aux2);
        assert_ne!(aux1, aux3);

        // Test Transparent equality
        let trans1 = LayoutChildType::Transparent(Arc::from("enc"));
        let trans2 = LayoutChildType::Transparent(Arc::from("enc"));
        let trans3 = LayoutChildType::Transparent(Arc::from("dec"));

        assert_eq!(trans1, trans2);
        assert_ne!(trans1, trans3);

        // Test cross-variant inequality
        assert_ne!(chunk1, field1);
        assert_ne!(field1, aux1);
        assert_ne!(aux1, trans1);
    }

    #[rstest]
    #[case(LayoutChildType::Chunk((0, 0)), "[0]", Some(0))]
    #[case(LayoutChildType::Chunk((999, 1000000)), "[999]", Some(1000000))]
    #[case(LayoutChildType::Field(FieldName::from("")), "", Some(0))]
    #[case(
        LayoutChildType::Field(FieldName::from("very_long_field_name_that_is_quite_lengthy")),
        "very_long_field_name_that_is_quite_lengthy",
        Some(0)
    )]
    #[case(LayoutChildType::Auxiliary(Arc::from("aux")), "aux", None)]
    #[case(LayoutChildType::Transparent(Arc::from("t")), "t", Some(0))]
    fn test_layout_child_type_parameterized(
        #[case] child_type: LayoutChildType,
        #[case] expected_name: &str,
        #[case] expected_offset: Option<u64>,
    ) {
        assert_eq!(child_type.name().as_ref(), expected_name);
        assert_eq!(child_type.row_offset(), expected_offset);
    }

    #[test]
    fn test_chunk_with_different_indices_and_offsets() {
        let chunks = [
            LayoutChildType::Chunk((0, 0)),
            LayoutChildType::Chunk((1, 100)),
            LayoutChildType::Chunk((2, 200)),
            LayoutChildType::Chunk((100, 10000)),
        ];

        for chunk in chunks.iter() {
            let name = chunk.name();
            assert!(name.starts_with('['));
            assert!(name.ends_with(']'));

            if let LayoutChildType::Chunk((idx, offset)) = chunk {
                assert_eq!(name.as_ref(), format!("[{}]", idx));
                assert_eq!(chunk.row_offset(), Some(*offset));
            }
        }
    }

    #[test]
    fn test_field_names_with_special_characters() {
        let special_fields: Vec<Arc<str>> = vec![
            Arc::from("field-with-dashes"),
            Arc::from("field_with_underscores"),
            Arc::from("field.with.dots"),
            Arc::from("field::with::colons"),
            Arc::from("field/with/slashes"),
            Arc::from("field@with#symbols"),
        ];

        for field_name in special_fields {
            let field = LayoutChildType::Field(Arc::clone(&field_name).into());
            assert_eq!(field.name(), field_name);
            assert_eq!(field.row_offset(), Some(0));
        }
    }

    #[test]
    fn test_struct_layout_display() {
        use vortex_array::dtype::Nullability::NonNullable;
        use vortex_array::dtype::PType;
        use vortex_array::dtype::StructFields;

        use crate::IntoLayout;
        use crate::layouts::chunked::ChunkedLayout;
        use crate::layouts::dict::DictLayout;
        use crate::layouts::flat::FlatLayout;
        use crate::layouts::struct_::StructLayout;
        use crate::segments::SegmentId;

        let ctx = ReadContext::new([]);

        // Create a flat layout for dict values (utf8 strings)
        let dict_values =
            FlatLayout::new(3, DType::Utf8(NonNullable), SegmentId::from(0), ctx.clone())
                .into_layout();

        // Test flat layout display shows segment
        assert_eq!(
            format!("{}", dict_values),
            "vortex.flat(utf8, rows=3, segments=[0])"
        );

        // Create a flat layout for dict codes
        let dict_codes = FlatLayout::new(
            10,
            DType::Primitive(PType::U16, NonNullable),
            SegmentId::from(1),
            ctx.clone(),
        )
        .into_layout();

        // Test flat layout display shows segment
        assert_eq!(
            format!("{}", dict_codes),
            "vortex.flat(u16, rows=10, segments=[1])"
        );

        // Create dict layout (column "name")
        let dict_layout =
            DictLayout::new(Arc::clone(&dict_values), Arc::clone(&dict_codes)).into_layout();

        // Test dict layout display (no direct segments)
        assert_eq!(format!("{}", dict_layout), "vortex.dict(utf8, rows=10)");

        // Create flat layouts for chunks
        let chunk1 = FlatLayout::new(
            5,
            DType::Primitive(PType::I64, NonNullable),
            SegmentId::from(2),
            ctx.clone(),
        )
        .into_layout();

        let chunk2 = FlatLayout::new(
            5,
            DType::Primitive(PType::I64, NonNullable),
            SegmentId::from(3),
            ctx,
        )
        .into_layout();

        // Create chunked layout (column "value")
        let chunked_layout = ChunkedLayout::new(
            10,
            DType::Primitive(PType::I64, NonNullable),
            crate::OwnedLayoutChildren::layout_children(vec![
                Arc::clone(&chunk1),
                Arc::clone(&chunk2),
            ]),
        )
        .into_layout();

        // Test chunked layout display (no direct segments)
        assert_eq!(
            format!("{}", chunked_layout),
            "vortex.chunked(i64, rows=10)"
        );

        // Test chunk displays show segments
        assert_eq!(
            format!("{}", chunk1),
            "vortex.flat(i64, rows=5, segments=[2])"
        );
        assert_eq!(
            format!("{}", chunk2),
            "vortex.flat(i64, rows=5, segments=[3])"
        );

        // Create struct layout with two fields
        let field_names: Vec<Arc<str>> = vec!["name".into(), "value".into()];
        let struct_dtype = DType::Struct(
            StructFields::new(
                field_names.into(),
                vec![
                    DType::Utf8(NonNullable),
                    DType::Primitive(PType::I64, NonNullable),
                ],
            ),
            NonNullable,
        );

        let struct_layout =
            StructLayout::new(10, struct_dtype, vec![dict_layout, chunked_layout]).into_layout();

        println!("{}", struct_layout.display_tree_verbose(true));

        // Test Display impl for struct (no direct segments)
        assert_eq!(
            format!("{}", struct_layout),
            "vortex.struct({name=utf8, value=i64}, rows=10)"
        );
    }
}
