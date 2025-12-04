// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
mod writer;

use std::sync::Arc;

use geo::BoundingRect;
use geo_types::Geometry;
use geozero::GeozeroGeometry;
use geozero::geo_types::GeoWriter;
use geozero::wkb;
use rstar::AABB;
use rstar::RTreeObject;
use vortex_array::ArrayContext;
use vortex_array::DeserializeMetadata;
use vortex_array::SerializeMetadata;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::StructFields;
use vortex_dtype::TryFromBytes;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
pub use writer::*;

use crate::LayoutChildType;
use crate::LayoutChildren;
use crate::LayoutEncodingRef;
use crate::LayoutId;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::LazyReaderChildren;
use crate::VTable;
use crate::children::OwnedLayoutChildren;
use crate::layouts::rtree::reader::RTreeReader;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;

/// A layout that is based on the `ZonedLayout` but instead of storing array stats, it only
/// stores R-tree indexes as a struct child with a single binary column (holding the encoded
/// r-trees using bincode).
#[derive(Clone, Debug)]
pub struct RTreeLayout {
    dtype: DType,
    children: Arc<dyn LayoutChildren>,
    len: usize,
}

impl RTreeLayout {
    #[allow(clippy::panic)]
    pub fn new(data: LayoutRef, tree: LayoutRef, n_trees: usize) -> Self {
        assert!(n_trees > 0, "Number of r-trees must be greater than zero");
        assert_eq!(
            tree.dtype(),
            &expected_rtree_dtype(),
            "Invalid DType for rtree child"
        );

        // TODO(aduffy): check the DType for the r-tree table
        Self {
            dtype: data.dtype().clone(),
            children: OwnedLayoutChildren::layout_children(vec![data, tree]),
            len: n_trees,
        }
    }
}

// the whole RTree is very large.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct RTreeMetadata {
    pub(super) len: u32,
}

impl DeserializeMetadata for RTreeMetadata {
    type Output = Self;

    fn deserialize(metadata: &[u8]) -> VortexResult<Self::Output> {
        let len = u32::try_from_le_bytes(&metadata[0..4])?;
        Ok(Self { len })
    }
}

impl SerializeMetadata for RTreeMetadata {
    fn serialize(self) -> Vec<u8> {
        let mut output = Vec::with_capacity(4);
        output.extend_from_slice(self.len.to_le_bytes().as_slice());
        output
    }
}

#[derive(Debug)]
pub struct RTreeLayoutEncoding;

vtable!(RTree);

impl VTable for RTreeVTable {
    type Layout = RTreeLayout;
    type Encoding = RTreeLayoutEncoding;
    type Metadata = RTreeMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new_ref("vortex.geo.rtree")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(RTreeLayoutEncoding.as_ref())
    }

    fn row_count(layout: &Self::Layout) -> u64 {
        // 0-th child is the data
        layout.children.child_row_count(0)
    }

    fn dtype(layout: &Self::Layout) -> &DType {
        &layout.dtype
    }

    #[allow(clippy::cast_possible_truncation)]
    fn metadata(layout: &Self::Layout) -> Self::Metadata {
        RTreeMetadata {
            len: layout.len as u32,
        }
    }

    fn segment_ids(_layout: &Self::Layout) -> Vec<SegmentId> {
        vec![]
    }

    fn nchildren(_layout: &Self::Layout) -> usize {
        2
    }

    fn child(layout: &Self::Layout, idx: usize) -> VortexResult<LayoutRef> {
        match idx {
            0 => layout.children.child(0, layout.dtype()),
            1 => layout.children.child(1, &expected_rtree_dtype()),
            _ => vortex_bail!("Invalid child index for RTreeLayout {idx}"),
        }
    }

    fn child_type(_layout: &Self::Layout, idx: usize) -> LayoutChildType {
        match idx {
            0 => LayoutChildType::Transparent(Arc::from("data")),
            1 => LayoutChildType::Auxiliary(Arc::from("rtree")),
            _ => unreachable!("invalid child index for RTreeLayout"),
        }
    }

    fn new_reader(
        layout: &RTreeLayout,
        name: Arc<str>,
        segment_source: Arc<dyn SegmentSource>,
        session: &VortexSession,
    ) -> VortexResult<LayoutReaderRef> {
        let names = vec![Arc::from(format!("{name}.data")), Arc::from("{name}.rtree")];
        let children = LazyReaderChildren::new(
            layout.children.clone(),
            vec![layout.dtype.clone(), expected_rtree_dtype()],
            names,
            segment_source,
            session.clone(),
        );

        Ok(Arc::new(RTreeReader {
            name,
            layout: layout.clone(),
            children,
        }))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        _row_count: u64,
        metadata: &RTreeMetadata,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _ctx: ArrayContext,
    ) -> VortexResult<Self::Layout> {
        Ok(RTreeLayout {
            len: metadata.len as usize,
            dtype: dtype.clone(),
            children: children.to_arc(),
        })
    }
}

/// Wrapper around the `geo_types::Geometry` so we can implement the `RTreeObject` trait on it.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct GeometryObject(Geometry<f64>);

impl RTreeObject for GeometryObject {
    type Envelope = AABB<[f64; 2]>;

    // TODO(aduffy): use CachedEnvelope to cache bbox calculation, this will probably
    //  be expensive otherwise
    fn envelope(&self) -> Self::Envelope {
        let bbox = self.0.bounding_rect().expect("bbox from geometry");
        let min = bbox.min();
        let max = bbox.max();
        AABB::from_corners([min.x, min.y], [max.x, max.y])
    }
}

/// Make a `GeometryObject` from a WKB-encoded geometry value.
pub(crate) fn make_geom(wkb: &[u8]) -> Option<GeometryObject> {
    let mut geo = GeoWriter::new();
    wkb::Wkb(wkb).process_geom(&mut geo).unwrap();
    geo.take_geometry().map(GeometryObject)
}

/// Expected type of the rtree array
pub(crate) fn expected_rtree_dtype() -> DType {
    DType::Struct(
        StructFields::from_iter([("rtree", DType::Binary(Nullability::Nullable))]),
        Nullability::NonNullable,
    )
}
