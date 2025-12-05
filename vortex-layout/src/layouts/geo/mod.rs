// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod reader;
mod writer;

use std::sync::{Arc, OnceLock};

use fastbloom::BloomFilter;
use futures::future::BoxFuture;
use futures::future::Shared;
use geo::ConvexHull;
use geo_types::Geometry;
use geozero::geo_types::GeoWriter;
use geozero::wkb;
use geozero::GeozeroGeometry;
use h3o::geom::ContainmentMode;
use h3o::geom::TilerBuilder;
use h3o::Resolution;
use itertools::Itertools;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::VarBinVTable;
use vortex_array::Array;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::DeserializeMetadata;
use vortex_array::SerializeMetadata;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_dtype::StructFields;
use vortex_dtype::TryFromBytes;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_error::SharedVortexResult;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
pub use writer::*;

use crate::children::OwnedLayoutChildren;
use crate::layouts::geo::reader::GeoReader;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::vtable;
use crate::LayoutChildType;
use crate::LayoutChildren;
use crate::LayoutEncodingRef;
use crate::LayoutId;
use crate::LayoutReaderRef;
use crate::LayoutRef;
use crate::LazyReaderChildren;
use crate::VTable;

/// A layout that stores a bloom filter of H3 tile IDs.
#[derive(Clone, Debug)]
pub struct GeoLayout {
    dtype: DType,
    children: Arc<dyn LayoutChildren>,
    /// How many rows are in a zone (except the last zone which might be shorter)
    zone_len: usize,
}

impl GeoLayout {
    #[allow(clippy::panic)]
    pub fn new(data: LayoutRef, tree: LayoutRef, zone_len: usize) -> Self {
        assert!(zone_len > 0);
        assert_eq!(
            tree.dtype(),
            &expected_rtree_dtype(),
            "Invalid DType for filters child"
        );

        // TODO(aduffy): check the DType for the r-tree table
        Self {
            dtype: data.dtype().clone(),
            children: OwnedLayoutChildren::layout_children(vec![data, tree]),
            zone_len,
        }
    }

    pub fn nzones(&self) -> usize {
        usize::try_from(self.children.child_row_count(1)).vortex_expect("Invalid number of zones")
    }
}

// the whole RTree is very large.
#[derive(Debug, PartialEq, Eq, Clone)]
pub struct GeoMetadata {
    pub(super) zone_len: u32,
}

impl DeserializeMetadata for GeoMetadata {
    type Output = Self;

    fn deserialize(metadata: &[u8]) -> VortexResult<Self::Output> {
        let zone_len = u32::try_from_le_bytes(&metadata[0..4])?;
        Ok(Self { zone_len })
    }
}

impl SerializeMetadata for GeoMetadata {
    fn serialize(self) -> Vec<u8> {
        let mut output = Vec::with_capacity(4);
        output.extend_from_slice(self.zone_len.to_le_bytes().as_slice());
        output
    }
}

#[derive(Debug)]
pub struct GeoLayoutEncoding;

vtable!(Geo);

impl VTable for GeoVTable {
    type Layout = GeoLayout;
    type Encoding = GeoLayoutEncoding;
    type Metadata = GeoMetadata;

    fn id(_encoding: &Self::Encoding) -> LayoutId {
        LayoutId::new_ref("vortex.geo.rtree")
    }

    fn encoding(_layout: &Self::Layout) -> LayoutEncodingRef {
        LayoutEncodingRef::new_ref(GeoLayoutEncoding.as_ref())
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
        GeoMetadata {
            zone_len: layout.zone_len as u32,
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
        layout: &GeoLayout,
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

        Ok(Arc::new(GeoReader {
            name,
            layout: layout.clone(),
            children,
            geo_filter: OnceLock::new(),
        }))
    }

    fn build(
        _encoding: &Self::Encoding,
        dtype: &DType,
        _row_count: u64,
        metadata: &GeoMetadata,
        _segment_ids: Vec<SegmentId>,
        children: &dyn LayoutChildren,
        _ctx: ArrayContext,
    ) -> VortexResult<Self::Layout> {
        Ok(GeoLayout {
            zone_len: metadata.zone_len as usize,
            dtype: dtype.clone(),
            children: children.to_arc(),
        })
    }
}

/// Make a `GeometryObject` from a WKB-encoded geometry value.
pub(crate) fn make_geom(wkb: &[u8]) -> Option<Geometry> {
    let mut geo = GeoWriter::new();
    wkb::Wkb(wkb).process_geom(&mut geo).unwrap();
    geo.take_geometry()
}

/// Expected type of the rtree array
pub(crate) fn expected_rtree_dtype() -> DType {
    DType::Struct(
        StructFields::from_iter([("rtree", DType::Binary(Nullability::Nullable))]),
        Nullability::NonNullable,
    )
}

/// A filter over geospatial data that allows for efficient pruning of intersection and containment
/// queries.
///
/// Physically, it is a set of zone-level bloom filters over the H3 cell IDs contained within the
/// chunk, up to a certain resolution level.
///
/// TODO(aduffy): make the cell ID resolution adaptive based on the chunk overview.
#[derive(Clone)]
pub(crate) struct GeoFilter {
    inner: Vec<BloomFilter>,
}

impl GeoFilter {
    pub fn try_load(array: ArrayRef) -> VortexResult<Self> {
        // What is the purpose of this large set of files?
        let Canonical::Struct(struct_array) = array.to_canonical() else {
            vortex_bail!("expected StructArray from GeoFilter layout");
        };

        let filters_col = struct_array.fields()[0].clone();
        // TODO(aduffy): this is dumb and we shouldn't actually force this, but for now we
        //  force to VarBin encoding b/c VarBinView is strictly larger.
        let filters_col_bin = filters_col.as_::<VarBinVTable>();
        let decoded_filters: Vec<BloomFilter> = filters_col_bin.with_iterator(|binary| {
            binary
                .map(|b| match b {
                    None => VortexResult::Ok(BloomFilter::from_vec(vec![0]).expected_items(0)),
                    Some(v) => {
                        let (filter, _) =
                            bincode::serde::decode_from_slice(v, bincode::config::standard())
                                .map_err(|e| vortex_err!("failed to decode BloomFilter: {e}"))?;
                        VortexResult::Ok(filter)
                    }
                })
                .try_collect()
        })?;

        Ok(Self {
            inner: decoded_filters,
        })
    }

    /// Probe the GeoFilter to see if the given zone *may* contain a geometry that is covered
    /// by the given geometry.
    pub fn filter_contains(&self, zone_id: usize, geom: &Geometry) -> bool {
        let filter = &self.inner[zone_id];
        // Generate the cell IDs for the geometry.
        // TODO(aduffy): store this config centrally so the read/write paths don't drift.
        let mut tiler = TilerBuilder::new(Resolution::Eight)
            .containment_mode(ContainmentMode::Covers)
            .build();
        tiler
            .add(geom.convex_hull())
            .unwrap_or_else(|e| vortex_panic!("failed to add polygon to tiler: {e}"));

        // If some cells of the geometry are contained by this zone, then it's possible that there
        // may be some geometry in this zone that would pass a query `ST_CONTAINS(geom, zone_geometry)`.
        for cell_id in tiler.into_coverage() {
            let cell_id = u64::from(cell_id);
            if filter.contains_hash(cell_id) {
                return true;
            }
        }

        false
    }
}

type SharedGeoFilter = Shared<BoxFuture<'static, SharedVortexResult<GeoFilter>>>;
