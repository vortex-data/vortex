// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod coordinate;
mod linestring;
mod multilinestring;
mod multipoint;
mod multipolygon;
mod point;
mod polygon;
mod rect;
mod wkb;

use std::fmt::Display;
use std::sync::Arc;

use ::wkb::reader::GeometryType;
use arrow_array::BinaryArray;
pub use coordinate::Dimension;
use geo_types::Geometry;
use geoarrow::array::GenericWkbArray;
use geoarrow::array::GeoArrowArray;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::Dimension as GeoArrowDimension;
use geoarrow::datatypes::GeoArrowType;
use geoarrow::datatypes::LineStringType;
use geoarrow::datatypes::Metadata;
use geoarrow::datatypes::MultiLineStringType;
use geoarrow::datatypes::MultiPointType;
use geoarrow::datatypes::MultiPolygonType;
use geoarrow::datatypes::PointType;
use geoarrow::datatypes::PolygonType;
use geoarrow::datatypes::WkbType;
use geoarrow_cast::cast::cast;
pub use linestring::*;
pub use multilinestring::*;
pub use multipoint::*;
pub use multipolygon::*;
pub use point::*;
pub use polygon::*;
use prost::Message;
pub use rect::*;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::ListViewArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::list::ListArraySlotsExt;
use vortex_array::arrays::listview::ListViewArraySlotsExt;
use vortex_array::arrays::listview::list_from_list_view;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::Scalar;
use vortex_arrow::ArrowSession;
use vortex_arrow::ArrowSessionExt;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
pub use wkb::*;

/// Whether `dtype` is one of the native geometry extension types the spatial kernels operate on.
pub(crate) fn is_native_geometry(dtype: &DType) -> bool {
    dtype.as_extension_opt().is_some_and(|ext| {
        ext.is::<Point>()
            || ext.is::<LineString>()
            || ext.is::<MultiPoint>()
            || ext.is::<Polygon>()
            || ext.is::<MultiLineString>()
            || ext.is::<MultiPolygon>()
            || ext.is::<Rect>()
    })
}

/// Flatten a native geometry column into a single coordinate `Struct<x, y, ...>` containing
/// every vertex of every geometry.
pub(crate) fn flatten_coordinates(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<StructArray> {
    if !is_native_geometry(array.dtype()) {
        vortex_bail!(
            "spatial: operand is not a native geometry extension type, was {}",
            array.dtype()
        );
    }
    let mut node = array
        .clone()
        .execute::<ExtensionArray>(ctx)?
        .storage_array()
        .clone();
    while node.dtype().is_list() {
        node = node.execute::<ListViewArray>(ctx)?.elements().clone();
    }
    node.execute::<StructArray>(ctx)
}

/// Flatten a native geometry `storage` array to its leaf coordinates, keeping track of which
/// row owns which coordinates.
///
/// Returns the flat coordinate `Struct` plus `row_offsets` (one entry per row plus a final
/// cap): row `r`'s coordinates are `coordinates[row_offsets[r]..row_offsets[r + 1]]`, an empty
/// range if the row has none.
///
/// Row boundaries are pushed down one `List` level at a time. A boundary at list `e` moves to
/// `offsets[e]`, where that list's children start. For example, a 2-row `MultiPolygon` column —
/// row 0 = one polygon of two rings (3 + 4 vertices), row 1 = one polygon of one ring (5
/// vertices):
///
/// ```text
/// level                offsets       row_offsets after the level
/// rows     → polygons  [0,1,2]       [0,1,2]    row 1 starts at polygon 1
/// polygons → rings     [0,2,3]       [0,2,3]    row 1 starts at ring 2
/// rings    → vertices  [0,3,7,12]    [0,7,12]   row 1 starts at vertex 7
/// ```
pub(crate) fn flatten_row_offsets(
    storage: ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<(Vec<usize>, StructArray)> {
    let len = storage.len();

    // At the outermost level, row `r` starts at element `r`; the extra entry caps the last row.
    let mut row_offsets: Vec<usize> = (0..=len).collect();
    let mut level = storage;
    while level.dtype().is_list() {
        let list = list_from_list_view(level.execute::<ListViewArray>(ctx)?, ctx)?;
        let offsets = list
            .offsets()
            .clone()
            .cast(DType::Primitive(PType::U64, Nullability::NonNullable))?
            .execute::<Buffer<u64>>(ctx)?;
        for row_offset in &mut row_offsets {
            *row_offset = usize::try_from(offsets[*row_offset])
                .map_err(|_| vortex_err!("spatial: list offset exceeds usize"))?;
        }
        level = list.elements().clone();
    }
    Ok((row_offsets, level.execute::<StructArray>(ctx)?))
}

/// Decode a native geometry column to `geo_types`. A non-geometry operand is an error.
pub(crate) fn geometries(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Geometry<f64>>> {
    let Some(ext) = array.dtype().as_extension_opt() else {
        vortex_bail!(
            "spatial: operand is not a geometry extension type, was {}",
            array.dtype()
        );
    };
    let storage = array
        .clone()
        .execute::<ExtensionArray>(ctx)?
        .storage_array()
        .clone();
    if ext.is::<Point>() {
        point_geometries(&storage, ctx)
    } else if ext.is::<LineString>() {
        linestring_geometries(&storage, ctx)
    } else if ext.is::<MultiPoint>() {
        multipoint_geometries(&storage, ctx)
    } else if ext.is::<Polygon>() {
        polygon_geometries(&storage, ctx)
    } else if ext.is::<MultiLineString>() {
        multilinestring_geometries(&storage, ctx)
    } else if ext.is::<MultiPolygon>() {
        multipolygon_geometries(&storage, ctx)
    } else if ext.is::<Rect>() {
        rect_geometries(&storage, ctx)
    } else {
        vortex_bail!("spatial: unsupported geometry extension {}", array.dtype())
    }
}

/// Decode a constant operand scalar to one geometry, a constant of any
/// supported geometry type is decoded exactly like a column.
pub(crate) fn single_geometry(
    scalar: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Geometry<f64>> {
    let array = ConstantArray::new(scalar.clone(), 1).into_array();
    geometries(&array, ctx)?
        .pop()
        .ok_or_else(|| vortex_err!("spatial: constant operand decoded to no geometry"))
}

/// Decode a WKB geometry literal (DuckDB's wire form for `GEOMETRY` constants) to its native
/// `Point`/`Polygon`/`MultiPolygon` scalar. `None` for unsupported types. Plan-time, one value only.
pub fn native_geometry_scalar_from_wkb(
    bytes: &[u8],
    session: &VortexSession,
) -> VortexResult<Option<Scalar>> {
    let metadata = geoarrow_metadata(&SpatialMetadata::default());
    let binary = BinaryArray::from(vec![Some(bytes)]);
    let wkb = GenericWkbArray::<i32>::try_from((
        &binary as &dyn arrow_array::Array,
        WkbType::new(Arc::clone(&metadata)),
    ))
    .map_err(|e| vortex_err!("failed to read WKB literal: {e}"))?;

    // Cast the WKB value to `target`, import its native storage as a Vortex array.
    let to_storage = |target: &GeoArrowType| -> VortexResult<ArrayRef> {
        let native =
            cast(&wkb, target).map_err(|e| vortex_err!("failed to cast WKB literal: {e}"))?;
        session
            .arrow()
            .from_arrow_array(native.to_array_ref(), false)
    };

    let scalar = match Wkb::try_from_bytes(bytes)?.geometry_type() {
        GeometryType::Point => {
            let target = GeoArrowType::Point(
                PointType::new(GeoArrowDimension::XY, metadata)
                    .with_coord_type(CoordType::Separated),
            );
            spatial_ext_scalar(Point, to_storage(&target)?, session)?
        }
        GeometryType::LineString => {
            let target = GeoArrowType::LineString(
                LineStringType::new(GeoArrowDimension::XY, metadata)
                    .with_coord_type(CoordType::Separated),
            );
            spatial_ext_scalar(LineString, to_storage(&target)?, session)?
        }
        GeometryType::Polygon => {
            let target = GeoArrowType::Polygon(
                PolygonType::new(GeoArrowDimension::XY, metadata)
                    .with_coord_type(CoordType::Separated),
            );
            spatial_ext_scalar(Polygon, to_storage(&target)?, session)?
        }
        GeometryType::MultiPoint => {
            let target = GeoArrowType::MultiPoint(
                MultiPointType::new(GeoArrowDimension::XY, metadata)
                    .with_coord_type(CoordType::Separated),
            );
            spatial_ext_scalar(MultiPoint, to_storage(&target)?, session)?
        }
        GeometryType::MultiLineString => {
            let target = GeoArrowType::MultiLineString(
                MultiLineStringType::new(GeoArrowDimension::XY, metadata)
                    .with_coord_type(CoordType::Separated),
            );
            spatial_ext_scalar(MultiLineString, to_storage(&target)?, session)?
        }
        GeometryType::MultiPolygon => {
            let target = GeoArrowType::MultiPolygon(
                MultiPolygonType::new(GeoArrowDimension::XY, metadata)
                    .with_coord_type(CoordType::Separated),
            );
            spatial_ext_scalar(MultiPolygon, to_storage(&target)?, session)?
        }
        _ => return Ok(None),
    };
    Ok(Some(scalar))
}

/// Wrap cast-from-WKB `storage`, which has XY coordinates, in its `vtable` extension type and pull
/// out the single scalar.
fn spatial_ext_scalar<V: ExtVTable<Metadata = SpatialMetadata>>(
    vtable: V,
    storage: ArrayRef,
    session: &VortexSession,
) -> VortexResult<Scalar> {
    let metadata = SpatialMetadata::default().with_dimension(Dimension::Xy);
    let ext = ExtDType::try_with_vtable(vtable, metadata, storage.dtype().nullability())?.erased();
    ExtensionArray::try_new(ext, storage)?
        .into_array()
        .execute_scalar(0, &mut session.create_execution_ctx())
}

/// Extension metadata that is common to all the spatial extension types.
///
/// The serialized metadata holds just the coordinate reference system (CRS). We may wish to add a
/// second field for edges interpretation in the future similar to the GeoArrow standard.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct SpatialMetadata {
    pub(crate) crs: Option<String>,
    /// The coordinate dimension of a native geometry type, and `None` for [`WellKnownBinary`].
    ///
    /// It is not serialized, because the storage dtype holds it. A read takes it from the stored
    /// storage dtype.
    pub(crate) dimension: Option<Dimension>,
}

/// The serialized form of [`SpatialMetadata`].
#[derive(Clone, PartialEq, prost::Message)]
struct SpatialMetadataProto {
    #[prost(optional, string, tag = "1")]
    crs: Option<String>,
}

impl SpatialMetadata {
    /// Creates metadata with an optional coordinate reference system (CRS) and no coordinate
    /// dimension, as [`WellKnownBinary`] requires.
    pub fn new(crs: Option<String>) -> Self {
        Self {
            crs,
            dimension: None,
        }
    }

    /// Returns this metadata with the coordinate dimension that a native geometry type requires.
    pub fn with_dimension(self, dimension: Dimension) -> Self {
        Self {
            dimension: Some(dimension),
            ..self
        }
    }

    /// Returns the coordinate reference system, or `None` for unreferenced geometry.
    pub fn get_crs(&self) -> Option<&str> {
        self.crs.as_deref()
    }

    /// Serializes the CRS, the only part that the storage dtype does not hold.
    pub(crate) fn serialize(&self) -> Vec<u8> {
        SpatialMetadataProto {
            crs: self.crs.clone(),
        }
        .encode_to_vec()
    }

    /// Deserializes the CRS and adds `dimension`, which the stored storage dtype holds.
    pub(crate) fn deserialize(bytes: &[u8], dimension: Option<Dimension>) -> VortexResult<Self> {
        let crs = SpatialMetadataProto::decode(bytes)?.crs;
        Ok(Self { crs, dimension })
    }

    /// Checks that the metadata of the native geometry type `id` has a coordinate dimension.
    pub(crate) fn validate_native(&self, id: ExtId) -> VortexResult<()> {
        vortex_ensure!(
            self.dimension.is_some(),
            "{id} metadata must have a coordinate dimension, got none"
        );

        Ok(())
    }

    /// Returns the coordinate dimension of a native geometry type.
    ///
    /// # Panics
    ///
    /// Panics if there is none, which [`validate_native`](Self::validate_native) rejects.
    pub(crate) fn native_dimension(&self) -> Dimension {
        self.dimension
            .vortex_expect("validate_native checks that native geometry metadata has a dimension")
    }
}

impl Display for SpatialMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.crs.as_ref() {
            Some(crs) => write!(f, "Geometry(crs={crs})"),
            None => write!(f, "Geometry(unreferenced)"),
        }
    }
}

/// The GeoArrow [`Metadata`] equivalent of `spatial_metadata`.
pub(crate) fn geoarrow_metadata(spatial_metadata: &SpatialMetadata) -> Arc<Metadata> {
    Arc::new(Metadata::new(
        spatial_metadata
            .crs
            .as_ref()
            .map(|crs| Crs::from_unknown_crs_type(crs.to_string()))
            .unwrap_or_default(),
        None,
    ))
}

/// Serialize a native geometry array to WKB (a `WkbView` array) via geoarrow's cast.
/// Shared by the `to_wkb` methods on the geometry extension types.
pub(crate) fn geoarrow_to_wkb(
    geoarrow_array: &dyn GeoArrowArray,
    session: &ArrowSession,
) -> VortexResult<ArrayRef> {
    let wkb_type =
        GeoArrowType::WkbView(WkbType::new(geoarrow_metadata(&SpatialMetadata::default())));
    let wkb = cast(geoarrow_array, &wkb_type)
        .map_err(|e| vortex_err!("failed to cast geometry to WKB: {e}"))?;
    session.from_arrow_array(wkb.to_array_ref(), false)
}

/// Recover [`SpatialMetadata`] from GeoArrow metadata.
pub(crate) fn spatial_metadata_from_arrow(metadata: &Metadata) -> SpatialMetadata {
    let crs = metadata.crs().crs_value().map(|value| {
        // `Crs::from_unknown_crs_type` stores the user's string verbatim as a JSON string
        // value, so prefer the raw string when available to round-trip cleanly. For other
        // CRS encodings (PROJJSON object, etc.), fall back to the JSON-encoded form.
        value
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| value.to_string())
    });
    SpatialMetadata::new(crs)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::dtype::extension::ExtDTypePlugin;
    use vortex_array::dtype::extension::ExtVTable;
    use vortex_array::dtype::proto::dtype as pb;
    use vortex_array::dtype::proto::dtype::d_type::DtypeType;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;

    use super::LineString;
    use super::MultiLineString;
    use super::MultiPoint;
    use super::MultiPolygon;
    use super::Point;
    use super::Polygon;
    use super::Rect;
    use super::WellKnownBinary;
    use crate::extension::SpatialMetadata;
    use crate::extension::coordinate::Dimension;
    use crate::extension::coordinate::coordinate_storage_dtype;
    use crate::test_harness::spatial_session;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// The metadata of an unreferenced geometry with XY coordinates.
    fn xy_metadata() -> SpatialMetadata {
        SpatialMetadata::default().with_dimension(Dimension::Xy)
    }

    /// Test shim: decode with an explicitly constructed session.
    fn native_geometry_scalar_from_wkb(
        bytes: &[u8],
    ) -> VortexResult<Option<vortex_array::scalar::Scalar>> {
        super::native_geometry_scalar_from_wkb(bytes, &VortexSession::empty())
    }

    #[test]
    fn test_metadata() -> VortexResult<()> {
        let meta = SpatialMetadata::new(Some("EPSG:4326".to_string()));
        assert_eq!(meta.to_string(), "Geometry(crs=EPSG:4326)");

        // Files store the protobuf encoding of the CRS as field 1, so these bytes must not change:
        // the key `0x0a`, the length 9, and the CRS.
        let bytes = meta.serialize();
        assert_eq!(bytes, b"\x0a\x09EPSG:4326");

        let decoded = SpatialMetadata::deserialize(&bytes, None)?;
        assert_eq!(decoded, meta);

        Ok(())
    }

    /// [`ExtDType::try_with_vtable`] validates the metadata before it asks for the storage
    /// declaration, so a native geometry type without a coordinate dimension returns an error
    /// instead of a panic. WKB metadata with a coordinate dimension is also invalid.
    #[rstest]
    #[case::point(
        Point,
        SpatialMetadata::default(),
        "vortex.st.point metadata must have a coordinate dimension, got none"
    )]
    #[case::linestring(
        LineString,
        SpatialMetadata::default(),
        "vortex.st.linestring metadata must have a coordinate dimension, got none"
    )]
    #[case::polygon(
        Polygon,
        SpatialMetadata::default(),
        "vortex.st.polygon metadata must have a coordinate dimension, got none"
    )]
    #[case::multipoint(
        MultiPoint,
        SpatialMetadata::default(),
        "vortex.st.multipoint metadata must have a coordinate dimension, got none"
    )]
    #[case::multilinestring(
        MultiLineString,
        SpatialMetadata::default(),
        "vortex.st.multilinestring metadata must have a coordinate dimension, got none"
    )]
    #[case::multipolygon(
        MultiPolygon,
        SpatialMetadata::default(),
        "vortex.st.multipolygon metadata must have a coordinate dimension, got none"
    )]
    #[case::rect(
        Rect,
        SpatialMetadata::default(),
        "vortex.st.box metadata must have a coordinate dimension, got none"
    )]
    #[case::wkb(
        WellKnownBinary,
        xy_metadata(),
        "vortex.st.wkb metadata must not have a coordinate dimension, got Xy"
    )]
    fn invalid_dimension_returns_error(
        #[case] vtable: impl ExtVTable<Metadata = SpatialMetadata>,
        #[case] metadata: SpatialMetadata,
        #[case] expected: &str,
    ) {
        let result = ExtDType::try_with_vtable(vtable, metadata, Nullability::NonNullable);
        assert!(result.is_err_and(|error| error.to_string().contains(expected)));
    }

    /// A little-endian WKB `POINT` literal decodes to the native `Point` extension scalar.
    #[test]
    fn decodes_wkb_point_to_native() -> VortexResult<()> {
        let mut wkb = vec![1u8]; // little-endian byte order
        wkb.extend_from_slice(&1u32.to_le_bytes()); // geometry type: point
        wkb.extend_from_slice(&1.0f64.to_le_bytes()); // x
        wkb.extend_from_slice(&2.0f64.to_le_bytes()); // y

        let scalar = native_geometry_scalar_from_wkb(&wkb)?.expect("a point scalar");
        let DType::Extension(ext) = scalar.dtype() else {
            panic!("expected an extension dtype, got {}", scalar.dtype());
        };
        assert!(ext.is::<Point>());
        assert_eq!(ext.metadata::<Point>(), &xy_metadata());
        let mut ctx = SESSION.create_execution_ctx();
        let expected =
            crate::test_harness::point_column(vec![1.0], vec![2.0])?.execute_scalar(0, &mut ctx)?;
        let actual_storage = scalar.as_extension().to_storage_scalar();
        let expected_storage = expected.as_extension().to_storage_scalar();
        assert_eq!(actual_storage.dtype(), expected_storage.dtype());
        assert_eq!(actual_storage, expected_storage);
        Ok(())
    }

    /// A little-endian WKB `POLYGON` literal decodes to the native `Polygon` extension scalar.
    #[test]
    fn decodes_wkb_polygon_to_native() -> VortexResult<()> {
        let ring = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.0, 0.0)];
        let mut wkb = vec![1u8]; // little-endian byte order
        wkb.extend_from_slice(&3u32.to_le_bytes()); // geometry type: polygon
        wkb.extend_from_slice(&1u32.to_le_bytes()); // one ring
        let ring_len = u32::try_from(ring.len()).map_err(|e| vortex_err!("{e}"))?;
        wkb.extend_from_slice(&ring_len.to_le_bytes());
        for (x, y) in ring {
            wkb.extend_from_slice(&f64::to_le_bytes(x));
            wkb.extend_from_slice(&f64::to_le_bytes(y));
        }

        let scalar = native_geometry_scalar_from_wkb(&wkb)?.expect("a polygon scalar");
        let DType::Extension(ext) = scalar.dtype() else {
            panic!("expected an extension dtype, got {}", scalar.dtype());
        };
        assert!(ext.is::<Polygon>());
        assert_eq!(ext.metadata::<Polygon>(), &xy_metadata());
        let mut ctx = SESSION.create_execution_ctx();
        let expected = crate::test_harness::polygon_column(vec![vec![vec![
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.0, 0.0),
        ]]])?
        .execute_scalar(0, &mut ctx)?;
        let actual_storage = scalar.as_extension().to_storage_scalar();
        let expected_storage = expected.as_extension().to_storage_scalar();
        assert_eq!(actual_storage.dtype(), expected_storage.dtype());
        assert_eq!(actual_storage, expected_storage);
        Ok(())
    }

    /// A little-endian WKB `LINESTRING` literal decodes to the native `LineString` extension scalar.
    #[test]
    fn decodes_wkb_linestring_to_native() -> VortexResult<()> {
        let points = [(0.0, 0.0), (1.0, 1.0)];
        let mut wkb = vec![1u8]; // little-endian byte order
        wkb.extend_from_slice(&2u32.to_le_bytes()); // geometry type: linestring
        let len = u32::try_from(points.len()).map_err(|e| vortex_err!("{e}"))?;
        wkb.extend_from_slice(&len.to_le_bytes());
        for (x, y) in points {
            wkb.extend_from_slice(&f64::to_le_bytes(x));
            wkb.extend_from_slice(&f64::to_le_bytes(y));
        }

        let scalar = native_geometry_scalar_from_wkb(&wkb)?.expect("a linestring scalar");
        let DType::Extension(ext) = scalar.dtype() else {
            panic!("expected an extension dtype, got {}", scalar.dtype());
        };
        assert!(ext.is::<LineString>());
        assert_eq!(ext.metadata::<LineString>(), &xy_metadata());
        let mut ctx = SESSION.create_execution_ctx();
        let expected = crate::test_harness::linestring_column(vec![vec![(0.0, 0.0), (1.0, 1.0)]])?
            .execute_scalar(0, &mut ctx)?;
        let actual_storage = scalar.as_extension().to_storage_scalar();
        let expected_storage = expected.as_extension().to_storage_scalar();
        assert_eq!(actual_storage.dtype(), expected_storage.dtype());
        assert_eq!(actual_storage, expected_storage);
        Ok(())
    }

    /// A little-endian WKB `MULTIPOINT` literal decodes to the native `MultiPoint` extension scalar.
    #[test]
    fn decodes_wkb_multipoint_to_native() -> VortexResult<()> {
        let points = [(0.0, 0.0), (1.0, 1.0)];
        let mut wkb = vec![1u8]; // little-endian byte order
        wkb.extend_from_slice(&4u32.to_le_bytes()); // geometry type: multipoint
        let len = u32::try_from(points.len()).map_err(|e| vortex_err!("{e}"))?;
        wkb.extend_from_slice(&len.to_le_bytes());
        for (x, y) in points {
            // each member is a full WKB point
            wkb.push(1u8);
            wkb.extend_from_slice(&1u32.to_le_bytes());
            wkb.extend_from_slice(&f64::to_le_bytes(x));
            wkb.extend_from_slice(&f64::to_le_bytes(y));
        }

        let scalar = native_geometry_scalar_from_wkb(&wkb)?.expect("a multipoint scalar");
        let DType::Extension(ext) = scalar.dtype() else {
            panic!("expected an extension dtype, got {}", scalar.dtype());
        };
        assert!(ext.is::<MultiPoint>());
        assert_eq!(ext.metadata::<MultiPoint>(), &xy_metadata());
        let mut ctx = SESSION.create_execution_ctx();
        let expected = crate::test_harness::multipoint_column(vec![vec![(0.0, 0.0), (1.0, 1.0)]])?
            .execute_scalar(0, &mut ctx)?;
        let actual_storage = scalar.as_extension().to_storage_scalar();
        let expected_storage = expected.as_extension().to_storage_scalar();
        assert_eq!(actual_storage.dtype(), expected_storage.dtype());
        assert_eq!(actual_storage, expected_storage);
        Ok(())
    }

    /// A little-endian WKB `MULTILINESTRING` literal decodes to the native `MultiLineString` scalar.
    #[test]
    fn decodes_wkb_multilinestring_to_native() -> VortexResult<()> {
        let lines = [[(0.0, 0.0), (1.0, 1.0)], [(2.0, 2.0), (3.0, 3.0)]];
        let mut wkb = vec![1u8]; // little-endian byte order
        wkb.extend_from_slice(&5u32.to_le_bytes()); // geometry type: multilinestring
        let num_lines = u32::try_from(lines.len()).map_err(|e| vortex_err!("{e}"))?;
        wkb.extend_from_slice(&num_lines.to_le_bytes());
        for line in lines {
            // each member is a full WKB linestring
            wkb.push(1u8);
            wkb.extend_from_slice(&2u32.to_le_bytes());
            let len = u32::try_from(line.len()).map_err(|e| vortex_err!("{e}"))?;
            wkb.extend_from_slice(&len.to_le_bytes());
            for (x, y) in line {
                wkb.extend_from_slice(&f64::to_le_bytes(x));
                wkb.extend_from_slice(&f64::to_le_bytes(y));
            }
        }

        let scalar = native_geometry_scalar_from_wkb(&wkb)?.expect("a multilinestring scalar");
        let DType::Extension(ext) = scalar.dtype() else {
            panic!("expected an extension dtype, got {}", scalar.dtype());
        };
        assert!(ext.is::<MultiLineString>());
        assert_eq!(ext.metadata::<MultiLineString>(), &xy_metadata());
        let mut ctx = SESSION.create_execution_ctx();
        let expected = crate::test_harness::multilinestring_column(vec![vec![
            vec![(0.0, 0.0), (1.0, 1.0)],
            vec![(2.0, 2.0), (3.0, 3.0)],
        ]])?
        .execute_scalar(0, &mut ctx)?;
        let actual_storage = scalar.as_extension().to_storage_scalar();
        let expected_storage = expected.as_extension().to_storage_scalar();
        assert_eq!(actual_storage.dtype(), expected_storage.dtype());
        assert_eq!(actual_storage, expected_storage);
        Ok(())
    }

    /// A non-nullable `List` of `element`.
    fn list(element: DType) -> DType {
        DType::List(Arc::new(element), Nullability::NonNullable)
    }

    /// The non-nullable 2D coordinate `Struct`.
    fn xy() -> DType {
        coordinate_storage_dtype(Dimension::Xy, Nullability::NonNullable)
    }

    /// The list-based geometry types fail to read storage with a nullable inner `List` or
    /// coordinate `Struct`, and the error reports the rejected storage.
    #[rstest]
    #[case::linestring_coordinates(     LineString,      list(xy().as_nullable()))]
    #[case::multipoint_coordinates(     MultiPoint,      list(xy().as_nullable()))]
    #[case::polygon_rings(              Polygon,         list(list(xy()).as_nullable()))]
    #[case::polygon_coordinates(        Polygon,         list(list(xy().as_nullable())))]
    #[case::multilinestring_lines(      MultiLineString, list(list(xy()).as_nullable()))]
    #[case::multilinestring_coordinates(MultiLineString, list(list(xy().as_nullable())))]
    #[case::multipolygon_polygons(      MultiPolygon,    list(list(list(xy())).as_nullable()))]
    #[case::multipolygon_rings(         MultiPolygon,    list(list(list(xy()).as_nullable())))]
    #[case::multipolygon_coordinates(   MultiPolygon,    list(list(list(xy().as_nullable()))))]
    fn nullable_nested_level_is_rejected(
        #[case] vtable: impl ExtVTable<Metadata = SpatialMetadata>,
        #[case] storage: DType,
    ) {
        let result = vtable.deserialize(&[], storage.clone());
        assert!(result.is_err_and(|error| error.to_string().contains(&format!(", got {storage}"))));
    }

    /// A serialized `vortex.st.linestring` dtype with a nullable coordinate `Struct` fails to read,
    /// with an error that names the expected storage. No constructor builds that storage, so the
    /// test builds the serialized extension dtype by hand.
    #[test]
    fn nullable_nested_level_fails_to_read() -> VortexResult<()> {
        let serialized = pb::DType {
            dtype_type: Some(DtypeType::Extension(Box::new(pb::Extension {
                id: "vortex.st.linestring".to_string(),
                storage_dtype: Some(Box::new(pb::DType::try_from(&list(xy().as_nullable()))?)),
                metadata: None,
            }))),
        };

        let result = DType::from_proto(&serialized, &spatial_session());
        assert!(result.is_err_and(|error| error.to_string().contains(
            "linestring storage must be list({x=f64, y=f64}), got list({x=f64, y=f64}?)"
        )));
        Ok(())
    }
}
