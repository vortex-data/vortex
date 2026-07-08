// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod coordinate;
mod linestring;
mod multilinestring;
mod multipoint;
mod multipolygon;
mod point;
mod polygon;
mod wkb;

use std::fmt::Display;
use std::sync::Arc;

use arrow_array::BinaryArray;
use arrow_array::StringArray;
use geo_traits::GeometryTrait;
use geo_traits::GeometryType;
use geo_types::Geometry;
use geoarrow::array::GenericWkbArray;
use geoarrow::array::GenericWktArray;
use geoarrow::array::GeoArrowArray;
use geoarrow::array::GeoArrowArrayAccessor;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::Dimension;
use geoarrow::datatypes::GeoArrowType;
use geoarrow::datatypes::LineStringType;
use geoarrow::datatypes::Metadata;
use geoarrow::datatypes::MultiLineStringType;
use geoarrow::datatypes::MultiPointType;
use geoarrow::datatypes::MultiPolygonType;
use geoarrow::datatypes::PointType;
use geoarrow::datatypes::PolygonType;
use geoarrow::datatypes::WkbType;
use geoarrow::datatypes::WktType;
use geoarrow_cast::cast::cast;
pub use linestring::*;
pub use multilinestring::*;
pub use multipoint::*;
pub use multipolygon::*;
pub use point::*;
pub use polygon::*;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrow::FromArrowArray;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
pub use wkb::*;

/// Decode a native geometry column to `geo_types`. A non-geometry operand is an error.
pub(crate) fn geometries(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Geometry<f64>>> {
    let Some(ext) = array.dtype().as_extension_opt() else {
        vortex_bail!(
            "geo: operand is not a geometry extension type, was {}",
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
    } else {
        vortex_bail!("geo: unsupported geometry extension {}", array.dtype())
    }
}

/// Decode a constant operand scalar to one geo geometry, a constant of any
/// supported geometry type is decoded exactly like a column.
pub(crate) fn single_geometry(
    scalar: &Scalar,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Geometry<f64>> {
    let array = ConstantArray::new(scalar.clone(), 1).into_array();
    geometries(&array, ctx)?
        .pop()
        .ok_or_else(|| vortex_err!("geo: constant operand decoded to no geometry"))
}

/// Decode a WKB geometry literal (DuckDB's wire form for `GEOMETRY` constants) to its native
/// geometry scalar. `None` for kinds without a native type (e.g. GeometryCollection). Plan-time, one value only.
pub fn native_geometry_scalar_from_wkb(bytes: &[u8]) -> VortexResult<Option<Scalar>> {
    let metadata = geoarrow_metadata(&GeoMetadata::default());
    let binary = BinaryArray::from(vec![Some(bytes)]);
    let src = GenericWkbArray::<i32>::try_from((
        &binary as &dyn arrow_array::Array,
        WkbType::new(Arc::clone(&metadata)),
    ))
    .map_err(|e| vortex_err!("failed to read WKB literal: {e}"))?;
    let geometry = src
        .value(0)
        .map_err(|e| vortex_err!("failed to decode WKB literal: {e}"))?;
    native_scalar_from_geometry(&geometry, &src, metadata)
}

/// Decode a single WKT geometry literal to its native geometry scalar. `None` for kinds without a native type.
/// Plan-time, one value only.
pub fn native_geometry_scalar_from_wkt(wkt: &str) -> VortexResult<Option<Scalar>> {
    let metadata = geoarrow_metadata(&GeoMetadata::default());
    let string = StringArray::from(vec![Some(wkt)]);
    let src = GenericWktArray::<i32>::try_from((
        &string as &dyn arrow_array::Array,
        WktType::new(Arc::clone(&metadata)),
    ))
    .map_err(|e| vortex_err!("failed to read WKT literal: {e}"))?;
    let geometry = src
        .value(0)
        .map_err(|e| vortex_err!("failed to parse WKT literal: {e}"))?;
    native_scalar_from_geometry(&geometry, &src, metadata)
}

/// Dispatch a decoded geoarrow geometry element to its native geometry scalar. `None` for geometry kinds without a native
/// type (e.g. GeometryCollection).
fn native_scalar_from_geometry(
    geometry: &impl GeometryTrait<T = f64>,
    src: &dyn GeoArrowArray,
    metadata: Arc<Metadata>,
) -> VortexResult<Option<Scalar>> {
    Ok(match geometry.as_type() {
        GeometryType::Point(_) => Some(geo_scalar_from_geoarrow(
            src,
            &point_target(metadata),
            Point,
        )?),
        GeometryType::LineString(_) => Some(geo_scalar_from_geoarrow(
            src,
            &linestring_target(metadata),
            LineString,
        )?),
        GeometryType::Polygon(_) => Some(geo_scalar_from_geoarrow(
            src,
            &polygon_target(metadata),
            Polygon,
        )?),
        GeometryType::MultiPoint(_) => Some(geo_scalar_from_geoarrow(
            src,
            &multipoint_target(metadata),
            MultiPoint,
        )?),
        GeometryType::MultiLineString(_) => Some(geo_scalar_from_geoarrow(
            src,
            &multilinestring_target(metadata),
            MultiLineString,
        )?),
        GeometryType::MultiPolygon(_) => Some(geo_scalar_from_geoarrow(
            src,
            &multipolygon_target(metadata),
            MultiPolygon,
        )?),
        _ => None,
    })
}

fn point_target(metadata: Arc<Metadata>) -> GeoArrowType {
    GeoArrowType::Point(
        PointType::new(Dimension::XY, metadata).with_coord_type(CoordType::Separated),
    )
}

fn linestring_target(metadata: Arc<Metadata>) -> GeoArrowType {
    GeoArrowType::LineString(
        LineStringType::new(Dimension::XY, metadata).with_coord_type(CoordType::Separated),
    )
}

fn polygon_target(metadata: Arc<Metadata>) -> GeoArrowType {
    GeoArrowType::Polygon(
        PolygonType::new(Dimension::XY, metadata).with_coord_type(CoordType::Separated),
    )
}

fn multipoint_target(metadata: Arc<Metadata>) -> GeoArrowType {
    GeoArrowType::MultiPoint(
        MultiPointType::new(Dimension::XY, metadata).with_coord_type(CoordType::Separated),
    )
}

fn multilinestring_target(metadata: Arc<Metadata>) -> GeoArrowType {
    GeoArrowType::MultiLineString(
        MultiLineStringType::new(Dimension::XY, metadata).with_coord_type(CoordType::Separated),
    )
}

fn multipolygon_target(metadata: Arc<Metadata>) -> GeoArrowType {
    GeoArrowType::MultiPolygon(
        MultiPolygonType::new(Dimension::XY, metadata).with_coord_type(CoordType::Separated),
    )
}

/// WKB/WKT literal decoders: cast the geoarrow `src` array to the native
/// `target` type, import the result as a Vortex storage array, and wrap it in the geo extension
/// `vtable`, returning the single decoded scalar.
fn geo_scalar_from_geoarrow<V: ExtVTable<Metadata = GeoMetadata>>(
    src: &dyn GeoArrowArray,
    target: &GeoArrowType,
    vtable: V,
) -> VortexResult<Scalar> {
    let native =
        cast(src, target).map_err(|e| vortex_err!("failed to cast geometry literal: {e}"))?;
    let storage = ArrayRef::from_arrow(native.to_array_ref().as_ref(), false)?;
    geo_ext_scalar(vtable, storage)
}

/// Wrap cast-from-geometry `storage` in its `vtable` extension type and pull out the single scalar.
// `scalar_at` is deprecated for `execute_scalar`, but there is no execution context at plan time.
#[allow(deprecated)]
fn geo_ext_scalar<V: ExtVTable<Metadata = GeoMetadata>>(
    vtable: V,
    storage: ArrayRef,
) -> VortexResult<Scalar> {
    let ext = ExtDType::try_with_vtable(vtable, GeoMetadata::default(), storage.dtype().clone())?
        .erased();
    ExtensionArray::try_new(ext, storage)?
        .into_array()
        .scalar_at(0)
}

/// Extension metadata that is common to all the geospatial extension types.
///
/// Currently, this is just the coordinate reference system (CRS).
/// We may wish to add a second field for edges interpretation in the future similar to
/// the GeoArrow standard.
#[derive(Clone, PartialEq, Eq, Hash, prost::Message)]
pub struct GeoMetadata {
    #[prost(optional, string, tag = "1")]
    pub crs: Option<String>,
}

impl Display for GeoMetadata {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.crs.as_ref() {
            Some(crs) => write!(f, "Geometry(crs={crs})"),
            None => write!(f, "Geometry(unreferenced)"),
        }
    }
}

/// The GeoArrow [`Metadata`] equivalent of `geo_metadata`.
pub(crate) fn geoarrow_metadata(geo_metadata: &GeoMetadata) -> Arc<Metadata> {
    Arc::new(Metadata::new(
        geo_metadata
            .crs
            .as_ref()
            .map(|crs| Crs::from_unknown_crs_type(crs.to_string()))
            .unwrap_or_default(),
        None,
    ))
}

/// Serialize a native geometry array to WKB (a `WkbView` array) via geoarrow's cast.
/// Shared by the `to_wkb` methods on the geometry extension types.
pub(crate) fn geoarrow_to_wkb(geo_array: &dyn GeoArrowArray) -> VortexResult<ArrayRef> {
    let wkb_type = GeoArrowType::WkbView(WkbType::new(geoarrow_metadata(&GeoMetadata::default())));
    let wkb = cast(geo_array, &wkb_type)
        .map_err(|e| vortex_err!("failed to cast geometry to WKB: {e}"))?;
    ArrayRef::from_arrow(wkb.to_array_ref().as_ref(), false)
}

/// Recover [`GeoMetadata`] from GeoArrow metadata.
pub(crate) fn geo_metadata_from_arrow(metadata: &Metadata) -> GeoMetadata {
    let crs = metadata.crs().crs_value().map(|value| {
        // `Crs::from_unknown_crs_type` stores the user's string verbatim as a JSON string
        // value, so prefer the raw string when available to round-trip cleanly. For other
        // CRS encodings (PROJJSON object, etc.), fall back to the JSON-encoded form.
        value
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| value.to_string())
    });
    GeoMetadata { crs }
}

#[cfg(test)]
mod tests {
    use prost::Message;
    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::LineString;
    use super::MultiLineString;
    use super::MultiPoint;
    use super::Point;
    use super::Polygon;
    use super::native_geometry_scalar_from_wkb;
    use super::native_geometry_scalar_from_wkt;
    use crate::extension::GeoMetadata;

    #[test]
    fn test_metadata() {
        let meta = GeoMetadata {
            crs: Some("EPSG:4326".to_string()),
        };

        assert_eq!(meta.to_string(), "Geometry(crs=EPSG:4326)");
        // round trip
        let bytes = meta.encode_to_vec();
        let decoded = GeoMetadata::decode(bytes.as_slice()).unwrap();
        assert_eq!(decoded, meta);
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
        Ok(())
    }

    /// A WKT literal of each OGC simple-feature kind decodes to the matching native extension type
    /// (asserted via the extension id carried in the dtype).
    #[rstest]
    #[case::point("POINT (-111.7610 34.8697)", "vortex.geo.point")]
    #[case::linestring("LINESTRING (0 0, 1 1, 2 2)", "vortex.geo.linestring")]
    #[case::polygon("POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))", "vortex.geo.polygon")]
    #[case::multipoint("MULTIPOINT (0 0, 1 1)", "vortex.geo.multipoint")]
    #[case::multilinestring(
        "MULTILINESTRING ((0 0, 1 1), (2 2, 3 3))",
        "vortex.geo.multilinestring"
    )]
    #[case::multipolygon(
        "MULTIPOLYGON (((0 0, 1 0, 1 1, 0 1, 0 0)))",
        "vortex.geo.multipolygon"
    )]
    fn decodes_wkt_to_native(#[case] wkt: &str, #[case] ext_id: &str) -> VortexResult<()> {
        let scalar = native_geometry_scalar_from_wkt(wkt)?.expect("a native geometry scalar");
        assert!(
            matches!(scalar.dtype(), DType::Extension(_)),
            "expected an extension dtype, got {}",
            scalar.dtype()
        );
        assert!(
            scalar.dtype().to_string().contains(ext_id),
            "{} does not carry {ext_id}",
            scalar.dtype()
        );
        Ok(())
    }

    /// The same geometry decoded from WKT and from WKB yields identical native scalars, proving both
    /// source formats funnel through the same cast/import/wrap tail.
    #[test]
    fn wkt_matches_wkb() -> VortexResult<()> {
        // WKB for POINT(1 2), little-endian.
        let mut wkb = vec![1u8];
        wkb.extend_from_slice(&1u32.to_le_bytes());
        wkb.extend_from_slice(&1.0f64.to_le_bytes());
        wkb.extend_from_slice(&2.0f64.to_le_bytes());

        let from_wkb = native_geometry_scalar_from_wkb(&wkb)?.expect("wkb point");
        let from_wkt = native_geometry_scalar_from_wkt("POINT (1 2)")?.expect("wkt point");
        assert_eq!(from_wkb, from_wkt);
        Ok(())
    }
}
