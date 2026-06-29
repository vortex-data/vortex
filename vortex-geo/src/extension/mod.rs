// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub(crate) mod coordinate;
mod multipolygon;
mod point;
mod polygon;
mod wkb;

use std::fmt::Display;
use std::sync::Arc;

use ::wkb::reader::GeometryType;
use arrow_array::BinaryArray;
use geo_types::Geometry;
use geoarrow::array::GenericWkbArray;
use geoarrow::array::GeoArrowArray;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::Crs;
use geoarrow::datatypes::Dimension;
use geoarrow::datatypes::GeoArrowType;
use geoarrow::datatypes::Metadata;
use geoarrow::datatypes::MultiPolygonType;
use geoarrow::datatypes::PointType;
use geoarrow::datatypes::PolygonType;
use geoarrow::datatypes::WkbType;
use geoarrow_cast::cast::cast;
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
    } else if ext.is::<Polygon>() {
        polygon_geometries(&storage, ctx)
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
/// `Point`/`Polygon`/`MultiPolygon` scalar. `None` for unsupported types. Plan-time, one value only.
pub fn native_geometry_scalar_from_wkb(bytes: &[u8]) -> VortexResult<Option<Scalar>> {
    let metadata = geoarrow_metadata(&GeoMetadata::default());
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
        ArrayRef::from_arrow(native.to_array_ref().as_ref(), false)
    };

    let scalar = match Wkb::try_from_bytes(bytes)?.geometry_type() {
        GeometryType::Point => {
            let target = GeoArrowType::Point(
                PointType::new(Dimension::XY, metadata).with_coord_type(CoordType::Separated),
            );
            geo_ext_scalar(Point, to_storage(&target)?)?
        }
        GeometryType::Polygon => {
            let target = GeoArrowType::Polygon(
                PolygonType::new(Dimension::XY, metadata).with_coord_type(CoordType::Separated),
            );
            geo_ext_scalar(Polygon, to_storage(&target)?)?
        }
        GeometryType::MultiPolygon => {
            let target = GeoArrowType::MultiPolygon(
                MultiPolygonType::new(Dimension::XY, metadata)
                    .with_coord_type(CoordType::Separated),
            );
            geo_ext_scalar(MultiPolygon, to_storage(&target)?)?
        }
        _ => return Ok(None),
    };
    Ok(Some(scalar))
}

/// Wrap cast-from-WKB `storage` in its `vtable` extension type and pull out the single scalar.
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
    use vortex_array::dtype::DType;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;

    use super::Point;
    use super::Polygon;
    use super::native_geometry_scalar_from_wkb;
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

    /// A little-endian WKB `POINT` literal decodes to the native `Point` extension scalar (the Q1
    /// fast-path operand).
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

    /// A little-endian WKB `POLYGON` literal decodes to the native `Polygon` extension scalar (the Q3
    /// point→polygon operand), proving the literal decode is not point-only.
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
}
