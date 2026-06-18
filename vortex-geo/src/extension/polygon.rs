// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`Polygon`] geometry extension type (`vortex.geo.polygon`): rings of the
//! [`Point`](super::Point) coordinate struct, stored as `List<List<Struct<x, y[, z][, m]>>>` and tagged with
//! [`GeoMetadata`] (CRS). The first ring is the exterior boundary; the rest are holes.

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType;
use geo_traits::to_geo::ToGeoGeometry;
use geo_types::Geometry;
use geoarrow::array::GeoArrowArrayAccessor;
use geoarrow::array::IntoArrow;
use geoarrow::array::PolygonArray;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::PolygonType;
use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrow::ArrowExport;
use vortex_array::arrow::ArrowExportVTable;
use vortex_array::arrow::ArrowImport;
use vortex_array::arrow::ArrowImportVTable;
use vortex_array::arrow::ArrowSession;
use vortex_array::arrow::ArrowSessionExt;
use vortex_array::arrow::FromArrowArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::arrow::FromArrowType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use super::GeoMetadata;
use super::coordinate::Dimension;
use super::coordinate::coordinate_dimension;
use super::coordinate::coordinate_storage_dtype;
use super::geo_metadata_from_arrow;
use super::geoarrow_metadata;

/// A polygon: `geoarrow.polygon`, stored as `List<List<Struct<x, y[, z][, m]>>>` (rings of vertices).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Polygon;

impl ExtVTable for Polygon {
    type Metadata = GeoMetadata;
    // No cheap owned value like Point's `Coordinate`; expose the raw storage scalar.
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        ExtId::new_static("vortex.geo.polygon")
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.encode_to_vec())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(GeoMetadata::decode(metadata)?)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        polygon_dimension(ext_dtype.storage_dtype()).map(|_| ())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<&'a ScalarValue> {
        Ok(storage_value)
    }
}

/// Canonical polygon storage: an outer list of rings, each a list of the coordinate `Struct`.
pub(crate) fn polygon_storage_dtype(dim: Dimension, nullability: Nullability) -> DType {
    let coords = coordinate_storage_dtype(dim, Nullability::NonNullable);
    let ring = DType::List(Arc::new(coords), Nullability::NonNullable);
    DType::List(Arc::new(ring), nullability)
}

/// Validate `dtype` is `List<List<coordinate-struct>>` and return its [`Dimension`].
pub(crate) fn polygon_dimension(dtype: &DType) -> VortexResult<Dimension> {
    let DType::List(ring, _) = dtype else {
        vortex_bail!("polygon storage must be a List of rings, was {dtype}");
    };
    let DType::List(coords, _) = ring.as_ref() else {
        vortex_bail!("polygon ring storage must be a List of coordinates, was {ring}");
    };
    coordinate_dimension(coords)
}

static ARROW_POLYGON: CachedId = CachedId::new(PolygonType::NAME);

/// The `geoarrow.polygon` extension type for `dimension`, with separated (struct) coordinates
/// matching `Polygon` storage.
fn polygon_type(geo_metadata: &GeoMetadata, dimension: Dimension) -> PolygonType {
    PolygonType::new(dimension.into(), geoarrow_metadata(geo_metadata))
}

/// Decode `Polygon` storage (`List<List<coordinate>>`) to `geo_types` polygons, for the geo scalar
/// functions. CRS does not affect planar geometry ops, so default metadata is used.
pub(crate) fn polygon_geometries(
    storage: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Geometry<f64>>> {
    let polygon_type = polygon_type(&GeoMetadata::default(), polygon_dimension(storage.dtype())?);
    let session = ctx.session().clone();
    let arrow = session.arrow().execute_arrow(storage.clone(), None, ctx)?;
    let polygons = PolygonArray::try_from((arrow.as_ref(), polygon_type))
        .map_err(|e| vortex_err!("failed to construct PolygonArray: {e}"))?;
    polygons
        .iter()
        .map(|geometry| -> VortexResult<Geometry<f64>> {
            Ok(geometry
                .ok_or_else(|| vortex_err!("geo: null geometry is not supported"))?
                .map_err(|e| vortex_err!("geo: geometry access failed: {e}"))?
                .to_geometry())
        })
        .collect()
}

impl ArrowExportVTable for Polygon {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_POLYGON
    }

    fn vortex_id(&self) -> Id {
        self.id()
    }

    fn to_arrow_field(
        &self,
        name: &str,
        dtype: &DType,
        session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        let ext_type = dtype.as_extension();
        let geo_metadata = ext_type.metadata::<Polygon>();
        let dimension = polygon_dimension(ext_type.storage_dtype())?;

        let mut field = session.to_arrow_field(name, ext_type.storage_dtype())?;
        field.try_with_extension_type(polygon_type(geo_metadata, dimension))?;

        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let is_polygon = array
            .dtype()
            .as_extension_opt()
            .map(|ext| ext.is::<Polygon>())
            .unwrap_or(false);
        if !is_polygon {
            return Ok(ArrowExport::Unsupported(array));
        }

        let Ok(polygon_meta) = target.try_extension_type::<PolygonType>() else {
            return Ok(ArrowExport::Unsupported(array));
        };
        if polygon_meta.coord_type() != CoordType::Separated {
            return Ok(ArrowExport::Unsupported(array));
        }

        let executed = array.execute::<ExtensionArray>(ctx)?;
        let storage = executed.storage_array().clone();

        let storage_field = Field::new(
            String::new(),
            target.data_type().clone(),
            target.is_nullable(),
        );
        let session = ctx.session().clone();
        let arrow_storage = session
            .arrow()
            .execute_arrow(storage, Some(&storage_field), ctx)?;

        // Round-trip through GeoArrow's polygon array; `into_arrow` is concrete, so wrap in `Arc`.
        let polygons = PolygonArray::try_from((arrow_storage.as_ref(), polygon_meta))
            .map_err(|e| vortex_err!("failed to construct PolygonArray: {e}"))?;

        Ok(ArrowExport::Exported(Arc::new(polygons.into_arrow())))
    }
}

impl ArrowImportVTable for Polygon {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_POLYGON
    }

    /// Import a `geoarrow.polygon` field as the [`Polygon`] dtype. Keyed off the standard GeoArrow
    /// name, so any producer (DataFusion, DuckDB, geoarrow-rs, …) resolves here. Accepts the full
    /// `PolygonType` extension, or — for a metadata-less geometry literal — the name alone, inferring
    /// the dimension from the coordinate field names.
    fn from_arrow_field(&self, field: &Field) -> VortexResult<Option<DType>> {
        let (dimension, metadata) =
            if let Ok(polygon_meta) = field.try_extension_type::<PolygonType>() {
                vortex_ensure!(
                    polygon_meta.coord_type() == CoordType::Separated,
                    "geoarrow.polygon with interleaved coordinates is not supported; \
                 re-encode with separated (struct) coordinates"
                );
                (
                    polygon_meta.dimension().into(),
                    geo_metadata_from_arrow(polygon_meta.metadata()),
                )
            } else {
                // Infer the dimension from the field names, not the canonical storage check: a literal's
                // coordinate fields may be nullable, which that check rejects. Peel the two `List` layers
                // (polygon → rings → coordinates) to reach the struct.
                if field.extension_type_name() != Some(PolygonType::NAME) {
                    return Ok(None);
                }
                let DType::List(ring, _) = DType::from_arrow(field) else {
                    return Ok(None);
                };
                let DType::List(coords, _) = ring.as_ref() else {
                    return Ok(None);
                };
                let DType::Struct(fields, _) = coords.as_ref() else {
                    return Ok(None);
                };
                let Ok(dimension) = Dimension::from_field_names(fields.names()) else {
                    return Ok(None);
                };
                (dimension, GeoMetadata::default())
            };

        let storage_dtype = polygon_storage_dtype(dimension, field.is_nullable().into());
        Ok(Some(DType::Extension(
            ExtDType::try_with_vtable(Polygon, metadata, storage_dtype)?.erased(),
        )))
    }

    fn from_arrow_array(
        &self,
        array: ArrowArrayRef,
        field: &Field,
        dtype: &DType,
    ) -> VortexResult<ArrowImport> {
        let Some(ext_dtype) = dtype.as_extension_opt() else {
            return Ok(ArrowImport::Unsupported(array));
        };
        if !ext_dtype.is::<Polygon>()
            || field.try_extension_type::<PolygonType>().is_err()
            || !matches!(array.data_type(), DataType::List(_))
        {
            return Ok(ArrowImport::Unsupported(array));
        }

        let storage = ArrayRef::from_arrow(array.as_ref(), field.is_nullable())?;
        Ok(ArrowImport::Imported(
            ExtensionArray::try_new(ext_dtype.clone(), storage)?.into_array(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_error::VortexResult;

    use super::Polygon;
    use super::polygon_storage_dtype;
    use crate::extension::GeoMetadata;
    use crate::extension::coordinate::Dimension;
    use crate::extension::coordinate::coordinate_storage_dtype;

    fn geo_meta() -> GeoMetadata {
        GeoMetadata {
            crs: Some("EPSG:4326".to_string()),
        }
    }

    /// `Polygon` accepts the canonical `List<List<coordinate-struct>>` storage of every dimension.
    #[rstest]
    #[case::xy(Dimension::Xy)]
    #[case::xyz(Dimension::Xyz)]
    #[case::xym(Dimension::Xym)]
    #[case::xyzm(Dimension::Xyzm)]
    fn polygon_validates_every_dimension(#[case] dim: Dimension) -> VortexResult<()> {
        let storage = polygon_storage_dtype(dim, Nullability::NonNullable);
        ExtDType::<Polygon>::try_new(geo_meta(), storage)?;
        Ok(())
    }

    /// Non-polygon storage is rejected at dtype construction: a bare struct (point) and a single
    /// list (linestring) both fail.
    #[test]
    fn polygon_rejects_invalid_storage() -> VortexResult<()> {
        let primitive = DType::Primitive(PType::F64, Nullability::NonNullable);
        assert!(ExtDType::<Polygon>::try_new(geo_meta(), primitive).is_err());

        // A single list of coordinates is a LineString, not a Polygon.
        let coords = coordinate_storage_dtype(Dimension::Xy, Nullability::NonNullable);
        let line = DType::List(Arc::new(coords), Nullability::NonNullable);
        assert!(ExtDType::<Polygon>::try_new(geo_meta(), line).is_err());
        Ok(())
    }
}
