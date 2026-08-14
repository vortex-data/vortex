// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`Polygon`] geometry extension type (`vortex.st.polygon`): rings of the
//! [`Point`](super::Point) coordinate struct, stored as `List<List<Struct<x, y[, z][, m]>>>` and tagged with
//! [`SpatialMetadata`] (CRS). The first ring is the exterior boundary; the rest are holes.

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType;
use geo_traits::to_geo::ToGeoGeometry;
use geo_types::Geometry;
use geoarrow::array::GeoArrowArray;
use geoarrow::array::GeoArrowArrayAccessor;
use geoarrow::array::IntoArrow;
use geoarrow::array::PolygonArray;
use geoarrow::array::PolygonBuilder;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::PolygonType;
use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::ScalarValue;
use vortex_arrow::ArrowExport;
use vortex_arrow::ArrowExportVTable;
use vortex_arrow::ArrowImport;
use vortex_arrow::ArrowImportVTable;
use vortex_arrow::ArrowSession;
use vortex_arrow::ArrowSessionExt;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use super::SpatialMetadata;
use super::coordinate::Dimension;
use super::coordinate::coordinate_dimension;
use super::coordinate::coordinate_storage_dtype;
use super::geoarrow_metadata;
use super::geoarrow_to_wkb;
use super::placeholder_geometry;
use super::spatial_metadata_from_arrow;

/// A polygon: `geoarrow.polygon`, stored as `List<List<Struct<x, y[, z][, m]>>>` (rings of vertices).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Polygon;

impl ExtVTable for Polygon {
    type Metadata = SpatialMetadata;
    // No cheap owned value like Point's `Coordinate`; expose the raw storage scalar.
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.st.polygon");
        *ID
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.encode_to_vec())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(SpatialMetadata::decode(metadata)?)
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
fn polygon_type(spatial_metadata: &SpatialMetadata, dimension: Dimension) -> PolygonType {
    PolygonType::new(dimension.into(), geoarrow_metadata(spatial_metadata))
}

/// Build a native 2-D [`Polygon`] array from row-oriented `geo_types` polygons.
pub(crate) fn build_polygon_array(
    polygons: &[Option<geo_types::Polygon<f64>>],
    metadata: SpatialMetadata,
    nullability: Nullability,
    session: &ArrowSession,
) -> VortexResult<ArrayRef> {
    let polygons =
        PolygonBuilder::from_nullable_polygons(polygons, polygon_type(&metadata, Dimension::Xy))
            .finish();
    let storage_dtype = polygon_storage_dtype(Dimension::Xy, nullability);
    let storage = session
        .from_arrow_array(
            polygons.to_array_ref(),
            nullability == Nullability::Nullable,
        )?
        .cast(storage_dtype.clone())?;
    let ext_dtype = ExtDType::<Polygon>::try_new(metadata, storage_dtype)?;
    Ok(ExtensionArray::try_new(ext_dtype.erased(), storage)?.into_array())
}

/// Decode `Polygon` storage (`List<List<coordinate>>`) to `geo_types` polygons, for the spatial scalar
/// functions. CRS does not affect planar geometry ops, so default metadata is used.
pub(crate) fn polygon_geometries(
    storage: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Geometry<f64>>> {
    polygon_array(storage, ctx)?
        .iter()
        .map(|geometry| -> VortexResult<Geometry<f64>> {
            Ok(geometry
                .ok_or_else(|| vortex_err!("spatial: null geometry is not supported"))?
                .map_err(|e| vortex_err!("spatial: geometry access failed: {e}"))?
                .to_geometry())
        })
        .collect()
}

/// Like [`polygon_geometries`], but a null row decodes to the placeholder geometry instead of
/// failing. The caller guarantees null rows are never read.
pub(crate) fn polygon_geometries_null_tolerant(
    storage: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Geometry<f64>>> {
    polygon_array(storage, ctx)?
        .iter()
        .map(|geometry| match geometry {
            None => Ok(placeholder_geometry()),
            Some(geometry) => Ok(geometry
                .map_err(|e| vortex_err!("spatial: geometry access failed: {e}"))?
                .to_geometry()),
        })
        .collect()
}

/// Build a geoarrow `PolygonArray` from a `Polygon`'s `List<List<coordinate>>` storage.
fn polygon_array(storage: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<PolygonArray> {
    let polygon_type = polygon_type(
        &SpatialMetadata::default(),
        polygon_dimension(storage.dtype())?,
    );
    let session = ctx.session().clone();
    let arrow = session.arrow().execute_arrow(storage.clone(), None, ctx)?;
    PolygonArray::try_from((arrow.as_ref(), polygon_type))
        .map_err(|e| vortex_err!("failed to construct PolygonArray: {e}"))
}

/// A validated `Polygon` array (`try_from` checks the extension type).
pub struct PolygonData(ExtensionArray);

impl TryFrom<ExtensionArray> for PolygonData {
    type Error = VortexError;

    fn try_from(ext: ExtensionArray) -> Result<Self, Self::Error> {
        vortex_ensure!(
            ext.ext_dtype().is::<Polygon>(),
            "expected a Polygon extension array"
        );
        Ok(PolygonData(ext))
    }
}

impl PolygonData {
    /// Serialize polygons to WKB (a view array) — the form DuckDB `GEOMETRY` takes.
    pub fn to_wkb(&self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        geoarrow_to_wkb(
            &polygon_array(self.0.storage_array(), ctx)?,
            &ctx.session().arrow(),
        )
    }
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
        let spatial_metadata = ext_type.metadata::<Polygon>();
        let dimension = polygon_dimension(ext_type.storage_dtype())?;

        let mut field = session.to_arrow_field(name, ext_type.storage_dtype())?;
        field.try_with_extension_type(polygon_type(spatial_metadata, dimension))?;

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
    fn from_arrow_field(
        &self,
        field: &Field,
        session: &ArrowSession,
    ) -> VortexResult<Option<DType>> {
        let (dimension, metadata) =
            if let Ok(polygon_meta) = field.try_extension_type::<PolygonType>() {
                vortex_ensure!(
                    polygon_meta.coord_type() == CoordType::Separated,
                    "geoarrow.polygon with interleaved coordinates is not supported; \
                 re-encode with separated (struct) coordinates"
                );
                (
                    polygon_meta.dimension().into(),
                    spatial_metadata_from_arrow(polygon_meta.metadata()),
                )
            } else {
                // Infer the dimension from the field names, not the canonical storage check: a literal's
                // coordinate fields may be nullable, which that check rejects. Peel the two `List` layers
                // (polygon → rings → coordinates) to reach the struct.
                if field.extension_type_name() != Some(PolygonType::NAME) {
                    return Ok(None);
                }
                let Ok(DType::List(ring, _)) =
                    session.from_arrow_datatype(field.data_type(), field.is_nullable().into())
                else {
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
                (dimension, SpatialMetadata::default())
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
        session: &ArrowSession,
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

        let storage = session.from_arrow_array(array, field.is_nullable())?;
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
    use crate::extension::SpatialMetadata;
    use crate::extension::coordinate::Dimension;
    use crate::extension::coordinate::coordinate_storage_dtype;

    fn spatial_meta() -> SpatialMetadata {
        SpatialMetadata {
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
        ExtDType::<Polygon>::try_new(spatial_meta(), storage)?;
        Ok(())
    }

    /// Non-polygon storage is rejected at dtype construction: a bare struct (point) and a single
    /// list (linestring) both fail.
    #[test]
    fn polygon_rejects_invalid_storage() -> VortexResult<()> {
        let primitive = DType::Primitive(PType::F64, Nullability::NonNullable);
        assert!(ExtDType::<Polygon>::try_new(spatial_meta(), primitive).is_err());

        // A single list of coordinates is a LineString, not a Polygon.
        let coords = coordinate_storage_dtype(Dimension::Xy, Nullability::NonNullable);
        let line = DType::List(Arc::new(coords), Nullability::NonNullable);
        assert!(ExtDType::<Polygon>::try_new(spatial_meta(), line).is_err());
        Ok(())
    }
}
