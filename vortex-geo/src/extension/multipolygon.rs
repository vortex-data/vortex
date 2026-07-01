// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`MultiPolygon`] extension type (`vortex.geo.multipolygon`), stored as
//! `List<List<List<Struct<x, y[, z][, m]>>>>` (polygons → rings → coordinates) and tagged with
//! [`GeoMetadata`]. A single `Polygon` is a one-element multipolygon.

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType;
use geo_traits::to_geo::ToGeoGeometry;
use geo_types::Geometry;
use geoarrow::array::GeoArrowArrayAccessor;
use geoarrow::array::IntoArrow;
use geoarrow::array::MultiPolygonArray;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::MultiPolygonType;
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
use vortex_error::VortexError;
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
use super::geoarrow_to_wkb;

/// A multipolygon (`geoarrow.multipolygon`); a single `Polygon` is a one-element multipolygon.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct MultiPolygon;

impl ExtVTable for MultiPolygon {
    type Metadata = GeoMetadata;
    // No cheap owned value like Point's `Coordinate`; expose the raw storage scalar.
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.geo.multipolygon");
        *ID
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.encode_to_vec())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(GeoMetadata::decode(metadata)?)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        multipolygon_dimension(ext_dtype.storage_dtype()).map(|_| ())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<&'a ScalarValue> {
        Ok(storage_value)
    }
}

/// Storage `List<List<List<Struct>>>`: polygons → rings → coordinates.
pub(crate) fn multipolygon_storage_dtype(dim: Dimension, nullability: Nullability) -> DType {
    let coords = coordinate_storage_dtype(dim, Nullability::NonNullable);
    let ring = DType::List(Arc::new(coords), Nullability::NonNullable);
    let polygon = DType::List(Arc::new(ring), Nullability::NonNullable);
    DType::List(Arc::new(polygon), nullability)
}

/// Validate `dtype` is `List<List<List<coordinate-struct>>>` and return its [`Dimension`].
pub(crate) fn multipolygon_dimension(dtype: &DType) -> VortexResult<Dimension> {
    let DType::List(polygon, _) = dtype else {
        vortex_bail!("multipolygon storage must be a List of polygons, was {dtype}");
    };
    let DType::List(ring, _) = polygon.as_ref() else {
        vortex_bail!("multipolygon polygon storage must be a List of rings, was {polygon}");
    };
    let DType::List(coords, _) = ring.as_ref() else {
        vortex_bail!("multipolygon ring storage must be a List of coordinates, was {ring}");
    };
    coordinate_dimension(coords)
}

static ARROW_MULTIPOLYGON: CachedId = CachedId::new(MultiPolygonType::NAME);

/// The `geoarrow.multipolygon` type for `dimension`, with separated (struct) coordinates.
fn multipolygon_type(geo_metadata: &GeoMetadata, dimension: Dimension) -> MultiPolygonType {
    MultiPolygonType::new(dimension.into(), geoarrow_metadata(geo_metadata))
}

/// Decode storage to `geo_types` for the geo scalar functions (CRS is irrelevant to planar ops).
pub(crate) fn multipolygon_geometries(
    storage: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Geometry<f64>>> {
    multipolygon_array(storage, ctx)?
        .iter()
        .map(|geometry| -> VortexResult<Geometry<f64>> {
            Ok(geometry
                .ok_or_else(|| vortex_err!("geo: null geometry is not supported"))?
                .map_err(|e| vortex_err!("geo: geometry access failed: {e}"))?
                .to_geometry())
        })
        .collect()
}

/// Build a geoarrow `MultiPolygonArray` from the `MultiPolygon` storage.
fn multipolygon_array(
    storage: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<MultiPolygonArray> {
    let multipolygon_type = multipolygon_type(
        &GeoMetadata::default(),
        multipolygon_dimension(storage.dtype())?,
    );
    let session = ctx.session().clone();
    let arrow = session.arrow().execute_arrow(storage.clone(), None, ctx)?;
    MultiPolygonArray::try_from((arrow.as_ref(), multipolygon_type))
        .map_err(|e| vortex_err!("failed to construct MultiPolygonArray: {e}"))
}

/// A validated `MultiPolygon` array (`try_from` checks the extension type).
pub struct MultiPolygonData(ExtensionArray);

impl TryFrom<ExtensionArray> for MultiPolygonData {
    type Error = VortexError;

    fn try_from(ext: ExtensionArray) -> Result<Self, Self::Error> {
        vortex_ensure!(
            ext.ext_dtype().is::<MultiPolygon>(),
            "expected a MultiPolygon extension array"
        );
        Ok(MultiPolygonData(ext))
    }
}

impl MultiPolygonData {
    /// Serialize multipolygons to WKB (a view array) — the form DuckDB `GEOMETRY` takes.
    pub fn to_wkb(&self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        geoarrow_to_wkb(&multipolygon_array(self.0.storage_array(), ctx)?)
    }
}

impl ArrowExportVTable for MultiPolygon {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_MULTIPOLYGON
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
        let geo_metadata = ext_type.metadata::<MultiPolygon>();
        let dimension = multipolygon_dimension(ext_type.storage_dtype())?;

        let mut field = session.to_arrow_field(name, ext_type.storage_dtype())?;
        field.try_with_extension_type(multipolygon_type(geo_metadata, dimension))?;

        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let is_multipolygon = array
            .dtype()
            .as_extension_opt()
            .map(|ext| ext.is::<MultiPolygon>())
            .unwrap_or(false);
        if !is_multipolygon {
            return Ok(ArrowExport::Unsupported(array));
        }

        let Ok(multipolygon_meta) = target.try_extension_type::<MultiPolygonType>() else {
            return Ok(ArrowExport::Unsupported(array));
        };
        if multipolygon_meta.coord_type() != CoordType::Separated {
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

        let multipolygons =
            MultiPolygonArray::try_from((arrow_storage.as_ref(), multipolygon_meta))
                .map_err(|e| vortex_err!("failed to construct MultiPolygonArray: {e}"))?;

        Ok(ArrowExport::Exported(Arc::new(multipolygons.into_arrow())))
    }
}

impl ArrowImportVTable for MultiPolygon {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_MULTIPOLYGON
    }

    /// Import a `geoarrow.multipolygon` field (matched by GeoArrow name). Accepts the full
    /// `MultiPolygonType`, or a metadata-less literal (name only), inferring the dimension.
    fn from_arrow_field(&self, field: &Field) -> VortexResult<Option<DType>> {
        let (dimension, metadata) =
            if let Ok(multipolygon_meta) = field.try_extension_type::<MultiPolygonType>() {
                vortex_ensure!(
                    multipolygon_meta.coord_type() == CoordType::Separated,
                    "geoarrow.multipolygon with interleaved coordinates is not supported; \
                 re-encode with separated (struct) coordinates"
                );
                (
                    multipolygon_meta.dimension().into(),
                    geo_metadata_from_arrow(multipolygon_meta.metadata()),
                )
            } else {
                // Literal: peel the three `List` layers to the coordinate struct and read its
                // dimension from the field names (the canonical check rejects nullable coordinates).
                if field.extension_type_name() != Some(MultiPolygonType::NAME) {
                    return Ok(None);
                }
                let DType::List(polygon, _) = DType::from_arrow(field) else {
                    return Ok(None);
                };
                let DType::List(ring, _) = polygon.as_ref() else {
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

        let storage_dtype = multipolygon_storage_dtype(dimension, field.is_nullable().into());
        Ok(Some(DType::Extension(
            ExtDType::try_with_vtable(MultiPolygon, metadata, storage_dtype)?.erased(),
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
        if !ext_dtype.is::<MultiPolygon>()
            || field.try_extension_type::<MultiPolygonType>().is_err()
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

    use super::MultiPolygon;
    use super::multipolygon_storage_dtype;
    use crate::extension::GeoMetadata;
    use crate::extension::coordinate::Dimension;
    use crate::extension::coordinate::coordinate_storage_dtype;

    fn geo_meta() -> GeoMetadata {
        GeoMetadata {
            crs: Some("EPSG:4326".to_string()),
        }
    }

    /// `MultiPolygon` accepts the canonical `List<List<List<coordinate-struct>>>` storage of every
    /// dimension.
    #[rstest]
    #[case::xy(Dimension::Xy)]
    #[case::xyz(Dimension::Xyz)]
    #[case::xym(Dimension::Xym)]
    #[case::xyzm(Dimension::Xyzm)]
    fn multipolygon_validates_every_dimension(#[case] dim: Dimension) -> VortexResult<()> {
        let storage = multipolygon_storage_dtype(dim, Nullability::NonNullable);
        ExtDType::<MultiPolygon>::try_new(geo_meta(), storage)?;
        Ok(())
    }

    /// Non-multipolygon storage is rejected at dtype construction: a bare struct (point) and a
    /// double list (polygon) both fail.
    #[test]
    fn multipolygon_rejects_invalid_storage() -> VortexResult<()> {
        let primitive = DType::Primitive(PType::F64, Nullability::NonNullable);
        assert!(ExtDType::<MultiPolygon>::try_new(geo_meta(), primitive).is_err());

        // A double list (polygon) is not a multipolygon.
        let coords = coordinate_storage_dtype(Dimension::Xy, Nullability::NonNullable);
        let ring = DType::List(Arc::new(coords), Nullability::NonNullable);
        let polygon = DType::List(Arc::new(ring), Nullability::NonNullable);
        assert!(ExtDType::<MultiPolygon>::try_new(geo_meta(), polygon).is_err());
        Ok(())
    }
}
