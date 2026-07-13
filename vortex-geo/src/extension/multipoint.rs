// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`MultiPoint`] geometry extension type (`vortex.geo.multipoint`): an unordered set of the
//! [`Point`](super::Point) coordinate struct, stored as `List<Struct<x, y[, z][, m]>>` and tagged
//! with [`GeoMetadata`] (CRS). The storage layout matches [`LineString`](super::LineString); the
//! two are distinguished by their GeoArrow extension name, not their shape.

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType;
use geo_traits::to_geo::ToGeoGeometry;
use geo_types::Geometry;
use geoarrow::array::GeoArrowArrayAccessor;
use geoarrow::array::IntoArrow;
use geoarrow::array::MultiPointArray;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::MultiPointType;
use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
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
use vortex_arrow::FromArrowArray;
use vortex_arrow::FromArrowType;
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

/// A multipoint: `geoarrow.multipoint`, stored as `List<Struct<x, y[, z][, m]>>` (a set of points).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct MultiPoint;

impl ExtVTable for MultiPoint {
    type Metadata = GeoMetadata;
    // No cheap owned value like Point's `Coordinate`; expose the raw storage scalar.
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.geo.multipoint");
        *ID
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.encode_to_vec())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(GeoMetadata::decode(metadata)?)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        multipoint_dimension(ext_dtype.storage_dtype()).map(|_| ())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<&'a ScalarValue> {
        Ok(storage_value)
    }
}

/// Canonical multipoint storage: a list of the coordinate `Struct`.
pub(crate) fn multipoint_storage_dtype(dim: Dimension, nullability: Nullability) -> DType {
    let coords = coordinate_storage_dtype(dim, Nullability::NonNullable);
    DType::List(Arc::new(coords), nullability)
}

/// Validate `dtype` is `List<coordinate-struct>` and return its [`Dimension`].
pub(crate) fn multipoint_dimension(dtype: &DType) -> VortexResult<Dimension> {
    let DType::List(coords, _) = dtype else {
        vortex_bail!("multipoint storage must be a List of coordinates, was {dtype}");
    };
    coordinate_dimension(coords)
}

static ARROW_MULTIPOINT: CachedId = CachedId::new(MultiPointType::NAME);

/// The `geoarrow.multipoint` extension type for `dimension`, with separated (struct) coordinates.
fn multipoint_type(geo_metadata: &GeoMetadata, dimension: Dimension) -> MultiPointType {
    MultiPointType::new(dimension.into(), geoarrow_metadata(geo_metadata))
}

/// Decode `MultiPoint` storage (`List<coordinate>`) to `geo_types`, for the geo scalar functions.
pub(crate) fn multipoint_geometries(
    storage: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Geometry<f64>>> {
    multipoint_array(storage, ctx)?
        .iter()
        .map(|geometry| -> VortexResult<Geometry<f64>> {
            Ok(geometry
                .ok_or_else(|| vortex_err!("geo: null geometry is not supported"))?
                .map_err(|e| vortex_err!("geo: geometry access failed: {e}"))?
                .to_geometry())
        })
        .collect()
}

/// Build a geoarrow `MultiPointArray` from a `MultiPoint`'s `List<coordinate>` storage.
fn multipoint_array(storage: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<MultiPointArray> {
    let multipoint_type = multipoint_type(
        &GeoMetadata::default(),
        multipoint_dimension(storage.dtype())?,
    );
    let session = ctx.session().clone();
    let arrow = session.arrow().execute_arrow(storage.clone(), None, ctx)?;
    MultiPointArray::try_from((arrow.as_ref(), multipoint_type))
        .map_err(|e| vortex_err!("failed to construct MultiPointArray: {e}"))
}

/// A validated `MultiPoint` array (`try_from` checks the extension type).
pub struct MultiPointData(ExtensionArray);

impl TryFrom<ExtensionArray> for MultiPointData {
    type Error = VortexError;

    fn try_from(ext: ExtensionArray) -> Result<Self, Self::Error> {
        vortex_ensure!(
            ext.ext_dtype().is::<MultiPoint>(),
            "expected a MultiPoint extension array"
        );
        Ok(MultiPointData(ext))
    }
}

impl MultiPointData {
    /// Serialize multipoints to WKB (a view array) — the form DuckDB `GEOMETRY` takes.
    pub fn to_wkb(&self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        geoarrow_to_wkb(&multipoint_array(self.0.storage_array(), ctx)?)
    }
}

impl ArrowExportVTable for MultiPoint {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_MULTIPOINT
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
        let geo_metadata = ext_type.metadata::<MultiPoint>();
        let dimension = multipoint_dimension(ext_type.storage_dtype())?;

        let mut field = session.to_arrow_field(name, ext_type.storage_dtype())?;
        field.try_with_extension_type(multipoint_type(geo_metadata, dimension))?;

        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let is_multipoint = array
            .dtype()
            .as_extension_opt()
            .map(|ext| ext.is::<MultiPoint>())
            .unwrap_or(false);
        if !is_multipoint {
            return Ok(ArrowExport::Unsupported(array));
        }

        let Ok(multipoint_meta) = target.try_extension_type::<MultiPointType>() else {
            return Ok(ArrowExport::Unsupported(array));
        };
        if multipoint_meta.coord_type() != CoordType::Separated {
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

        // Round-trip through GeoArrow's multipoint array; `into_arrow` is concrete, so wrap in `Arc`.
        let multipoints = MultiPointArray::try_from((arrow_storage.as_ref(), multipoint_meta))
            .map_err(|e| vortex_err!("failed to construct MultiPointArray: {e}"))?;

        Ok(ArrowExport::Exported(Arc::new(multipoints.into_arrow())))
    }
}

impl ArrowImportVTable for MultiPoint {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_MULTIPOINT
    }

    /// Import a `geoarrow.multipoint` field as the [`MultiPoint`] dtype (matched by GeoArrow name).
    /// Accepts the full `MultiPointType`, or a metadata-less literal (name only), inferring the
    /// dimension from the coordinate field names.
    fn from_arrow_field(&self, field: &Field) -> VortexResult<Option<DType>> {
        let (dimension, metadata) =
            if let Ok(multipoint_meta) = field.try_extension_type::<MultiPointType>() {
                vortex_ensure!(
                    multipoint_meta.coord_type() == CoordType::Separated,
                    "geoarrow.multipoint with interleaved coordinates is not supported; \
                 re-encode with separated (struct) coordinates"
                );
                (
                    multipoint_meta.dimension().into(),
                    geo_metadata_from_arrow(multipoint_meta.metadata()),
                )
            } else {
                if field.extension_type_name() != Some(MultiPointType::NAME) {
                    return Ok(None);
                }
                let DType::List(coords, _) = DType::from_arrow(field) else {
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

        let storage_dtype = multipoint_storage_dtype(dimension, field.is_nullable().into());
        Ok(Some(DType::Extension(
            ExtDType::try_with_vtable(MultiPoint, metadata, storage_dtype)?.erased(),
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
        if !ext_dtype.is::<MultiPoint>()
            || field.try_extension_type::<MultiPointType>().is_err()
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
    use rstest::rstest;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_error::VortexResult;

    use super::MultiPoint;
    use super::multipoint_storage_dtype;
    use crate::extension::GeoMetadata;
    use crate::extension::coordinate::Dimension;

    fn geo_meta() -> GeoMetadata {
        GeoMetadata {
            crs: Some("EPSG:4326".to_string()),
        }
    }

    /// `MultiPoint` accepts the canonical `List<coordinate-struct>` storage of every dimension.
    #[rstest]
    #[case::xy(Dimension::Xy)]
    #[case::xyz(Dimension::Xyz)]
    #[case::xym(Dimension::Xym)]
    #[case::xyzm(Dimension::Xyzm)]
    fn multipoint_validates_every_dimension(#[case] dim: Dimension) -> VortexResult<()> {
        let storage = multipoint_storage_dtype(dim, Nullability::NonNullable);
        ExtDType::<MultiPoint>::try_new(geo_meta(), storage)?;
        Ok(())
    }

    /// Non-multipoint storage is rejected at dtype construction: a bare coordinate struct (point) is
    /// not a list of coordinates.
    #[test]
    fn multipoint_rejects_invalid_storage() -> VortexResult<()> {
        let primitive = DType::Primitive(PType::F64, Nullability::NonNullable);
        assert!(ExtDType::<MultiPoint>::try_new(geo_meta(), primitive).is_err());
        Ok(())
    }
}
