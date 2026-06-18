// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`Point`] geometry extension type (`vortex.geo.point`): a location stored columnarly as
//! `Struct<x, y[, z][, m]>` of non-nullable `f64` — the four GeoArrow dimensions XY, XYZ, XYM,
//! XYZM — tagged with [`GeoMetadata`] (CRS). `z` is an optional elevation and `m` an optional
//! measure: an arbitrary per-point value such as distance along a route or a timestamp.

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType;
use geo_traits::to_geo::ToGeoGeometry;
use geo_types::Geometry;
use geoarrow::array::GeoArrowArrayAccessor;
use geoarrow::array::IntoArrow;
use geoarrow::array::PointArray;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::PointType;
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
use vortex_array::dtype::arrow::FromArrowType;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use super::GeoMetadata;
use super::coordinate::Coordinate;
use super::coordinate::Dimension;
use super::coordinate::coordinate_dimension;
use super::coordinate::coordinate_from_struct;
use super::coordinate::coordinate_storage_dtype;
use super::geo_metadata_from_arrow;
use super::geoarrow_metadata;

/// A single location: `geoarrow.point`, stored as `Struct<x, y[, z][, m]>` of non-nullable `f64`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Point;

impl ExtVTable for Point {
    type Metadata = GeoMetadata;
    type NativeValue<'a> = Coordinate;

    fn id(&self) -> ExtId {
        ExtId::new_static("vortex.geo.point")
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.encode_to_vec())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(GeoMetadata::decode(metadata)?)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        coordinate_dimension(ext_dtype.storage_dtype()).map(|_| ())
    }

    fn unpack_native<'a>(
        ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<Coordinate> {
        let storage = Scalar::try_new(
            ext_dtype.storage_dtype().clone(),
            Some(storage_value.clone()),
        )?;
        coordinate_from_struct(&storage)
    }
}

static ARROW_POINT: CachedId = CachedId::new(PointType::NAME);

/// The `geoarrow.point` extension type for `dimension`, with separated (struct) coordinates
/// matching `Point` storage.
fn point_type(geo_metadata: &GeoMetadata, dimension: Dimension) -> PointType {
    PointType::new(dimension.into(), geoarrow_metadata(geo_metadata))
}

/// Decode `Point` storage to `geo_types` points, for the geo scalar functions.
pub(crate) fn point_geometries(
    storage: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Geometry<f64>>> {
    let point_type = point_type(
        &GeoMetadata::default(),
        coordinate_dimension(storage.dtype())?,
    );
    let session = ctx.session().clone();
    let arrow = session.arrow().execute_arrow(storage.clone(), None, ctx)?;
    let points = PointArray::try_from((arrow.as_ref(), point_type))
        .map_err(|e| vortex_err!("failed to construct PointArray: {e}"))?;
    points
        .iter()
        .map(|geometry| -> VortexResult<Geometry<f64>> {
            Ok(geometry
                .ok_or_else(|| vortex_err!("geo: null geometry is not supported"))?
                .map_err(|e| vortex_err!("geo: geometry access failed: {e}"))?
                .to_geometry())
        })
        .collect()
}

impl ArrowExportVTable for Point {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_POINT
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
        let geo_metadata = ext_type.metadata::<Point>();
        let dimension = coordinate_dimension(ext_type.storage_dtype())?;

        let mut field = session.to_arrow_field(name, ext_type.storage_dtype())?;
        field.try_with_extension_type(point_type(geo_metadata, dimension))?;

        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let is_point = array
            .dtype()
            .as_extension_opt()
            .map(|ext| ext.is::<Point>())
            .unwrap_or(false);
        if !is_point {
            return Ok(ArrowExport::Unsupported(array));
        }

        let Ok(point_meta) = target.try_extension_type::<PointType>() else {
            return Ok(ArrowExport::Unsupported(array));
        };
        if point_meta.coord_type() != CoordType::Separated {
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

        // Round-trip through the GeoArrow point array type: this validates that the storage is
        // the separated-coordinate struct layout expected for a `PointType` extension field.
        let points = PointArray::try_from((arrow_storage.as_ref(), point_meta))
            .map_err(|e| vortex_err!("failed to construct PointArray: {e}"))?;

        Ok(ArrowExport::Exported(points.into_arrow()))
    }
}

impl ArrowImportVTable for Point {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_POINT
    }

    /// Import a `geoarrow.point` field as the [`Point`] dtype. Keyed off the standard GeoArrow name,
    /// so any producer (DataFusion, DuckDB, geoarrow-rs, …) resolves here. Accepts the full
    /// `PointType` extension, or — for a metadata-less geometry literal — the name alone, inferring
    /// the dimension from the coordinate field names.
    fn from_arrow_field(&self, field: &Field) -> VortexResult<Option<DType>> {
        let (dimension, metadata) = if let Ok(point_meta) = field.try_extension_type::<PointType>()
        {
            vortex_ensure!(
                point_meta.coord_type() == CoordType::Separated,
                "geoarrow.point with interleaved coordinates is not supported; \
                 re-encode with separated (struct) coordinates"
            );
            (
                point_meta.dimension().into(),
                geo_metadata_from_arrow(point_meta.metadata()),
            )
        } else {
            // Infer the dimension from the field names, not the canonical storage check: a literal's
            // coordinate fields may be nullable, which that check rejects.
            if field.extension_type_name() != Some(PointType::NAME) {
                return Ok(None);
            }
            let DType::Struct(fields, _) = DType::from_arrow(field) else {
                return Ok(None);
            };
            let Ok(dimension) = Dimension::from_field_names(fields.names()) else {
                return Ok(None);
            };
            (dimension, GeoMetadata::default())
        };

        let storage_dtype = coordinate_storage_dtype(dimension, field.is_nullable().into());
        Ok(Some(DType::Extension(
            ExtDType::try_with_vtable(Point, metadata, storage_dtype)?.erased(),
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
        if !ext_dtype.is::<Point>()
            || field.try_extension_type::<PointType>().is_err()
            || !matches!(array.data_type(), DataType::Struct(_))
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
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_error::VortexResult;

    use super::Point;
    use crate::extension::GeoMetadata;
    use crate::extension::coordinate::Coordinate;
    use crate::extension::coordinate::Dimension;
    use crate::extension::coordinate::coordinate_storage_dtype;
    use crate::test_harness::coordinate_from_scalar;
    use crate::test_harness::point_column;

    fn geo_meta() -> GeoMetadata {
        GeoMetadata {
            crs: Some("EPSG:4326".to_string()),
        }
    }

    /// `Point` accepts the canonical coordinate storage of every GeoArrow dimension.
    #[rstest]
    #[case::xy(Dimension::Xy)]
    #[case::xyz(Dimension::Xyz)]
    #[case::xym(Dimension::Xym)]
    #[case::xyzm(Dimension::Xyzm)]
    fn point_validates_every_dimension(#[case] dim: Dimension) -> VortexResult<()> {
        let storage = coordinate_storage_dtype(dim, Nullability::NonNullable);
        ExtDType::<Point>::try_new(geo_meta(), storage)?;
        Ok(())
    }

    /// Invalid storage is rejected at dtype construction: both non-struct storage and a struct whose
    /// fields are not GeoArrow coordinates.
    #[test]
    fn point_rejects_invalid_storage() -> VortexResult<()> {
        let primitive = DType::Primitive(PType::F64, Nullability::NonNullable);
        assert!(ExtDType::<Point>::try_new(geo_meta(), primitive).is_err());

        let wrong_fields = StructArray::from_fields(&[
            ("a", PrimitiveArray::from_iter(vec![0.0f64]).into_array()),
            ("b", PrimitiveArray::from_iter(vec![0.0f64]).into_array()),
        ])?
        .into_array();
        assert!(ExtDType::<Point>::try_new(geo_meta(), wrong_fields.dtype().clone()).is_err());
        Ok(())
    }

    /// A `Point` column round-trips through scalar execution back to the original coordinates.
    #[test]
    fn point_unpacks_coordinates() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let points = point_column(vec![1.0, -111.7610], vec![2.0, 34.8697])?;

        assert_eq!(
            coordinate_from_scalar(&points.execute_scalar(0, &mut ctx)?)?,
            Coordinate::xy(1.0, 2.0)
        );
        assert_eq!(
            coordinate_from_scalar(&points.execute_scalar(1, &mut ctx)?)?,
            Coordinate::xy(-111.7610, 34.8697)
        );
        Ok(())
    }
}
