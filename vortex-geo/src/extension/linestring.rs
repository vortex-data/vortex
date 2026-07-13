// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`LineString`] geometry extension type (`vortex.geo.linestring`): an ordered path of the
//! [`Point`](super::Point) coordinate struct, stored as `List<Struct<x, y[, z][, m]>>` and tagged
//! with [`GeoMetadata`] (CRS).

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType;
use geo_traits::to_geo::ToGeoGeometry;
use geo_types::Geometry;
use geoarrow::array::GeoArrowArrayAccessor;
use geoarrow::array::IntoArrow;
use geoarrow::array::LineStringArray;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::LineStringType;
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

/// A line string: `geoarrow.linestring`, stored as `List<Struct<x, y[, z][, m]>>` (an ordered path
/// of vertices).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct LineString;

impl ExtVTable for LineString {
    type Metadata = GeoMetadata;
    // No cheap owned value like Point's `Coordinate`; expose the raw storage scalar.
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.geo.linestring");
        *ID
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.encode_to_vec())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(GeoMetadata::decode(metadata)?)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        linestring_dimension(ext_dtype.storage_dtype()).map(|_| ())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<&'a ScalarValue> {
        Ok(storage_value)
    }
}

/// Canonical line-string storage: a list of the coordinate `Struct`.
pub(crate) fn linestring_storage_dtype(dim: Dimension, nullability: Nullability) -> DType {
    let coords = coordinate_storage_dtype(dim, Nullability::NonNullable);
    DType::List(Arc::new(coords), nullability)
}

/// Validate `dtype` is `List<coordinate-struct>` and return its [`Dimension`].
pub(crate) fn linestring_dimension(dtype: &DType) -> VortexResult<Dimension> {
    let DType::List(coords, _) = dtype else {
        vortex_bail!("linestring storage must be a List of coordinates, was {dtype}");
    };
    coordinate_dimension(coords)
}

static ARROW_LINESTRING: CachedId = CachedId::new(LineStringType::NAME);

/// The `geoarrow.linestring` extension type for `dimension`, with separated (struct) coordinates
/// matching `LineString` storage.
fn linestring_type(geo_metadata: &GeoMetadata, dimension: Dimension) -> LineStringType {
    LineStringType::new(dimension.into(), geoarrow_metadata(geo_metadata))
}

/// Decode `LineString` storage (`List<coordinate>`) to `geo_types` line strings, for the geo scalar
/// functions. CRS does not affect planar geometry ops, so default metadata is used.
pub(crate) fn linestring_geometries(
    storage: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Geometry<f64>>> {
    linestring_array(storage, ctx)?
        .iter()
        .map(|geometry| -> VortexResult<Geometry<f64>> {
            Ok(geometry
                .ok_or_else(|| vortex_err!("geo: null geometry is not supported"))?
                .map_err(|e| vortex_err!("geo: geometry access failed: {e}"))?
                .to_geometry())
        })
        .collect()
}

/// Build a geoarrow `LineStringArray` from a `LineString`'s `List<coordinate>` storage.
fn linestring_array(storage: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<LineStringArray> {
    let linestring_type = linestring_type(
        &GeoMetadata::default(),
        linestring_dimension(storage.dtype())?,
    );
    let session = ctx.session().clone();
    let arrow = session.arrow().execute_arrow(storage.clone(), None, ctx)?;
    LineStringArray::try_from((arrow.as_ref(), linestring_type))
        .map_err(|e| vortex_err!("failed to construct LineStringArray: {e}"))
}

/// A validated `LineString` array (`try_from` checks the extension type).
pub struct LineStringData(ExtensionArray);

impl TryFrom<ExtensionArray> for LineStringData {
    type Error = VortexError;

    fn try_from(ext: ExtensionArray) -> Result<Self, Self::Error> {
        vortex_ensure!(
            ext.ext_dtype().is::<LineString>(),
            "expected a LineString extension array"
        );
        Ok(LineStringData(ext))
    }
}

impl LineStringData {
    /// Serialize line strings to WKB (a view array) — the form DuckDB `GEOMETRY` takes.
    pub fn to_wkb(&self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        geoarrow_to_wkb(&linestring_array(self.0.storage_array(), ctx)?)
    }
}

impl ArrowExportVTable for LineString {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_LINESTRING
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
        let geo_metadata = ext_type.metadata::<LineString>();
        let dimension = linestring_dimension(ext_type.storage_dtype())?;

        let mut field = session.to_arrow_field(name, ext_type.storage_dtype())?;
        field.try_with_extension_type(linestring_type(geo_metadata, dimension))?;

        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let is_linestring = array
            .dtype()
            .as_extension_opt()
            .map(|ext| ext.is::<LineString>())
            .unwrap_or(false);
        if !is_linestring {
            return Ok(ArrowExport::Unsupported(array));
        }

        let Ok(linestring_meta) = target.try_extension_type::<LineStringType>() else {
            return Ok(ArrowExport::Unsupported(array));
        };
        if linestring_meta.coord_type() != CoordType::Separated {
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

        // Round-trip through GeoArrow's line-string array; `into_arrow` is concrete, so wrap in `Arc`.
        let linestrings = LineStringArray::try_from((arrow_storage.as_ref(), linestring_meta))
            .map_err(|e| vortex_err!("failed to construct LineStringArray: {e}"))?;

        Ok(ArrowExport::Exported(Arc::new(linestrings.into_arrow())))
    }
}

impl ArrowImportVTable for LineString {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_LINESTRING
    }

    /// Import a `geoarrow.linestring` field as the [`LineString`] dtype. Keyed off the standard
    /// GeoArrow name, so any producer resolves here. Accepts the full `LineStringType` extension, or
    /// — for a metadata-less geometry literal — the name alone, inferring the dimension from the
    /// coordinate field names.
    fn from_arrow_field(&self, field: &Field) -> VortexResult<Option<DType>> {
        let (dimension, metadata) =
            if let Ok(linestring_meta) = field.try_extension_type::<LineStringType>() {
                vortex_ensure!(
                    linestring_meta.coord_type() == CoordType::Separated,
                    "geoarrow.linestring with interleaved coordinates is not supported; \
                 re-encode with separated (struct) coordinates"
                );
                (
                    linestring_meta.dimension().into(),
                    geo_metadata_from_arrow(linestring_meta.metadata()),
                )
            } else {
                // Literal: peel the `List` layer to the coordinate struct and read its dimension from
                // the field names (the canonical check rejects nullable coordinates).
                if field.extension_type_name() != Some(LineStringType::NAME) {
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

        let storage_dtype = linestring_storage_dtype(dimension, field.is_nullable().into());
        Ok(Some(DType::Extension(
            ExtDType::try_with_vtable(LineString, metadata, storage_dtype)?.erased(),
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
        if !ext_dtype.is::<LineString>()
            || field.try_extension_type::<LineStringType>().is_err()
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

    use super::LineString;
    use super::linestring_storage_dtype;
    use crate::extension::GeoMetadata;
    use crate::extension::coordinate::Dimension;

    fn geo_meta() -> GeoMetadata {
        GeoMetadata {
            crs: Some("EPSG:4326".to_string()),
        }
    }

    /// `LineString` accepts the canonical `List<coordinate-struct>` storage of every dimension.
    #[rstest]
    #[case::xy(Dimension::Xy)]
    #[case::xyz(Dimension::Xyz)]
    #[case::xym(Dimension::Xym)]
    #[case::xyzm(Dimension::Xyzm)]
    fn linestring_validates_every_dimension(#[case] dim: Dimension) -> VortexResult<()> {
        let storage = linestring_storage_dtype(dim, Nullability::NonNullable);
        ExtDType::<LineString>::try_new(geo_meta(), storage)?;
        Ok(())
    }

    /// Non-linestring storage is rejected at dtype construction: a bare coordinate struct (point) is
    /// not a list of coordinates.
    #[test]
    fn linestring_rejects_invalid_storage() -> VortexResult<()> {
        let primitive = DType::Primitive(PType::F64, Nullability::NonNullable);
        assert!(ExtDType::<LineString>::try_new(geo_meta(), primitive).is_err());
        Ok(())
    }
}
