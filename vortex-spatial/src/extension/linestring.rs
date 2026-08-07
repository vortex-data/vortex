// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`LineString`] geometry extension type (`vortex.st.linestring`): an ordered path of the
//! [`Point`](super::Point) coordinate struct, stored as `List<Struct<x, y[, z][, m]>>` and tagged
//! with [`SpatialMetadata`] (CRS).

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
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::InterleaveArray;
use vortex_array::arrays::ListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::ScalarValue;
use vortex_array::validity::Validity;
use vortex_arrow::ArrowExport;
use vortex_arrow::ArrowExportVTable;
use vortex_arrow::ArrowImport;
use vortex_arrow::ArrowImportVTable;
use vortex_arrow::ArrowSession;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::FromArrowArray;
use vortex_buffer::Buffer;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use super::SpatialMetadata;
use super::coordinate::Dimension;
use super::coordinate::coordinate_dimension;
use super::coordinate::coordinate_storage_dtype;
use super::geoarrow_metadata;
use super::geoarrow_to_wkb;
use super::spatial_metadata_from_arrow;

/// A line string: `geoarrow.linestring`, stored as `List<Struct<x, y[, z][, m]>>` (an ordered path
/// of vertices).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct LineString;

impl ExtVTable for LineString {
    type Metadata = SpatialMetadata;
    // No cheap owned value like Point's `Coordinate`; expose the raw storage scalar.
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.st.linestring");
        *ID
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.encode_to_vec())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(SpatialMetadata::decode(metadata)?)
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

/// Return one coordinate ordinate, filling an ordinate absent from the point dimension with zero.
fn point_ordinate(
    points: &StructArray,
    dimension: Dimension,
    name: &str,
) -> VortexResult<ArrayRef> {
    if dimension.field_names().contains(&name) {
        points.unmasked_field_by_name(name).cloned()
    } else {
        Ok(ConstantArray::new(0.0f64, points.len()).into_array())
    }
}

/// Build one native [`LineString`] per corresponding pair of point coordinate rows.
pub(crate) fn linestring_array_from_point_pairs(
    ext_dtype: &ExtDType<LineString>,
    starts: &StructArray,
    ends: &StructArray,
    validity: Validity,
) -> VortexResult<ArrayRef> {
    let len = starts.len();
    vortex_ensure_eq!(
        len,
        ends.len(),
        "spatial: line string point columns must have equal lengths"
    );
    let vertex_count = len
        .checked_mul(2)
        .ok_or_else(|| vortex_err!("spatial: two-vertex line string length overflow"))?;
    let last_offset = i32::try_from(vertex_count)
        .map_err(|_| vortex_err!("spatial: two-vertex line string offset overflow"))?;
    let row_count = u32::try_from(len)
        .map_err(|_| vortex_err!("spatial: two-vertex line string row count overflow"))?;
    let dimension = linestring_dimension(ext_dtype.storage_dtype())?;
    let start_dimension = coordinate_dimension(starts.dtype())?;
    let end_dimension = coordinate_dimension(ends.dtype())?;

    let array_indices = PrimitiveArray::from_iter((0..len).flat_map(|_| [0u8, 1])).into_array();
    let row_indices = Buffer::from_iter((0..row_count).flat_map(|row| [row; 2])).into_array();

    let ordinates = dimension
        .field_names()
        .iter()
        .map(|name| {
            Ok(InterleaveArray::try_new(
                vec![
                    point_ordinate(starts, start_dimension, name)?,
                    point_ordinate(ends, end_dimension, name)?,
                ],
                array_indices.clone(),
                row_indices.clone(),
            )?
            .into_array())
        })
        .collect::<VortexResult<Vec<_>>>()?;

    let vertices = StructArray::try_new(
        FieldNames::from(dimension.field_names()),
        ordinates,
        vertex_count,
        Validity::NonNullable,
    )?
    .into_array();

    let offsets = Buffer::from_iter((0..=last_offset).step_by(2)).into_array();
    let storage = ListArray::try_new(vertices, offsets, validity)?.into_array();

    Ok(ExtensionArray::try_new(ext_dtype.clone().erased(), storage)?.into_array())
}

static ARROW_LINESTRING: CachedId = CachedId::new(LineStringType::NAME);

/// The `geoarrow.linestring` extension type for `dimension`, with separated (struct) coordinates
/// matching `LineString` storage.
fn linestring_type(spatial_metadata: &SpatialMetadata, dimension: Dimension) -> LineStringType {
    LineStringType::new(dimension.into(), geoarrow_metadata(spatial_metadata))
}

/// Decode `LineString` storage (`List<coordinate>`) to `geo_types` line strings, for the spatial scalar
/// functions. CRS does not affect planar geometry ops, so default metadata is used.
pub(crate) fn linestring_geometries(
    storage: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Geometry<f64>>> {
    linestring_array(storage, ctx)?
        .iter()
        .map(|geometry| -> VortexResult<Geometry<f64>> {
            Ok(geometry
                .ok_or_else(|| vortex_err!("spatial: null geometry is not supported"))?
                .map_err(|e| vortex_err!("spatial: geometry access failed: {e}"))?
                .to_geometry())
        })
        .collect()
}

/// Build a geoarrow `LineStringArray` from a `LineString`'s `List<coordinate>` storage.
fn linestring_array(storage: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<LineStringArray> {
    let linestring_type = linestring_type(
        &SpatialMetadata::default(),
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
        let spatial_metadata = ext_type.metadata::<LineString>();
        let dimension = linestring_dimension(ext_type.storage_dtype())?;

        let mut field = session.to_arrow_field(name, ext_type.storage_dtype())?;
        field.try_with_extension_type(linestring_type(spatial_metadata, dimension))?;

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
    fn from_arrow_field(
        &self,
        field: &Field,
        session: &ArrowSession,
    ) -> VortexResult<Option<DType>> {
        let (dimension, metadata) =
            if let Ok(linestring_meta) = field.try_extension_type::<LineStringType>() {
                vortex_ensure!(
                    linestring_meta.coord_type() == CoordType::Separated,
                    "geoarrow.linestring with interleaved coordinates is not supported; \
                 re-encode with separated (struct) coordinates"
                );
                (
                    linestring_meta.dimension().into(),
                    spatial_metadata_from_arrow(linestring_meta.metadata()),
                )
            } else {
                // Literal: peel the `List` layer to the coordinate struct and read its dimension from
                // the field names (the canonical check rejects nullable coordinates).
                if field.extension_type_name() != Some(LineStringType::NAME) {
                    return Ok(None);
                }
                let Ok(DType::List(coords, _)) =
                    session.from_arrow_datatype(field.data_type(), field.is_nullable().into())
                else {
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
    use crate::extension::SpatialMetadata;
    use crate::extension::coordinate::Dimension;

    fn spatial_meta() -> SpatialMetadata {
        SpatialMetadata {
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
        ExtDType::<LineString>::try_new(spatial_meta(), storage)?;
        Ok(())
    }

    /// Non-linestring storage is rejected at dtype construction: a bare coordinate struct (point) is
    /// not a list of coordinates.
    #[test]
    fn linestring_rejects_invalid_storage() -> VortexResult<()> {
        let primitive = DType::Primitive(PType::F64, Nullability::NonNullable);
        assert!(ExtDType::<LineString>::try_new(spatial_meta(), primitive).is_err());
        Ok(())
    }
}
