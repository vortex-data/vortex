// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Native multi-line-string geometry (`vortex.st.multilinestring`).
//!
//! [`MultiLineString`] stores line strings as `List<List<Struct<x, y[, z][, m]>>>`.
//! [`SpatialMetadata`] declares the coordinate dimension and optional coordinate reference system.
//! Coordinates use non-nullable `f64` fields, and only the outer list can be nullable. The storage
//! layout matches [`Polygon`](super::Polygon), and the extension ID identifies the geometry type.

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType;
use geo_traits::to_geo::ToGeoGeometry;
use geo_types::Geometry;
use geoarrow::array::GeoArrowArrayAccessor;
use geoarrow::array::IntoArrow;
use geoarrow::array::MultiLineStringArray;
use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::MultiLineStringType;
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
use vortex_arrow::ArrowExportOptions;
use vortex_arrow::ArrowExtVTable;
use vortex_arrow::ArrowImport;
use vortex_arrow::ArrowMetadataPolicy;
use vortex_arrow::ArrowSession;
use vortex_arrow::ArrowSessionExt;
use vortex_arrow::ExecuteArrowExt;
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
use super::spatial_metadata_from_arrow;

/// A multilinestring: `geoarrow.multilinestring`, stored as `List<List<Struct<x, y[, z][, m]>>>`
/// (line strings of vertices).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct MultiLineString;

impl ExtVTable for MultiLineString {
    type Metadata = SpatialMetadata;
    type StoredMetadata = super::SpatialStoredMetadata;
    // No cheap owned value like Point's `Coordinate`; expose the raw storage scalar.
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.st.multilinestring");
        *ID
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.serialize())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::StoredMetadata> {
        super::SpatialStoredMetadata::deserialize(metadata)
    }

    fn metadata_from_storage(
        &self,
        stored_metadata: Self::StoredMetadata,
        storage_dtype: &DType,
    ) -> VortexResult<Self::Metadata> {
        Ok(SpatialMetadata::from_stored_metadata(
            stored_metadata,
            Some(multilinestring_dimension(storage_dtype)?),
        ))
    }

    fn storage_dtype(metadata: &Self::Metadata) -> DType {
        multilinestring_storage_dtype(metadata.native_dimension(), Nullability::NonNullable)
    }

    fn validate_metadata(metadata: &Self::Metadata) -> VortexResult<()> {
        metadata.validate_native(MultiLineString.id())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<&'a ScalarValue> {
        Ok(storage_value)
    }
}

/// Canonical multilinestring storage: an outer list of line strings, each a list of the coordinate
/// `Struct`.
pub(crate) fn multilinestring_storage_dtype(dim: Dimension, nullability: Nullability) -> DType {
    let coords = coordinate_storage_dtype(dim, Nullability::NonNullable);
    let line = DType::List(Arc::new(coords), Nullability::NonNullable);
    DType::List(Arc::new(line), nullability)
}

/// Infers the coordinate [`Dimension`] of `List<List<coordinate-struct>>` storage.
///
/// Checks the list nesting and delegates coordinate field-name inference to
/// [`coordinate_dimension`]. Reads validate the complete storage dtype against the declaration.
pub(crate) fn multilinestring_dimension(dtype: &DType) -> VortexResult<Dimension> {
    let DType::List(line, _) = dtype else {
        vortex_bail!("multilinestring storage must be a List of line strings, got {dtype}");
    };
    let DType::List(coords, _) = line.as_ref() else {
        vortex_bail!("multilinestring line storage must be a List of coordinates, got {line}");
    };
    coordinate_dimension(coords)
}

static ARROW_MULTILINESTRING: CachedId = CachedId::new(MultiLineStringType::NAME);

/// The `geoarrow.multilinestring` type for `dimension`, with separated (struct) coordinates.
fn multilinestring_type(
    spatial_metadata: &SpatialMetadata,
    dimension: Dimension,
) -> MultiLineStringType {
    MultiLineStringType::new(dimension.into(), geoarrow_metadata(spatial_metadata))
}

/// Decode storage to `geo_types` for the spatial scalar functions (CRS is irrelevant to planar ops).
pub(crate) fn multilinestring_geometries(
    storage: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Geometry<f64>>> {
    multilinestring_array(storage, ctx)?
        .iter()
        .map(|geometry| -> VortexResult<Geometry<f64>> {
            Ok(geometry
                .ok_or_else(|| vortex_err!("spatial: null geometry is not supported"))?
                .map_err(|e| vortex_err!("spatial: geometry access failed: {e}"))?
                .to_geometry())
        })
        .collect()
}

/// Build a geoarrow `MultiLineStringArray` from the `MultiLineString` storage.
fn multilinestring_array(
    storage: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<MultiLineStringArray> {
    let multilinestring_type = multilinestring_type(
        &SpatialMetadata::default(),
        multilinestring_dimension(storage.dtype())?,
    );
    let arrow = storage.clone().execute_arrow(None, ctx)?;
    MultiLineStringArray::try_from((arrow.as_ref(), multilinestring_type))
        .map_err(|e| vortex_err!("failed to construct MultiLineStringArray: {e}"))
}

/// A validated `MultiLineString` array (`try_from` checks the extension type).
pub struct MultiLineStringData(ExtensionArray);

impl TryFrom<ExtensionArray> for MultiLineStringData {
    type Error = VortexError;

    fn try_from(ext: ExtensionArray) -> Result<Self, Self::Error> {
        vortex_ensure!(
            ext.ext_dtype().is::<MultiLineString>(),
            "expected a MultiLineString extension array"
        );
        Ok(MultiLineStringData(ext))
    }
}

impl MultiLineStringData {
    /// Serialize multilinestrings to WKB (a view array) — the form DuckDB `GEOMETRY` takes.
    pub fn to_wkb(&self, ctx: &mut ExecutionCtx) -> VortexResult<ArrayRef> {
        geoarrow_to_wkb(
            &multilinestring_array(self.0.storage_array(), ctx)?,
            &ctx.session().arrow(),
        )
    }
}

impl ArrowExtVTable for MultiLineString {
    fn metadata_policy() -> ArrowMetadataPolicy {
        ArrowMetadataPolicy::ArrowNative
    }

    fn arrow_ext_id(&self) -> Id {
        *ARROW_MULTILINESTRING
    }

    fn to_arrow_field(
        &self,
        name: &str,
        ext_dtype: &ExtDType<Self>,
        _session: &ArrowSession,
    ) -> VortexResult<Option<Field>> {
        let spatial_metadata = ext_dtype.metadata();
        let dimension = spatial_metadata.native_dimension();

        let arrow_type = multilinestring_type(spatial_metadata, dimension);
        let mut field = Field::new(
            name,
            arrow_type.data_type(),
            ext_dtype.storage_dtype().is_nullable(),
        );
        field.try_with_extension_type(arrow_type)?;

        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        options: &ArrowExportOptions,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let Ok(multilinestring_meta) = target.try_extension_type::<MultiLineStringType>() else {
            return Ok(ArrowExport::Unsupported(array));
        };
        if multilinestring_meta.coord_type() != CoordType::Separated {
            return Ok(ArrowExport::Unsupported(array));
        }

        let executed = array.execute::<ExtensionArray>(ctx)?;
        let storage = executed.storage_array().clone();

        let storage_field = Field::new(
            String::new(),
            target.data_type().clone(),
            target.is_nullable(),
        );
        let arrow_storage =
            storage.execute_arrow_with_options(Some(&storage_field), options, ctx)?;

        let multilinestrings =
            MultiLineStringArray::try_from((arrow_storage.as_ref(), multilinestring_meta))
                .map_err(|e| vortex_err!("failed to construct MultiLineStringArray: {e}"))?;

        Ok(ArrowExport::Exported(Arc::new(
            multilinestrings.into_arrow(),
        )))
    }

    fn from_arrow_field(
        field: &Field,
        session: &ArrowSession,
    ) -> VortexResult<Option<ExtDType<Self>>> {
        let (dimension, metadata) =
            if let Ok(multilinestring_meta) = field.try_extension_type::<MultiLineStringType>() {
                vortex_ensure!(
                    multilinestring_meta.coord_type() == CoordType::Separated,
                    "geoarrow.multilinestring with interleaved coordinates is not supported; \
             re-encode with separated (struct) coordinates"
                );
                (
                    multilinestring_meta.dimension().into(),
                    spatial_metadata_from_arrow(multilinestring_meta.metadata()),
                )
            } else {
                // Literal: peel the two `List` layers to the coordinate struct and read its dimension
                // from the field names (the canonical check rejects nullable coordinates).
                if field.extension_type_name() != Some(MultiLineStringType::NAME) {
                    return Ok(None);
                }
                let Ok(DType::List(line, _)) =
                    session.from_arrow_datatype(field.data_type(), field.is_nullable().into())
                else {
                    return Ok(None);
                };
                let DType::List(coords, _) = line.as_ref() else {
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

        Ok(Some(ExtDType::try_with_vtable(
            MultiLineString,
            metadata.with_dimension(dimension),
            field.is_nullable().into(),
        )?))
    }

    fn from_arrow_array(
        array: ArrowArrayRef,
        field: &Field,
        ext_dtype: &ExtDType<Self>,
        session: &ArrowSession,
    ) -> VortexResult<ArrowImport> {
        if field.try_extension_type::<MultiLineStringType>().is_err()
            || !matches!(array.data_type(), DataType::List(_))
        {
            return Ok(ArrowImport::Unsupported(array));
        }

        let storage = session.from_arrow_array(array, field.is_nullable())?;
        Ok(ArrowImport::Imported(
            ExtensionArray::try_new(ext_dtype.clone().erased(), storage)?.into_array(),
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
    use vortex_array::dtype::extension::ExtDTypePlugin;
    use vortex_error::VortexResult;

    use super::MultiLineString;
    use super::multilinestring_storage_dtype;
    use crate::extension::SpatialMetadata;
    use crate::extension::coordinate::Dimension;
    use crate::extension::coordinate::coordinate_storage_dtype;

    fn spatial_meta() -> SpatialMetadata {
        SpatialMetadata::new(Some("EPSG:4326".to_string()))
    }

    /// `MultiLineString` reads the canonical `List<List<coordinate-struct>>` storage of every
    /// dimension.
    #[rstest]
    #[case::xy(Dimension::Xy)]
    #[case::xyz(Dimension::Xyz)]
    #[case::xym(Dimension::Xym)]
    #[case::xyzm(Dimension::Xyzm)]
    fn multilinestring_reads_every_dimension(#[case] dim: Dimension) -> VortexResult<()> {
        let storage = multilinestring_storage_dtype(dim, Nullability::NonNullable);
        MultiLineString.deserialize(&spatial_meta().serialize(), storage)?;
        Ok(())
    }

    /// Non-multilinestring storage fails to read, for example `f64` or a single list (linestring).
    #[test]
    fn multilinestring_rejects_invalid_storage() -> VortexResult<()> {
        let primitive = DType::Primitive(PType::F64, Nullability::NonNullable);
        assert!(
            MultiLineString
                .deserialize(&spatial_meta().serialize(), primitive)
                .is_err()
        );

        // A bare list of coordinates is a single line string, not a multilinestring.
        let coords = coordinate_storage_dtype(Dimension::Xy, Nullability::NonNullable);
        let line = DType::List(Arc::new(coords), Nullability::NonNullable);
        assert!(
            MultiLineString
                .deserialize(&spatial_meta().serialize(), line)
                .is_err()
        );
        Ok(())
    }
}
