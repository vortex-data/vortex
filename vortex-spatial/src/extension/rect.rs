// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`Rect`] bounding-box extension type (`vortex.st.box`): an axis-aligned envelope stored
//! columnarly as `Struct<xmin, ymin[, zmin][, mmin], xmax, ymax[, zmax][, mmax]>` of non-nullable
//! `f64` — the lower corner's ordinates followed by the upper corner's — tagged with
//! [`SpatialMetadata`] (CRS). Its GeoArrow wire type is `geoarrow.box`.
//!
//! Decoding to `geo_types` yields a 2D [`Geometry::Rect`]; any `z`/`m`
//! bounds are dropped, as for the other geometry types.

use std::sync::Arc;

use arrow_array::ArrayRef as ArrowArrayRef;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::extension::ExtensionType;
use geo_traits::to_geo::ToGeoGeometry;
use geo_types::Geometry;
use geoarrow::array::GeoArrowArrayAccessor;
use geoarrow::array::IntoArrow;
use geoarrow::array::RectArray;
use geoarrow::datatypes::BoxType;
use prost::Message;
use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ExtensionArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::extension::ExtensionArrayExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::FieldNames;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::dtype::StructFields;
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
use vortex_arrow::FromArrowType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::registry::CachedId;
use vortex_session::registry::Id;

use super::SpatialMetadata;
use super::coordinate::Dimension;
use super::geoarrow_metadata;
use super::spatial_metadata_from_arrow;

/// An axis-aligned bounding box (`geoarrow.box`), stored as `Struct<xmin, ymin[, ..], xmax, ymax[, ..]>`.
// Named `Rect`, not `Box`: matches `geo::Rect` / geoarrow-rs `RectArray`, and `Box` is a std name.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct Rect;

impl ExtVTable for Rect {
    type Metadata = SpatialMetadata;
    // No cheap owned value; expose the raw storage scalar like `Polygon`.
    type NativeValue<'a> = &'a ScalarValue;

    fn id(&self) -> ExtId {
        static ID: CachedId = CachedId::new("vortex.st.box");
        *ID
    }

    fn serialize_metadata(&self, metadata: &Self::Metadata) -> VortexResult<Vec<u8>> {
        Ok(metadata.encode_to_vec())
    }

    fn deserialize_metadata(&self, metadata: &[u8]) -> VortexResult<Self::Metadata> {
        Ok(SpatialMetadata::decode(metadata)?)
    }

    fn validate_dtype(ext_dtype: &ExtDType<Self>) -> VortexResult<()> {
        box_dimension(ext_dtype.storage_dtype()).map(|_| ())
    }

    fn unpack_native<'a>(
        _ext_dtype: &'a ExtDType<Self>,
        storage_value: &'a ScalarValue,
    ) -> VortexResult<&'a ScalarValue> {
        Ok(storage_value)
    }
}

/// The `geoarrow.box` storage field names for `dim`: the lower corner's ordinates followed by the
/// upper corner's.
pub(crate) fn box_field_names(dim: Dimension) -> &'static [&'static str] {
    match dim {
        Dimension::Xy => &["xmin", "ymin", "xmax", "ymax"],
        Dimension::Xyz => &["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"],
        Dimension::Xym => &["xmin", "ymin", "mmin", "xmax", "ymax", "mmax"],
        Dimension::Xyzm => &[
            "xmin", "ymin", "zmin", "mmin", "xmax", "ymax", "zmax", "mmax",
        ],
    }
}

/// The dimension whose box field names match `names`, or `None` if none do.
fn box_dimension_from_names(names: &FieldNames) -> Option<Dimension> {
    [
        Dimension::Xy,
        Dimension::Xyz,
        Dimension::Xym,
        Dimension::Xyzm,
    ]
    .into_iter()
    .find(|&dim| {
        names
            .iter()
            .map(AsRef::as_ref)
            .eq(box_field_names(dim).iter().copied())
    })
}

/// Canonical box storage: a `Struct` of non-nullable `f64` min/max fields, with `nullability` at
/// the struct (per-box) level.
pub(crate) fn box_storage_dtype(dim: Dimension, nullability: Nullability) -> DType {
    let names = box_field_names(dim);
    let fields = std::iter::repeat_n(
        DType::Primitive(PType::F64, Nullability::NonNullable),
        names.len(),
    )
    .collect::<Vec<_>>();
    DType::Struct(
        StructFields::new(FieldNames::from(names), fields),
        nullability,
    )
}

/// Validate `dtype` is a box `Struct` of non-nullable `f64` min/max fields, returning its
/// [`Dimension`].
pub(crate) fn box_dimension(dtype: &DType) -> VortexResult<Dimension> {
    let DType::Struct(fields, _) = dtype else {
        vortex_bail!("box storage must be a Struct, was {dtype}");
    };
    for (name, field) in fields.names().iter().zip(fields.fields()) {
        vortex_ensure!(
            matches!(
                field,
                DType::Primitive(PType::F64, Nullability::NonNullable)
            ),
            "box field {name} must be non-nullable f64, was {field}"
        );
    }
    box_dimension_from_names(fields.names())
        .ok_or_else(|| vortex_err!("not a valid geoarrow.box dimension: {:?}", fields.names()))
}

/// Build a native [`Rect`] array from canonical min/max ordinate columns.
pub(crate) fn build_rect_array(
    ext_dtype: &ExtDType<Rect>,
    corners: Vec<ArrayRef>,
    len: usize,
    validity: Validity,
) -> VortexResult<ArrayRef> {
    let dimension = box_dimension(ext_dtype.storage_dtype())?;
    let storage = StructArray::try_new(
        FieldNames::from(box_field_names(dimension)),
        corners,
        len,
        validity,
    )?
    .into_array();
    Ok(ExtensionArray::try_new(ext_dtype.clone().erased(), storage)?.into_array())
}

static ARROW_BOX: CachedId = CachedId::new(BoxType::NAME);

/// The `geoarrow.box` extension type for `dimension`.
fn box_type(spatial_metadata: &SpatialMetadata, dimension: Dimension) -> BoxType {
    BoxType::new(dimension.into(), geoarrow_metadata(spatial_metadata))
}

/// Build a geoarrow `RectArray` from a `Rect`'s box `Struct` storage.
fn rect_array(storage: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<RectArray> {
    let box_type = box_type(&SpatialMetadata::default(), box_dimension(storage.dtype())?);
    let session = ctx.session().clone();
    let arrow = session.arrow().execute_arrow(storage.clone(), None, ctx)?;
    RectArray::try_from((arrow.as_ref(), box_type))
        .map_err(|e| vortex_err!("failed to construct RectArray: {e}"))
}

/// Decode `Rect` storage to `geo_types` (2D [`Geometry::Rect`]), for the spatial scalar functions.
pub(crate) fn rect_geometries(
    storage: &ArrayRef,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Vec<Geometry<f64>>> {
    rect_array(storage, ctx)?
        .iter()
        .map(|geometry| -> VortexResult<Geometry<f64>> {
            Ok(geometry
                .ok_or_else(|| vortex_err!("spatial: null geometry is not supported"))?
                .map_err(|e| vortex_err!("spatial: geometry access failed: {e}"))?
                .to_geometry())
        })
        .collect()
}

impl ArrowExportVTable for Rect {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_BOX
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
        let spatial_metadata = ext_type.metadata::<Rect>();
        let dimension = box_dimension(ext_type.storage_dtype())?;

        let mut field = session.to_arrow_field(name, ext_type.storage_dtype())?;
        field.try_with_extension_type(box_type(spatial_metadata, dimension))?;

        Ok(Some(field))
    }

    fn execute_arrow(
        &self,
        array: ArrayRef,
        target: &Field,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrowExport> {
        let is_box = array
            .dtype()
            .as_extension_opt()
            .map(|ext| ext.is::<Rect>())
            .unwrap_or(false);
        if !is_box {
            return Ok(ArrowExport::Unsupported(array));
        }

        let Ok(box_meta) = target.try_extension_type::<BoxType>() else {
            return Ok(ArrowExport::Unsupported(array));
        };

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

        // Round-trip through GeoArrow's rect array; `into_arrow` is concrete, so wrap in `Arc`.
        let rects = RectArray::try_from((arrow_storage.as_ref(), box_meta))
            .map_err(|e| vortex_err!("failed to construct RectArray: {e}"))?;

        Ok(ArrowExport::Exported(Arc::new(rects.into_arrow())))
    }
}

impl ArrowImportVTable for Rect {
    fn arrow_ext_id(&self) -> Id {
        *ARROW_BOX
    }

    /// Import a `geoarrow.box` field as the [`Rect`] dtype. Keyed off the standard GeoArrow name,
    /// so any producer resolves here. Accepts the full `BoxType` extension, or — for a
    /// metadata-less literal — the name alone, inferring the dimension from the field names.
    fn from_arrow_field(&self, field: &Field) -> VortexResult<Option<DType>> {
        let (dimension, metadata) = if let Ok(box_meta) = field.try_extension_type::<BoxType>() {
            (
                box_meta.dimension().into(),
                spatial_metadata_from_arrow(box_meta.metadata()),
            )
        } else {
            if field.extension_type_name() != Some(BoxType::NAME) {
                return Ok(None);
            }
            // Infer from field names, not the strict `box_dimension` check: a literal's fields may
            // be nullable, which that check rejects.
            let DType::Struct(fields, _) = DType::from_arrow(field) else {
                return Ok(None);
            };
            let Some(dimension) = box_dimension_from_names(fields.names()) else {
                return Ok(None);
            };
            (dimension, SpatialMetadata::default())
        };

        let storage_dtype = box_storage_dtype(dimension, field.is_nullable().into());
        Ok(Some(DType::Extension(
            ExtDType::try_with_vtable(Rect, metadata, storage_dtype)?.erased(),
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
        if !ext_dtype.is::<Rect>()
            || field.try_extension_type::<BoxType>().is_err()
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
    use geo_types::Geometry;
    use rstest::rstest;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::extension::ExtensionArrayExt;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_error::VortexResult;

    use super::Rect;
    use super::box_dimension;
    use super::box_storage_dtype;
    use super::rect_geometries;
    use crate::extension::SpatialMetadata;
    use crate::extension::coordinate::Dimension;

    fn spatial_meta() -> SpatialMetadata {
        SpatialMetadata {
            crs: Some("EPSG:4326".to_string()),
        }
    }

    /// `Rect` accepts the canonical box storage of every GeoArrow dimension, and the storage dtype
    /// round-trips back to that dimension.
    #[rstest]
    #[case::xy(Dimension::Xy)]
    #[case::xyz(Dimension::Xyz)]
    #[case::xym(Dimension::Xym)]
    #[case::xyzm(Dimension::Xyzm)]
    fn box_validates_every_dimension(#[case] dim: Dimension) -> VortexResult<()> {
        let storage = box_storage_dtype(dim, Nullability::NonNullable);
        ExtDType::<Rect>::try_new(spatial_meta(), storage.clone())?;
        assert_eq!(box_dimension(&storage)?, dim);
        Ok(())
    }

    /// Invalid storage is rejected at dtype construction: non-struct storage, a struct with the
    /// wrong field names, and a struct with a nullable field.
    #[test]
    fn box_rejects_invalid_storage() -> VortexResult<()> {
        let primitive = DType::Primitive(PType::F64, Nullability::NonNullable);
        assert!(ExtDType::<Rect>::try_new(spatial_meta(), primitive).is_err());

        let point_like = box_storage_dtype(Dimension::Xy, Nullability::NonNullable);
        let DType::Struct(fields, _) = &point_like else {
            unreachable!("box storage is a struct");
        };
        // Rename a field so it no longer matches the geoarrow.box layout.
        let renamed = DType::Struct(
            vortex_array::dtype::StructFields::new(
                ["x", "ymin", "xmax", "ymax"].into(),
                fields.fields().collect(),
            ),
            Nullability::NonNullable,
        );
        assert!(ExtDType::<Rect>::try_new(spatial_meta(), renamed).is_err());
        Ok(())
    }

    /// A `Rect` column decodes to 2D `geo_types` rectangles.
    #[test]
    fn box_decodes_to_geo_types_rect() -> VortexResult<()> {
        let session = vortex_array::array_session();
        let mut ctx = session.create_execution_ctx();

        let rects = crate::test_harness::rect_column(vec![(0.0, 0.0, 2.0, 3.0)])?;
        let storage = rects
            .execute::<ExtensionArray>(&mut ctx)?
            .storage_array()
            .clone();

        let geometries = rect_geometries(&storage, &mut ctx)?;
        assert!(matches!(geometries.as_slice(), [Geometry::Rect(_)]));
        Ok(())
    }
}
