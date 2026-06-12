// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`Point`] geometry extension type (`vortex.geo.point`): a location stored columnarly as
//! `Struct<x, y[, z][, m]>` of non-nullable `f64` — the four GeoArrow dimensions XY, XYZ, XYM,
//! XYZM — tagged with [`GeoMetadata`] (CRS). `z` is an optional elevation and `m` an optional
//! measure: an arbitrary per-point value such as distance along a route or a timestamp.

use prost::Message;
use vortex_array::dtype::extension::ExtDType;
use vortex_array::dtype::extension::ExtId;
use vortex_array::dtype::extension::ExtVTable;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_error::VortexResult;

use super::GeoMetadata;
use super::coordinate::Coordinate;
use super::coordinate::coordinate_dimension;
use super::coordinate::coordinate_from_struct;

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

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::StructArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::FieldNames;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::Point;
    use crate::extension::GeoMetadata;
    use crate::extension::coordinate::Coordinate;
    use crate::extension::coordinate::Dimension;
    use crate::extension::coordinate::coordinate_dimension;
    use crate::extension::coordinate::coordinate_from_scalar;

    fn geo_meta() -> GeoMetadata {
        GeoMetadata {
            crs: Some("EPSG:4326".to_string()),
        }
    }

    /// A coordinate storage dtype with the given field names, non-nullable `f64` per field.
    fn coordinate_dtype(names: &[&'static str]) -> DType {
        let fields = std::iter::repeat_n(
            DType::Primitive(PType::F64, Nullability::NonNullable),
            names.len(),
        )
        .collect::<Vec<_>>();
        DType::Struct(
            StructFields::new(FieldNames::from(names), fields),
            Nullability::NonNullable,
        )
    }

    /// `Point` accepts every GeoArrow dimension; the canonical field names round-trip to their
    /// dimension, so a `z`/`m` swap or a mislabel would be caught.
    #[test]
    fn point_validates_every_dimension() -> VortexResult<()> {
        let cases = [
            (Dimension::Xy, ["x", "y"].as_slice()),
            (Dimension::Xyz, ["x", "y", "z"].as_slice()),
            (Dimension::Xym, ["x", "y", "m"].as_slice()),
            (Dimension::Xyzm, ["x", "y", "z", "m"].as_slice()),
        ];
        for (dim, names) in cases {
            let storage = coordinate_dtype(names);
            assert_eq!(coordinate_dimension(&storage)?, dim);
            ExtDType::<Point>::try_new(geo_meta(), storage)?;
        }
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
        let session = VortexSession::empty().with::<ArraySession>();
        let mut ctx = session.create_execution_ctx();

        let storage = StructArray::from_fields(&[
            (
                "x",
                PrimitiveArray::from_iter(vec![1.0f64, -111.7610]).into_array(),
            ),
            (
                "y",
                PrimitiveArray::from_iter(vec![2.0f64, 34.8697]).into_array(),
            ),
        ])?
        .into_array();
        let dtype = ExtDType::<Point>::try_new(geo_meta(), storage.dtype().clone())?;
        let points = ExtensionArray::new(dtype.erased(), storage).into_array();

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
