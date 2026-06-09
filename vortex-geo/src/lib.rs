// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_array::arrow::ArrowSessionExt;
use vortex_array::dtype::session::DTypeSessionExt;
use vortex_session::VortexSession;

use crate::extension::WellKnownBinary;

pub mod extension;
/// Set up a session with support for geospatial extension types, encodings and layouts.
pub fn initialize(session: &VortexSession) {
    // register geospatial extension types
    session.dtypes().register(WellKnownBinary);
    session.arrow().register_exporter(Arc::new(WellKnownBinary));
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use arrow_array::Array as _;
    use arrow_array::cast::AsArray;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::extension::ExtensionType as _;
    use geo_traits::to_geo::ToGeoGeometry;
    use geo_types::Coord;
    use geo_types::Geometry;
    use geo_types::LineString;
    use geo_types::Polygon;
    use geoarrow::datatypes::WkbType;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::varbin::builder::VarBinBuilder;
    use vortex_array::arrow::ArrowSessionExt;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::dtype::extension::ExtVTable;
    use vortex_array::session::ArraySession;
    use vortex_error::VortexResult;
    use vortex_error::vortex_err;
    use vortex_session::VortexSession;
    use wkb::writer::WriteOptions;

    use crate::extension::GeoMetadata;
    use crate::extension::WellKnownBinary;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty().with::<ArraySession>();
        crate::initialize(&session);
        session
    });

    #[test]
    fn test_array() -> VortexResult<()> {
        let mut execution_ctx = SESSION.create_execution_ctx();

        let mut buf = Vec::new();

        let mut builder = VarBinBuilder::<i32>::with_capacity(3);

        let polygon = Geometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord::zero(),
                Coord { x: 100.0, y: 0.0 },
                Coord { x: 100.0, y: 100.0 },
                Coord { x: 0.0, y: 100.0 },
                Coord::zero(),
            ]),
            vec![],
        ));

        // We should always prefer to write little-endian, which is the default option.
        wkb::writer::write_geometry(&mut buf, &polygon, &WriteOptions::default())
            .map_err(|e| vortex_err!("writing WKB failed: {e}"))?;

        // Push same polygon 3 times
        builder.append_value(&buf);
        builder.append_value(&buf);
        builder.append_value(&buf);

        let dtype = ExtDType::<WellKnownBinary>::try_new(
            GeoMetadata {
                crs: Some("EPSG:4326".to_string()),
            },
            DType::Binary(Nullability::NonNullable),
        )?;

        let array = builder.finish(DType::Binary(Nullability::NonNullable));
        let array = ExtensionArray::new(dtype.clone().erased(), array.into_array()).into_array();

        for idx in 0..3 {
            let geom = array.execute_scalar(idx, &mut execution_ctx)?;
            let wkb = WellKnownBinary::unpack_native(&dtype, geom.value().unwrap())?;

            assert_eq!(wkb.to_geometry(), polygon);
        }

        Ok(())
    }

    fn wkb_extension_array() -> VortexResult<(Vec<u8>, vortex_array::ArrayRef)> {
        let polygon = Geometry::Polygon(Polygon::new(
            LineString::new(vec![
                Coord::zero(),
                Coord { x: 100.0, y: 0.0 },
                Coord { x: 100.0, y: 100.0 },
                Coord { x: 0.0, y: 100.0 },
                Coord::zero(),
            ]),
            vec![],
        ));
        let mut buf = Vec::new();
        wkb::writer::write_geometry(&mut buf, &polygon, &WriteOptions::default())
            .map_err(|e| vortex_err!("writing WKB failed: {e}"))?;

        let mut builder = VarBinBuilder::<i32>::with_capacity(3);
        builder.append_value(&buf);
        builder.append_value(&buf);
        builder.append_value(&buf);

        let dtype = ExtDType::<WellKnownBinary>::try_new(
            GeoMetadata {
                crs: Some("EPSG:4326".to_string()),
            },
            DType::Binary(Nullability::NonNullable),
        )?;
        let storage = builder.finish(DType::Binary(Nullability::NonNullable));
        let array = ExtensionArray::new(dtype.erased(), storage.into_array()).into_array();
        Ok((buf, array))
    }

    #[test]
    fn export_arrow_field_carries_wkb_extension() -> VortexResult<()> {
        let (_, array) = wkb_extension_array()?;
        let field = SESSION.arrow().to_arrow_field("geom", array.dtype())?;
        assert_eq!(field.extension_type_name(), Some(WkbType::NAME));
        // Vortex's canonical mapping of `DType::Binary` is Arrow's `BinaryView`.
        assert_eq!(field.data_type(), &DataType::BinaryView);
        Ok(())
    }

    #[test]
    fn execute_arrow_exports_wkb_to_binary() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (wkb_bytes, array) = wkb_extension_array()?;

        let field = Field::new("geom", DataType::Binary, false)
            .with_extension_type(WkbType::new(Default::default()));
        let exported = SESSION
            .arrow()
            .execute_arrow(array, Some(&field), &mut ctx)?;

        assert_eq!(exported.data_type(), &DataType::Binary);
        let binary = exported.as_binary::<i32>();
        assert_eq!(binary.len(), 3);
        for idx in 0..3 {
            assert_eq!(binary.value(idx), wkb_bytes.as_slice());
        }
        Ok(())
    }

    #[test]
    fn execute_arrow_exports_wkb_to_large_binary() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (wkb_bytes, array) = wkb_extension_array()?;

        let field = Field::new("geom", DataType::LargeBinary, false)
            .with_extension_type(WkbType::new(Default::default()));
        let exported = SESSION
            .arrow()
            .execute_arrow(array, Some(&field), &mut ctx)?;

        assert_eq!(exported.data_type(), &DataType::LargeBinary);
        let binary = exported.as_binary::<i64>();
        assert_eq!(binary.len(), 3);
        for idx in 0..3 {
            assert_eq!(binary.value(idx), wkb_bytes.as_slice());
        }
        Ok(())
    }

    #[test]
    fn execute_arrow_exports_wkb_to_binary_view() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let (wkb_bytes, array) = wkb_extension_array()?;

        let field = Field::new("geom", DataType::BinaryView, false)
            .with_extension_type(WkbType::new(Default::default()));
        let exported = SESSION
            .arrow()
            .execute_arrow(array, Some(&field), &mut ctx)?;

        assert_eq!(exported.data_type(), &DataType::BinaryView);
        let binary = exported.as_binary_view();
        assert_eq!(binary.len(), 3);
        for idx in 0..3 {
            assert_eq!(binary.value(idx), wkb_bytes.as_slice());
        }
        Ok(())
    }
}
