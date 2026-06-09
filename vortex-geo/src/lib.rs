// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::session::DTypeSessionExt;
use vortex_session::VortexSession;

use crate::extension::WellKnownBinary;

pub mod extension;
/// Set up a session with support for geospatial extension types, encodings and layouts.
pub fn initialize(session: &VortexSession) {
    // register geospatial extension types
    session.dtypes().register(WellKnownBinary);
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use geo_traits::to_geo::ToGeoGeometry;
    use geo_types::Coord;
    use geo_types::Geometry;
    use geo_types::LineString;
    use geo_types::Polygon;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::ExtensionArray;
    use vortex_array::arrays::varbin::builder::VarBinBuilder;
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
}
