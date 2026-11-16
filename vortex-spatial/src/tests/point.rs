// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop for the `vortex.st.point` extension type (`geoarrow.point`).

use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::Dimension as GeoArrowDimension;
use geoarrow::datatypes::PointType;
use vortex_arrow::ArrowSessionExt;

use super::SESSION;

/// A field with interleaved (`FixedSizeList`) coordinates fails to import.
#[test]
fn import_interleaved_field_fails() {
    let point_type = PointType::new(GeoArrowDimension::XY, Default::default())
        .with_coord_type(CoordType::Interleaved);
    let field = point_type.to_field("loc", false);
    assert!(SESSION.arrow().from_arrow_field(&field).is_err());
}
