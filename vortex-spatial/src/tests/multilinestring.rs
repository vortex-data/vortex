// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Arrow interop for the `vortex.st.multilinestring` extension type (`geoarrow.multilinestring`).

use geoarrow::datatypes::CoordType;
use geoarrow::datatypes::Dimension as GeoArrowDimension;
use geoarrow::datatypes::MultiLineStringType;
use vortex_arrow::ArrowSessionExt;

use super::SESSION;

/// A field with interleaved (`FixedSizeList`) coordinates fails to import.
#[test]
fn import_interleaved_field_fails() {
    let multilinestring_type = MultiLineStringType::new(GeoArrowDimension::XY, Default::default())
        .with_coord_type(CoordType::Interleaved);
    let field = multilinestring_type.to_field("geom", false);
    assert!(SESSION.arrow().from_arrow_field(&field).is_err());
}
