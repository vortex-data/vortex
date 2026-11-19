// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Rejection of WKB fields with storage outside the binary family.

use arrow_schema::DataType;
use arrow_schema::Field;
use geoarrow::datatypes::WkbType;
use vortex_arrow::ArrowSessionExt;

use super::SESSION;

/// A WKB field over strings must fail instead of importing as plain storage.
#[test]
fn import_string_field_fails() {
    let field = Field::new("geom", DataType::Binary, false)
        .with_extension_type(WkbType::new(Default::default()))
        .with_data_type(DataType::Utf8);
    let err = SESSION
        .arrow()
        .from_arrow_field(&field)
        .expect_err("a string field must fail to import as WKB");
    assert!(
        err.to_string().contains("as geoarrow.wkb"),
        "unexpected error: {err}"
    );
}
