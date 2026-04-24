// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-array-type synthetic fixtures.
//!
//! Each fixture exercises a core Vortex array type with boundary values and nullable variants.

mod bool;
mod chunked;
mod datetime;
mod decimal;
mod fixed_size_list;
mod list;
mod listview;
mod null;
mod primitive;
mod struct_nested;
mod varbin;
mod varbinview;

use crate::fixtures::FlatLayoutFixture;

/// All per-array-type fixtures.
pub fn fixtures() -> Vec<Box<dyn FlatLayoutFixture>> {
    vec![
        Box::new(primitive::PrimitivesFixture),
        Box::new(varbin::VarBinFixture),
        Box::new(varbinview::VarBinViewFixture),
        Box::new(bool::BooleansFixture),
        Box::new(struct_nested::StructNestedFixture),
        Box::new(chunked::ChunkedFixture),
        Box::new(list::ListFixture),
        Box::new(fixed_size_list::FixedSizeListFixture),
        Box::new(null::NullFixture),
        Box::new(datetime::DateTimeFixture),
        Box::new(decimal::DecimalFixture),
        Box::new(listview::ListViewFixture),
    ]
}
