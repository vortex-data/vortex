// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests of aggregates over integer and decimal `Widen` arrays.

mod aggregates;
mod decimal_aggregates;

use std::sync::LazyLock;

use vortex_array::IntoArray;
use vortex_array::arrays::WidenArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::PType;
use vortex_error::VortexExpect;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

/// Returns a `Widen` of the `logical` type over `values`, with the nullability of `values`.
fn widen(values: impl IntoArray, logical: PType) -> WidenArray {
    let values = values.into_array();
    let dtype = DType::Primitive(logical, values.dtype().nullability());
    WidenArray::try_new(values, dtype).vortex_expect("fixture values are narrower integers")
}
