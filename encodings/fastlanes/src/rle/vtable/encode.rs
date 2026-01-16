// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::vtable::EncodeVTable;
use vortex_error::VortexResult;

use super::RLEVTable;
use crate::RLEArray;

impl EncodeVTable<RLEVTable> for RLEVTable {
    fn encode(canonical: &Canonical, like: Option<&V::Array>) -> VortexResult<Option<V::Array>> {
        let array = canonical.clone().into_primitive();
        Ok(Some(RLEArray::encode(&array)?))
    }
}
