// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::vtable::EncodeVTable;
use vortex_error::VortexResult;

use super::RLEVTable;
use crate::RLEArray;

impl EncodeVTable<RLEVTable> for RLEVTable {
    fn encode(
        _vtable: &RLEVTable,
        canonical: &Canonical,
        _like: Option<&RLEArray>,
    ) -> VortexResult<Option<RLEArray>> {
        let array = canonical.clone().into_primitive();
        Ok(Some(RLEArray::encode(&array)?))
    }
}
