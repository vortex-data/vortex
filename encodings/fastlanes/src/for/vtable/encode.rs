// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::vtable::EncodeVTable;
use vortex_error::VortexResult;

use super::FoRVTable;
use crate::FoRArray;

impl EncodeVTable<FoRVTable> for FoRVTable {
    fn encode(canonical: &Canonical, like: Option<&V::Array>) -> VortexResult<Option<V::Array>> {
        let parray = canonical.clone().into_primitive();
        Ok(Some(FoRArray::encode(parray)?))
    }
}
