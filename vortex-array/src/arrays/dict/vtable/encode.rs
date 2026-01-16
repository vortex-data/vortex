// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::DictVTable;
use crate::Canonical;
use crate::arrays::dict::DictArray;
use crate::builders::dict::dict_encode;
use crate::vtable::EncodeVTable;

impl EncodeVTable<DictVTable> for DictVTable {
    fn encode(canonical: &Canonical, like: Option<&V::Array>) -> VortexResult<Option<V::Array>> {
        Ok(Some(dict_encode(canonical.as_ref())?))
    }
}
