// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use super::DictVTable;
use crate::Canonical;
use crate::arrays::DictArray;
use crate::builders::dict::dict_encode;
use crate::vtable::EncodeVTable;

impl EncodeVTable<DictVTable> for DictVTable {
    fn encode(canonical: &Canonical, _like: Option<&DictArray>) -> VortexResult<Option<DictArray>> {
        Ok(Some(dict_encode(canonical.as_ref())?))
    }
}
