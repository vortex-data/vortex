// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::scalar_fn::ScalarFnVTable;

use crate::v2::reader::Reader;
use crate::v2::readers::scalar_fn::ScalarFnReader;

impl dyn Reader + '_ {
    /// If this reader is a [`ScalarFnReader`], return its scalar function options
    pub fn as_scalar_fn<V: ScalarFnVTable>(&self) -> Option<&V::Options> {
        self.as_any()
            .downcast_ref::<ScalarFnReader>()
            .and_then(|r| r.scalar_fn().as_opt::<V>())
    }
}
