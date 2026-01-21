// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;

use crate::DynVTable;
use crate::ExtId;
use crate::VTable;

#[derive(Debug, Default)]
pub struct DTypeSession {
    registry: Registry<&'static dyn DynVTable>,
}

impl DTypeSession {
    /// Register an extension DType with the Vortex session.
    pub fn register<V: VTable>(
        &self,
        id: impl Into<ExtId>,
        vtable: impl Into<&'static dyn DynVTable>,
    ) {
        self.registry.register(id, vtable);
    }
}

pub trait DTypeSessionExt: SessionExt {
    /// Get the DType session.
    fn dtypes(&self) -> Ref<DTypeSession>;
}

impl<S: SessionExt> DTypeSessionExt for S {
    fn dtypes(&self) -> Ref<DTypeSession> {
        self.get::<DTypeSession>()
    }
}
