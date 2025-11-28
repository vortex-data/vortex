// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;

use crate::expr::functions::scalar::ScalarFn;

#[derive(Default, Clone, Debug)]
pub struct FunctionSession {
    registry: Registry<ScalarFn>,
}

impl FunctionSession {
    pub fn registry(&self) -> &Registry<ScalarFn> {
        &self.registry
    }
}

pub trait ScalarFuncSessionExt: SessionExt {
    fn functions(&self) -> Ref<'_, FunctionSession> {
        self.get::<FunctionSession>()
    }
}
impl<S: SessionExt> ScalarFuncSessionExt for S {}
