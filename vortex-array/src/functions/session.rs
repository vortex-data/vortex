// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::Ref;
use vortex_session::SessionExt;

#[derive(Debug)]
pub struct FunctionSession {}

pub trait ScalarFuncSessionExt: SessionExt {
    fn functions(&self) -> Ref<'_, FunctionSession> {
        self.get::<FunctionSession>()
    }
}
impl<S: SessionExt> ScalarFuncSessionExt for S {}
