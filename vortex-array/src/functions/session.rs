// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::{Ref, SessionExt};

#[derive(Debug)]
pub struct FunctionSession {}

pub trait ScalarFuncSessionExt: SessionExt {
    fn functions(&self) -> Ref<FunctionSession> {
        self.get::<FunctionSession>()
    }
}
impl<S: SessionExt> ScalarFuncSessionExt for S {}
