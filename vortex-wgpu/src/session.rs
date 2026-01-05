// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::executor::WgpuSupport;
use vortex_array::vtable::ArrayId;
use vortex_utils::aliases::dash_map::DashMap;

#[derive(Default, Debug)]
pub struct WgpuSession {
    // Registry of supported array executors.
    executors: DashMap<ArrayId, &'static dyn WgpuSupport>,
}

impl WgpuSession {
    pub fn register_executor(&mut self, array_id: ArrayId, executor: &'static dyn WgpuSupport) {
        self.executors.insert(array_id, executor);
    }

    pub fn get_executor(&self, array_id: &ArrayId) -> Option<&'static dyn WgpuSupport> {
        self.executors.get(array_id).map(|s| *s.value())
    }
}
