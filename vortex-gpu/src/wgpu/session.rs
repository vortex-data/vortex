// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::vtable::ArrayId;
use vortex_utils::aliases::hash_map::HashMap;

use crate::wgpu::executor::WgpuSupport;

#[derive(Default)]
pub struct WgpuSession {
    // Registry of supported array executors.
    executors: HashMap<ArrayId, &'static dyn WgpuSupport>,
}

impl WgpuSession {
    pub fn register_executor(&mut self, array_id: ArrayId, executor: &'static dyn WgpuSupport) {
        self.executors.insert(array_id, executor);
    }

    pub fn get_executor(&self, array_id: &ArrayId) -> Option<&'static dyn WgpuSupport> {
        self.executors.get(array_id).copied()
    }
}
