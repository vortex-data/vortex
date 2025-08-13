// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A scan driver designed for engines that use a Tokio runtime for orchestrating work, including
//! CPU-bound tasks. Let's not beat around the bush, this is targeted at DataFusion :)

use crate::state::{Scan2, ScanTask, TaskSpawner};
use tokio::runtime::Handle;
use vortex_array::stream::{ArrayStream, ArrayStreamAdapter};

impl Scan2 {
    pub fn into_tokio_steam(self, handle: Handle) -> impl ArrayStream {
        let spawner: Box<dyn TaskSpawner> = Box::new(handle);
        let dtype = self.ctx.dtype.clone();
        ArrayStreamAdapter::new(dtype, self.into_scheduler(spawner))
    }
}

impl TaskSpawner for Handle {
    fn spawn_task(&self, task: Box<dyn ScanTask>) {
        // NOTE(ngates): we make an explicit choice not to spawn_blocking here as this is the
        //  compute model for DataFusion.
        let _ = Handle::spawn(self, async move { task.execute() });
    }
}
