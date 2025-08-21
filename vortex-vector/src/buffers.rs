// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Deref;
use std::sync::atomic::AtomicUsize;

static NEXT_ID: AtomicUsize = AtomicUsize::new(0);

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct BufferId(usize);

impl Default for BufferId {
    fn default() -> Self {
        Self::new()
    }
}

impl BufferId {
    /// Creates a new `BufferId` with a unique identifier.
    pub fn new() -> Self {
        BufferId(NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

impl Deref for BufferId {
    type Target = usize;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}