// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use cudarc::driver::{CudaStream, HostSlice, SyncOnDrop};

use crate::BufferMut;

impl<T> HostSlice<T> for BufferMut<T> {
    fn len(&self) -> usize {
        self.len()
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        _stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        (self.as_slice(), SyncOnDrop::Sync(None))
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        _stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        (self.as_mut_slice(), SyncOnDrop::Sync(None))
    }
}
