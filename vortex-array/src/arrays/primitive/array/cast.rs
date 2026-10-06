// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use super::PrimitiveData;
use crate::dtype::NativePType;

impl PrimitiveData {
    /// Return a slice of the array's buffer.
    ///
    /// NOTE: these values may be nonsense if the validity buffer indicates that the value is null.
    ///
    /// # Panic
    ///
    /// This operation will panic if the array is not backed by host memory.
    pub fn as_slice<T: NativePType>(&self) -> &[T] {
        if T::PTYPE != self.ptype() {
            vortex_panic!(
                "Attempted to get slice of type {} from array of type {}",
                T::PTYPE,
                self.ptype()
            )
        }

        let byte_buffer = self
            .buffer
            .as_host_opt()
            .vortex_expect("as_slice must be called on host buffer");
        let raw_slice = byte_buffer.as_ptr();

        // SAFETY: alignment of Buffer is checked on construction
        unsafe { std::slice::from_raw_parts(raw_slice.cast(), byte_buffer.len() / size_of::<T>()) }
    }
}
