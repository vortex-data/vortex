// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Small byte-table take with a fallback for unsupported targets and inputs.

cfg_if::cfg_if! {
    if #[cfg(all(target_arch = "aarch64", target_endian = "little"))] {
        mod neon;
        pub(super) use neon::try_take;
    } else if #[cfg(any(target_arch = "x86_64", target_arch = "x86"))] {
        mod avx2;
        pub(super) use avx2::try_take;
    } else {
        mod fallback {
            use vortex_buffer::Buffer;
            use vortex_buffer::BufferAllocatorRef;

            use super::super::FixedWidthTakeValue;
            use crate::dtype::UnsignedPType;

            pub(crate) fn try_take<T: FixedWidthTakeValue, I: UnsignedPType>(
                _values: &[T],
                _indices: &[I],
                _allocator: &BufferAllocatorRef,
            ) -> Option<Buffer<T>> {
                None
            }
        }
        pub(super) use fallback::try_take;
    }
}
