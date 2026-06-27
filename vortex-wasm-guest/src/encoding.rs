// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`WasmEncoding`] trait an encoding author implements, and the glue that exports it.

use vortex_error::VortexResult;

/// A decoder for a single WASM-embedded Vortex encoding.
///
/// Implementors parse the serialized array header (via [`ArrayHeader`](crate::header::ArrayHeader)),
/// fetch any decoded child arrays through [`host::decode_child`](crate::host::decode_child), and
/// return the decoded output as a [`CanonicalMessage`](crate::message) byte blob (typically built
/// with [`MessageBuilder`](crate::message::MessageBuilder)).
///
/// Wire it up with [`export_wasm_encoding!`](crate::export_wasm_encoding).
pub trait WasmEncoding {
    /// Decode the serialized array `input` into a `CanonicalMessage` blob.
    fn decode(input: &[u8]) -> VortexResult<Vec<u8>>;
}

/// Internal entry point invoked by [`export_wasm_encoding!`]. Not part of the stable API.
#[doc(hidden)]
pub fn __run_decode<E: WasmEncoding>(in_ptr: i32, in_len: i32) -> i32 {
    let input: &[u8] = if in_len <= 0 {
        &[]
    } else {
        unsafe { core::slice::from_raw_parts(in_ptr as *const u8, in_len as usize) }
    };
    match E::decode(input) {
        Ok(msg) => {
            let total = 4 + msg.len();
            let ptr = crate::host::__alloc(total);
            let len_prefix = (msg.len() as u32).to_le_bytes();
            unsafe {
                core::ptr::copy_nonoverlapping(len_prefix.as_ptr(), ptr, 4);
                core::ptr::copy_nonoverlapping(msg.as_ptr(), ptr.add(4), msg.len());
            }
            ptr as i32
        }
        Err(_) => -1,
    }
}

/// Export a [`WasmEncoding`] as a complete kernel: defines the `vx_alloc` and `vx_decode` exports
/// expected by the host ABI.
///
/// ```ignore
/// struct MyEncoding;
/// impl vortex_wasm_guest::WasmEncoding for MyEncoding { /* ... */ }
/// vortex_wasm_guest::export_wasm_encoding!(MyEncoding);
/// ```
#[macro_export]
macro_rules! export_wasm_encoding {
    ($ty:ty) => {
        /// Guest allocator export required by the host ABI.
        #[unsafe(no_mangle)]
        pub extern "C" fn vx_alloc(len: i32) -> i32 {
            $crate::host::__alloc(len.max(0) as usize) as i32
        }

        /// Decode entrypoint export required by the host ABI.
        #[unsafe(no_mangle)]
        pub extern "C" fn vx_decode(in_ptr: i32, in_len: i32) -> i32 {
            $crate::__run_decode::<$ty>(in_ptr, in_len)
        }
    };
}
