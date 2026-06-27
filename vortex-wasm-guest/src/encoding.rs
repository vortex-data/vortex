// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The [`WasmEncoding`] trait an encoding author implements, and the glue that exports it.

use crate::arrow::Decoded;
use crate::arrow::write_primitive;
use crate::error::GuestResult;

/// A decoder for a single WASM-embedded Vortex encoding.
///
/// Implementors read the encoding-specific input, fetch any child arrays via
/// [`host::decode_child`](crate::host::decode_child), and return the decoded result as a
/// [`Decoded`] primitive. The SDK lays it out as Arrow C Data Interface structs for the host.
///
/// Wire it up with [`export_wasm_encoding!`](crate::export_wasm_encoding).
pub trait WasmEncoding {
    /// Decode `input` (the encoding-specific bytes the host passes to `vx_decode`).
    fn decode(input: &[u8]) -> GuestResult<Decoded>;
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
        Ok(decoded) => write_primitive(&decoded),
        Err(_) => -1,
    }
}

/// Export a [`WasmEncoding`] as a complete kernel: defines the `vx_alloc` and `vx_decode` exports
/// expected by the host ABI. `vx_decode` returns a pointer to the `(array_ptr, schema_ptr)` pair of
/// the result's Arrow C Data Interface structs.
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
