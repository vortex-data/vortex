// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A minimal Vortex WASM decoder kernel: it decodes its single child input and returns it
//! unchanged. It demonstrates the smallest complete kernel — parsing nothing of its own, sourcing
//! all data from the host via `decode_child`, and returning a `CanonicalMessage`.
//!
//! Real encodings parse their own metadata/buffers from the serialized array header
//! ([`vortex_wasm_guest::header::ArrayHeader`]) and transform the child data before returning it.

use vortex_error::VortexResult;
use vortex_wasm_guest::WasmEncoding;
use vortex_wasm_guest::export_wasm_encoding;
use vortex_wasm_guest::host;

struct Identity;

impl WasmEncoding for Identity {
    fn decode(_input: &[u8]) -> VortexResult<Vec<u8>> {
        // Ask the host to decode child 0 and return its CanonicalMessage verbatim.
        host::decode_child(0)
    }
}

export_wasm_encoding!(Identity);
