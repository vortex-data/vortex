// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A Frame-of-Reference (FoR) decoder kernel for `i32` — the minimal real encoding.
//!
//! Input (`vx_decode`): a 4-byte little-endian `i32` reference. Child 0: the per-element deltas
//! (an `i32` array). Output: `reference + delta[i]` as an `i32` array, returned as Arrow C Data
//! Interface structs.
//!
//! Demonstrates the dependency-free guest SDK: read a child via [`host::decode_child`], return a
//! [`Decoded`], and wire it up with [`export_wasm_encoding!`].

use vortex_wasm_guest::GuestResult;
use vortex_wasm_guest::WasmEncoding;
use vortex_wasm_guest::abi::PType;
use vortex_wasm_guest::arrow::Decoded;
use vortex_wasm_guest::export_wasm_encoding;
use vortex_wasm_guest::guest_ensure;
use vortex_wasm_guest::host;

struct FrameOfReference;

fn read_i32(bytes: &[u8], i: usize) -> i32 {
    let o = i * 4;
    i32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]])
}

impl WasmEncoding for FrameOfReference {
    fn decode(input: &[u8]) -> GuestResult<Decoded> {
        guest_ensure!(
            input.len() >= 4,
            "FoR payload must contain an i32 reference"
        );
        let reference = read_i32(input, 0);

        let child = host::decode_child(0)?;
        guest_ensure!(child.ptype == PType::I32, "FoR expects i32 deltas");

        let mut values = Vec::with_capacity(child.len * 4);
        for i in 0..child.len {
            let delta = read_i32(child.values, i);
            values.extend_from_slice(&reference.wrapping_add(delta).to_le_bytes());
        }

        Ok(Decoded {
            ptype: PType::I32,
            len: child.len,
            values,
            validity: None,
        })
    }
}

export_wasm_encoding!(FrameOfReference);
