// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Frame-of-Reference + bit-packing decoder kernel for `i32`.
//!
//! Input (`vx_decode`): `[i32 reference][u8 bit_width][u32 len]` (9 bytes). Child 0: a `u8` array
//! holding the LSB-first bit-packed deltas. Output: `reference + delta[i]` as an `i32` array.

use vortex_wasm_guest::GuestResult;
use vortex_wasm_guest::WasmEncoding;
use vortex_wasm_guest::abi::PType;
use vortex_wasm_guest::arrow::Decoded;
use vortex_wasm_guest::bitpack;
use vortex_wasm_guest::export_wasm_encoding;
use vortex_wasm_guest::guest_ensure;
use vortex_wasm_guest::host;

struct ForBitpack;

impl WasmEncoding for ForBitpack {
    fn decode(input: &[u8]) -> GuestResult<Decoded> {
        guest_ensure!(input.len() >= 9, "FoR-bitpack payload must be 9 bytes");
        let reference = i32::from_le_bytes([input[0], input[1], input[2], input[3]]);
        let bit_width = input[4];
        let len = u32::from_le_bytes([input[5], input[6], input[7], input[8]]) as usize;

        let child = host::decode_child(0)?;
        guest_ensure!(
            child.ptype == PType::U8,
            "FoR-bitpack expects a u8 packed child"
        );

        let deltas = bitpack::unpack(child.values, len, bit_width);
        let mut values = Vec::with_capacity(len * 4);
        for delta in deltas {
            values.extend_from_slice(&reference.wrapping_add(delta as i32).to_le_bytes());
        }

        Ok(Decoded {
            ptype: PType::I32,
            len,
            values,
            validity: None,
        })
    }
}

export_wasm_encoding!(ForBitpack);
