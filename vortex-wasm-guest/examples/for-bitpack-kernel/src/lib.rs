// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Frame-of-Reference + bit-packing decoder kernel for `i32`.
//!
//! The whole encoded form fits in the opaque payload, so this kernel has **no child** — it reads
//! everything from `vx_decode`'s input. Input layout:
//! `[i32 reference][u8 bit_width][u32 len][packed bytes…]`, where the packed bytes are the LSB-first
//! bit-packed deltas (`ceil(len * bit_width / 8)` bytes). Output: `reference + delta[i]` as an
//! `i32` array.

use vortex_wasm_guest::GuestResult;
use vortex_wasm_guest::WasmEncoding;
use vortex_wasm_guest::abi::PType;
use vortex_wasm_guest::arrow::Decoded;
use vortex_wasm_guest::bitpack;
use vortex_wasm_guest::export_wasm_encoding;
use vortex_wasm_guest::guest_ensure;

struct ForBitpack;

impl WasmEncoding for ForBitpack {
    fn decode(input: &[u8]) -> GuestResult<Decoded> {
        guest_ensure!(input.len() >= 9, "FoR-bitpack header must be 9 bytes");
        let reference = i32::from_le_bytes([input[0], input[1], input[2], input[3]]);
        let bit_width = input[4];
        let len = u32::from_le_bytes([input[5], input[6], input[7], input[8]]) as usize;

        let packed = &input[9..];
        guest_ensure!(
            packed.len() >= bitpack::packed_len(len, bit_width),
            "FoR-bitpack payload is shorter than the packed deltas"
        );

        let deltas = bitpack::unpack(packed, len, bit_width);
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
