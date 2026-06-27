// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Frame-of-Reference + bit-packing decoder kernel for `i32`.
//!
//! This composes two transforms in one kernel and shows real on-disk size reduction: the encoder
//! stores `value - reference` in the *minimum* number of bits, and this kernel reconstructs the
//! original values.
//!
//! Wire layout:
//! - payload: `[i32 reference][u8 bit_width][u32 len]` (9 bytes).
//! - child 0: a `u8` array holding the LSB-first bit-packed deltas (the on-disk savings).
//!
//! The kernel reads the payload, decodes the packed child via [`host::decode_child`], unpacks the
//! deltas with [`bitpack::unpack`], and emits `reference + delta[i]`.

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_wasm_guest::WasmEncoding;
use vortex_wasm_guest::abi::MessageKind;
use vortex_wasm_guest::abi::PType;
use vortex_wasm_guest::bitpack;
use vortex_wasm_guest::export_wasm_encoding;
use vortex_wasm_guest::host;
use vortex_wasm_guest::message::MessageReader;
use vortex_wasm_guest::message::primitive_message;

struct ForBitpack;

impl WasmEncoding for ForBitpack {
    fn decode(input: &[u8]) -> VortexResult<Vec<u8>> {
        vortex_ensure!(input.len() >= 9, "FoR-bitpack payload must be 9 bytes");
        let reference = i32::from_le_bytes(input[0..4].try_into().expect("4 bytes"));
        let bit_width = input[4];
        let len = u32::from_le_bytes(input[5..9].try_into().expect("4 bytes")) as usize;

        // The packed deltas are delivered as a u8 child.
        let child = host::decode_child(0)?;
        let reader = MessageReader::new(&child)?;
        reader.expect_kind(MessageKind::Primitive)?;
        if reader.ptype() != PType::U8 as u8 {
            vortex_bail!("FoR-bitpack expects a u8 packed-delta child");
        }
        let packed = reader.first_buffer()?;

        let deltas = bitpack::unpack(packed, len, bit_width);
        let mut out = Vec::with_capacity(len * 4);
        for delta in deltas {
            out.extend_from_slice(&reference.wrapping_add(delta as i32).to_le_bytes());
        }
        Ok(primitive_message(PType::I32, len, &out))
    }
}

export_wasm_encoding!(ForBitpack);
