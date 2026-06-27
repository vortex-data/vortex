// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A Frame-of-Reference (FoR) decoder kernel for `i32` — the minimal real encoding.
//!
//! The on-disk representation is a reference value plus per-element deltas. The writer stores the
//! reference in the kernel's payload (`[i32 reference]`) and the deltas as the single child input.
//! This kernel reconstructs `reference + delta[i]` for each element.
//!
//! It demonstrates the full guest SDK surface: reading the payload, decoding a child via
//! [`host::decode_child`], reading a [`MessageReader`], and building the output with
//! [`primitive_message`].

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_wasm_guest::WasmEncoding;
use vortex_wasm_guest::abi::MessageKind;
use vortex_wasm_guest::abi::PType;
use vortex_wasm_guest::export_wasm_encoding;
use vortex_wasm_guest::host;
use vortex_wasm_guest::message::MessageReader;
use vortex_wasm_guest::message::primitive_message;

struct FrameOfReference;

impl WasmEncoding for FrameOfReference {
    fn decode(input: &[u8]) -> VortexResult<Vec<u8>> {
        // Payload: a single little-endian i32 reference value.
        vortex_ensure!(
            input.len() >= 4,
            "FoR payload must contain an i32 reference"
        );
        let reference = i32::from_le_bytes(input[0..4].try_into().expect("4 bytes"));

        // The deltas are the single child input.
        let child = host::decode_child(0)?;
        let reader = MessageReader::new(&child)?;
        reader.expect_kind(MessageKind::Primitive)?;
        if reader.ptype() != PType::I32 as u8 {
            vortex_bail!("FoR example kernel only supports i32 deltas");
        }

        let len = reader.length();
        let deltas = reader.first_buffer()?;
        vortex_ensure!(deltas.len() >= len * 4, "delta buffer shorter than length");

        let mut out = Vec::with_capacity(len * 4);
        for i in 0..len {
            let delta = i32::from_le_bytes(deltas[i * 4..i * 4 + 4].try_into().expect("4 bytes"));
            out.extend_from_slice(&reference.wrapping_add(delta).to_le_bytes());
        }

        Ok(primitive_message(PType::I32, len, &out))
    }
}

export_wasm_encoding!(FrameOfReference);
