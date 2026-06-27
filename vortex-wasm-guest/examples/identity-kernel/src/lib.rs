// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A minimal Vortex WASM decoder kernel: it returns child 0 unchanged. The smallest complete
//! kernel — it sources its data from the host via `decode_child` and re-emits it as Arrow C Data
//! Interface structs.

use vortex_wasm_guest::GuestResult;
use vortex_wasm_guest::WasmEncoding;
use vortex_wasm_guest::arrow::Decoded;
use vortex_wasm_guest::export_wasm_encoding;
use vortex_wasm_guest::host;

struct Identity;

impl WasmEncoding for Identity {
    fn decode(_input: &[u8]) -> GuestResult<Decoded> {
        let child = host::decode_child(0)?;
        Ok(Decoded {
            ptype: child.ptype,
            len: child.len,
            values: child.values.to_vec(),
            validity: child.validity.map(<[u8]>::to_vec),
        })
    }
}

export_wasm_encoding!(Identity);
