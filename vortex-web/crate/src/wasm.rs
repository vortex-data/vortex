// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::VortexSessionDefault;
use vortex::buffer::ByteBuffer;
use vortex::file::OpenOptionsSessionExt;
use vortex::io::runtime::wasm::WasmRuntime;
use vortex::io::session::RuntimeSessionExt;
use vortex::session::VortexSession;
use wasm_bindgen::prelude::*;

/// Initialize the WASM module (sets up panic hook for better error messages).
#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
}

/// Open a Vortex file from raw bytes and return a handle for exploration.
///
/// Call this from JavaScript after reading a `.vortex` file via drag-and-drop.
#[wasm_bindgen]
pub fn open_vortex_file(data: &[u8]) -> Result<VortexFileHandle, JsValue> {
    let session = VortexSession::default().with_handle(WasmRuntime::handle());
    let buffer = ByteBuffer::from(data.to_vec());

    let vxf = session
        .open_options()
        .open_buffer(buffer)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let row_count = vxf.row_count();
    let dtype = format!("{}", vxf.dtype());

    Ok(VortexFileHandle { row_count, dtype })
}

/// A handle to an opened Vortex file, exposing metadata to JavaScript.
#[wasm_bindgen]
pub struct VortexFileHandle {
    row_count: u64,
    dtype: String,
}

#[wasm_bindgen]
impl VortexFileHandle {
    /// The total number of rows in the file.
    #[wasm_bindgen(getter)]
    pub fn row_count(&self) -> u64 {
        self.row_count
    }

    /// The top-level DType of the file as a string.
    #[wasm_bindgen(getter)]
    pub fn dtype(&self) -> String {
        self.dtype.clone()
    }
}
