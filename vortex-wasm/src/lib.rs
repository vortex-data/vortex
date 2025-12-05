// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! WASM bindings for the Vortex benchmark website.
//!
//! This crate provides functions for fetching and processing benchmark data from S3, returning it
//! in a format ready for JavaScript to render.

pub mod website;

#[cfg(target_arch = "wasm32")]
mod wasm_init {
    use wasm_bindgen::prelude::*;

    /// Initialize the WASM module.
    #[wasm_bindgen(start)]
    pub fn init() {
        console_error_panic_hook::set_once();
    }

    /// Get version information.
    #[wasm_bindgen]
    pub fn get_version() -> String {
        format!("vortex-wasm v{}", env!("CARGO_PKG_VERSION"))
    }
}
