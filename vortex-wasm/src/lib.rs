// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! WASM bindings for the Vortex benchmark website.
//!
//! This module provides a `load_random_access_data()` function that fetches benchmark data from S3,
//! parses it, and returns it in a format ready for JavaScript to render.

pub mod website;

#[cfg(target_arch = "wasm32")]
mod wasm_bindings {
    use serde::Serialize;
    use vortex::VortexSessionDefault;
    use vortex::io::runtime::wasm::WasmRuntime;
    use vortex::io::session::RuntimeSessionExt;
    use vortex::session::VortexSession;
    use wasm_bindgen::prelude::*;

    use crate::website::read_s3::get_benchmark_data;
    use crate::website::read_s3::read_benchmark_entries;

    const DATA_KEY: &str = "data.vortex";
    const COMMITS_KEY: &str = "commits.vortex";

    // Legacy key for old random_access data format.
    const LEGACY_KEY: &str = "random_access.vortex";

    /// Helper macro for logging to browser console.
    macro_rules! log {
        ($($t:tt)*) => {
            web_sys::console::log_1(&format!($($t)*).into());
        }
    }

    /// A single random-access benchmark entry for JavaScript.
    #[derive(Serialize)]
    pub struct JsEntry {
        pub commit_id: String,
        pub series_name: String,
        pub value_ms: f64,
    }

    /// Load random-access benchmark data from S3.
    ///
    /// This function fetches the Vortex file from S3, parses it, and returns an array of benchmark
    /// entries ready for rendering.
    ///
    /// # Returns
    ///
    /// A JavaScript array of objects with:
    /// - `commit_id`: 40-character hex string (SHA-1 hash)
    /// - `series_name`: One of "vortex-nvme", "parquet-nvme", "lance-nvme"
    /// - `value_ms`: Value in milliseconds
    #[wasm_bindgen]
    pub async fn load_random_access_data() -> Result<JsValue, JsValue> {
        log!("Loading random-access benchmark data...");

        // Create a session configured with the WASM runtime.
        let session = VortexSession::default().with_handle(WasmRuntime::handle());

        let entries = read_benchmark_entries(&session, LEGACY_KEY)
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to read benchmark entries: {}", e)))?;

        log!("Loaded {} entries", entries.len());

        // Convert to JS-friendly format.
        let js_entries: Vec<JsEntry> = entries
            .iter()
            .map(|e| JsEntry {
                commit_id: e.commit_id.to_string(),
                series_name: e.series_name.clone(),
                value_ms: e.value as f64 / 1_000_000.0,
            })
            .collect();

        log!("Returning {} JS entries", js_entries.len());

        serde_wasm_bindgen::to_value(&js_entries)
            .map_err(|e| JsValue::from_str(&format!("Failed to serialize: {}", e)))
    }

    /// Load all benchmark data from S3.
    ///
    /// This function fetches the commits and benchmark data Vortex files from S3, parses them,
    /// aligns the data to commits, and returns a structured response.
    ///
    /// # Returns
    ///
    /// A JavaScript object with:
    /// - `benchmarks`: Nested object with group_name → charts → chart_name → aligned_series → series_name → values
    /// - `commits`: Array of commit objects with timestamp, author, message, and commit_id
    ///
    /// Values are in nanoseconds (u64). Convert to milliseconds in JavaScript by dividing by
    /// 1_000_000.
    #[wasm_bindgen]
    pub async fn load_benchmark_data() -> Result<JsValue, JsValue> {
        log!("Loading benchmark data...");

        let session = VortexSession::default().with_handle(WasmRuntime::handle());

        let result = get_benchmark_data(&session, COMMITS_KEY, DATA_KEY)
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to load benchmark data: {}", e)))?;

        log!("Benchmark data loaded successfully");

        Ok(result)
    }

    /// Initialize the WASM module.
    #[wasm_bindgen(start)]
    pub fn init() {
        console_error_panic_hook::set_once();
        log!("vortex-wasm initialized");
    }

    /// Get version information.
    #[wasm_bindgen]
    pub fn get_version() -> String {
        format!("vortex-wasm v{}", env!("CARGO_PKG_VERSION"))
    }
}
