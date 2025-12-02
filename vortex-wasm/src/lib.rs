// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! WASM bindings for the Vortex benchmark website.
//!
//! This module provides a `load_random_access_data()` function that fetches benchmark data from S3,
//! parses it, and returns it in a format ready for JavaScript to render.

pub mod website;

use serde::Serialize;
use vortex::VortexSessionDefault;
use vortex::session::VortexSession;
use wasm_bindgen::prelude::*;
use website::names::NAMES;
use website::read_s3::read_benchmark_entries;

const KEY: &str = "test/random_access.vortex";

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
    pub series_name: &'static str,
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

    let session = VortexSession::default();

    let entries = read_benchmark_entries(&session, KEY)
        .await
        .map_err(|e| JsValue::from_str(&format!("Failed to read benchmark entries: {}", e)))?;

    log!("Loaded {} entries", entries.len());

    // Convert to JS-friendly format.
    let js_entries: Vec<JsEntry> = entries
        .iter()
        .map(|e| JsEntry {
            commit_id: e.commit_id.to_string(),
            series_name: NAMES.get(&e.series_name.0).copied().unwrap_or("unknown"),
            value_ms: e.value as f64 / 1_000_000.0,
        })
        .collect();

    log!("Returning {} JS entries", js_entries.len());

    serde_wasm_bindgen::to_value(&js_entries)
        .map_err(|e| JsValue::from_str(&format!("Failed to serialize: {}", e)))
}

/// Initialize the WASM module.
#[wasm_bindgen(start)]
pub fn init() {
    log!("vortex-wasm initialized");
}

/// Get version information.
#[wasm_bindgen]
pub fn get_version() -> String {
    format!("vortex-wasm v{}", env!("CARGO_PKG_VERSION"))
}
