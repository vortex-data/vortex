// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod charts;
pub mod commit_id;
pub mod commit_info;
pub mod entry;
pub mod read_s3;

// `update_s3` uses `tokio` and `std::process::Command`, which are not available in WASM.
#[cfg(feature = "native")]
pub mod update_s3;

/// S3 key for the benchmark data Vortex file.
pub const DATA_KEY: &str = "data.vortex";

/// S3 key for the commits metadata Vortex file.
pub const COMMITS_KEY: &str = "commits.vortex";

#[cfg(target_arch = "wasm32")]
mod wasm_bindings {
    use std::sync::LazyLock;

    use vortex::VortexSessionDefault;
    use vortex::io::runtime::wasm::WasmRuntime;
    use vortex::io::session::RuntimeSessionExt;
    use vortex::session::VortexSession;
    use wasm_bindgen::prelude::*;

    use super::COMMITS_KEY;
    use super::DATA_KEY;
    use super::read_s3::get_benchmark_summary;
    use super::read_s3::get_chart_data;

    /// Cached Vortex session configured with the WASM runtime.
    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::default().with_handle(WasmRuntime::handle()));

    /// Load benchmark summary (metadata only, fast).
    ///
    /// This function fetches data from S3 (cached after first call), processes it, and returns
    /// a summary containing:
    /// - `commits`: Array of commit objects
    /// - `groups`: Object mapping group names to chart metadata (no values)
    ///
    /// Use this for fast initial load, then call `load_chart_data` for specific charts.
    ///
    /// # Returns
    ///
    /// A JSON string that must be parsed with `JSON.parse()` in JavaScript.
    #[wasm_bindgen]
    pub async fn load_benchmark_summary() -> Result<String, JsValue> {
        get_benchmark_summary(&SESSION, COMMITS_KEY, DATA_KEY)
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to load benchmark summary: {}", e)))
    }

    /// Load chart data for a specific group and chart.
    ///
    /// This function returns the aligned series data for a single chart. Data is cached after
    /// the first call to any load function, so subsequent calls are fast.
    ///
    /// # Arguments
    ///
    /// * `group` - The group name (e.g., "random-access", "tpch")
    /// * `chart` - The chart name within the group (e.g., "latency", "q1-sf1000-nvme")
    ///
    /// # Returns
    ///
    /// A JSON string containing `{ aligned_series: { series_name: [values...] } }`.
    /// Values are in nanoseconds (u64). Parse with `JSON.parse()` in JavaScript.
    #[wasm_bindgen]
    pub async fn load_chart_data(group: &str, chart: &str) -> Result<String, JsValue> {
        get_chart_data(&SESSION, COMMITS_KEY, DATA_KEY, group, chart)
            .await
            .map_err(|e| JsValue::from_str(&format!("Failed to load chart data: {}", e)))
    }
}
