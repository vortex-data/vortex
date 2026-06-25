// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generates the canonical read-boundary benchmark file shared by the Rust `read_boundary` bench
//! and the Java `VortexJniReadBenchmark` JMH benchmark.
//!
//! Run directly (`cargo run -p vortex-jni --example gen_bench_data -- [PATH]`) or via the Gradle
//! `generateBenchFile` task. With no argument it writes to [`canonical::default_path`]. Generation
//! is idempotent: an existing non-empty file is left untouched so both benches read the same bytes.

#[path = "../benches/canonical/mod.rs"]
mod canonical;

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let path: PathBuf = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(canonical::default_path);

    let existed = path.metadata().map(|m| m.len() > 0).unwrap_or(false);
    match canonical::ensure_canonical(&path) {
        Ok(()) => {
            if existed {
                println!("canonical bench file already present: {}", path.display());
            } else {
                println!(
                    "wrote canonical bench file ({} rows): {}",
                    canonical::ROWS,
                    path.display()
                );
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("failed to generate canonical bench file: {e}");
            ExitCode::FAILURE
        }
    }
}
