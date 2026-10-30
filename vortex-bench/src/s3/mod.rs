// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Opt-in, request-derived S3 simulation for benchmark clients using the S3 HTTP protocol.

mod model;
mod server;

pub use model::Profile;
pub use server::Server;

/// Benchmark storage label; the simulation launcher sets this only for its child process.
pub fn storage_name() -> &'static str {
    if std::env::var("VORTEX_BENCH_S3_SIMULATED").is_ok_and(|value| value == "1") {
        "s3-simulated"
    } else {
        "s3"
    }
}

#[cfg(test)]
mod tests;
