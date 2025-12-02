// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub mod entry;
pub mod names;
pub mod read_s3;

// update_s3 uses tokio and std::process::Command which are not available in WASM.
#[cfg(feature = "native")]
pub mod update_s3;
