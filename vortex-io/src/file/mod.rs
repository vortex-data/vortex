// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "object_store")]
pub mod object_store;
#[cfg(not(target_arch = "wasm32"))]
pub mod std_file;
