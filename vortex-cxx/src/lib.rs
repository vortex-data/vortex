// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod read;
mod write;

#[cxx::bridge(namespace = "vortex::ffi")]
mod ffi {}

// Workaround to conditionally generate bindings of the test function *and* compile the test function: https://github.com/dtolnay/cxx/issues/1325
// This is done with CMakeLists.txt together.
#[cfg(feature = "gen_test_data")]
mod gen_test_data;
