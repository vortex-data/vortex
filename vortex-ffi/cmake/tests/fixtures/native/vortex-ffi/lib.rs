// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

pub use native_helper::value;

/// Exercise both native dependencies through the C ABI.
// SAFETY: This fixture is the only definition of `vx_fixture_value`.
#[unsafe(no_mangle)]
pub extern "C" fn vx_fixture_value() -> i32 {
    value()
}
