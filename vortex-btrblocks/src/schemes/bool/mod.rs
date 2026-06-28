// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bool compression schemes.

mod runend;

pub use runend::BoolRunEndScheme;
pub use vortex_compressor::builtins::BoolConstantScheme;
pub use vortex_compressor::stats::BoolStats;
