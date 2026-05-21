// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scalar functions for lazy TurboQuant vector encode and decode operations.

mod decode;
mod encode;
mod metadata;

pub use decode::TQDecode;
pub use encode::TQEncode;
