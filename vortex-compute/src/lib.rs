// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A collection of compute functions primarily for operating over Vortex vectors.

#![deny(missing_docs)]
#![deny(clippy::missing_panics_doc)]
#![deny(clippy::missing_safety_doc)]

pub mod arithmetic;
#[cfg(feature = "arrow")]
pub mod arrow;
pub mod comparison;
pub mod filter;
pub mod logical;
pub mod mask;
