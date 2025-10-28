// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A collection of compute functions primarily for operating over Vortex vectors.

#![deny(missing_docs)]
#![deny(clippy::missing_panics_doc)]
#![deny(clippy::missing_safety_doc)]

#[cfg(feature = "arithmetic")]
pub mod arithmetic;
#[cfg(feature = "comparison")]
pub mod comparison;
#[cfg(feature = "filter")]
pub mod filter;
#[cfg(feature = "logical")]
pub mod logical;
