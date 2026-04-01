// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression scheme implementations.

pub mod bool;
pub mod float;
pub mod integer;
pub mod string;

pub mod decimal;
pub mod temporal;

pub(crate) mod patches;
pub(crate) mod rle;
