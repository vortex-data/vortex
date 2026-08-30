// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression: configuration, dictionary training, and string parsing.
//!
//! [`config`] is the public training configuration. [`trainer`] discovers a
//! dictionary from a sample; [`parser`] drives the [`lpm`] longest-prefix
//! matcher over the input to produce a column. [`hash`] is the shared hasher.

pub(crate) mod config;
pub(crate) mod hash;
pub(crate) mod lpm;
pub(crate) mod parser;
pub(crate) mod trainer;
