// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![deny(missing_docs)]

//! Lane-level compute kernels over Vortex buffers.
//!
//! Today this crate hosts the [`lane_kernels`] module — an [`IndexedSource`]/[`IndexedSink`]
//! abstraction plus mask-aware map kernels that the autovectorizer can drive through
//! independent lane reads/writes. Additional kernels will land here.
//!
//! [`IndexedSource`]: lane_kernels::IndexedSource
//! [`IndexedSink`]: lane_kernels::IndexedSink

pub mod lane_kernels;
