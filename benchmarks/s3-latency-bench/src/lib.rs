// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Simulated S3 for benchmarks: an `object_store` wrapper with the latency of a model fitted by
//! the `s3-latency-bench` binary.

#![allow(
    clippy::cast_possible_truncation,
    reason = "sizes, counts and durations here are far below the truncation limits"
)]

pub mod simulated;
