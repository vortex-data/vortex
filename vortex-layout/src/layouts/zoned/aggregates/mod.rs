// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregate functions selected by the zoned layout.

// TODO (joacoc) Allow dead code here until the writer interface for accessing the
// Bloom filter lands in https://github.com/vortex-data/vortex/pull/9413,
// unless another access point exists that I am unaware of.
#[allow(dead_code)]
pub mod bloom_filter;
