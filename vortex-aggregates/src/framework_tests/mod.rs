// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Tests of the `vortex_array::aggregate_fn` framework that need concrete aggregate functions.

mod accumulator;
mod chunked_kernel;
mod core_consistency;
mod listview_is_constant;
mod proto;
mod stats;
mod storage_pushdown;
mod struct_is_constant;
mod union_uncompressed_size;
