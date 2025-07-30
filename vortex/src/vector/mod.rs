// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! I'm calling these vectors for two reasons: first, so I don't confuse myself with what we
//! currently call arrays (we're probably on Arrays 5.0 at this point), and second, because
//! as first writing this, I'm not entirely sure if vectors are distinct from arrays. Anyway,
//! you're here for the ride now!
//!
//! Goals:
//! - Bring Vortex performance up to state-of-the-art.
//! - Support zero-copy decoding of data from disk into externally provided buffers.
//! - All without a wire break (I'm quite confident in this).
//!
//! How I plan to achieve this:
//! - Lean heavily on SIMD compute and CPU cache locality.
//!
//! Therefore, some meta-goals that fall out of this:
//! - Thread-locality and core affinity is important. Keep data within the L1 cache as much as
//!   possible. This has the additional benefit of avoiding overhead of concurrency and
//!   synchronization primitives.
//! - Data to be processed in much smaller chunks, fitting in the L1 cache, rather than now where
//!   data is largely processed in the chunks as they appear in the file.
//! - Outputs need to be passed in to the scan / compute functions in order to support externally
//!   provided buffers, such as Arrow, Numpy, etc.
//!
//! Evaluation:
//! - Our primary focus is on DuckDB performance, largely because the execution model aligns so
//!   well. If we can return DuckDB's 2k vectors efficiently, then we can hopefully keep the entire
//!   pipeline from disk through to the DuckDB result within the L1 or L2 caches.
//! - We care more about the performance of scan-heavy queries, less about join-heavy queries.
//!   We do care about the performance of highly selective queries to explore how masking interacts
//!   with pipelined compute.
//!
//! ## Pipelined Compute
//!
//! The core component if this change is to introduce a new compute model that allows for better
//! pipelining of operations over smaller chunks of data.
//!
//! In this world, an Array becomes actually _more_ like a Layout, in that it can be converted into
//! a compute pipeline (evaluation) to be executed piecemeal. An array holds onto zero-copy data
//! from disk, where the data is only accessed at the time of evaluation. A pipeline is then driven
//! with small chunks of data at a time.
//!
//! Arrays still support compute functions that take and return arrays, but internally, these are
//! implemented using pipelined evaluation. The array on which the compute function was invoked is
//! known as the "entry point" array, and it is responsible for constructing an evaluation, driving
//! it, and collecting the result. For example, a DictArray can drive separate evaluations for its
//! values and codes, and then re-assemble the results into a dictionary. Note that this dict
//! push-down will therefore only occur if the top-level entry point is a DictArray.
//!
//! So each array has one function to get a compute kernel, and one function to get a compute
//! evaluation. If either fails to return, a default canonical implementation is used, as now.
//!

#![allow(dead_code)]
#![allow(unused_variables)]

mod export;
mod expression;
mod impls;
mod pipeline;
mod vector;
