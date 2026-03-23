// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

/// A SplitPlan captures the I/O dependency graph of a single split of the layout execution.
///
/// Each split may have zero or more I/O requests, possibly chained in some conditional way.
/// The SplitPlan captures these relationships, allowing the scheduler to have full visibility
/// into future requests and manage probabilistic pre-fetching of these segments.
pub trait SplitPlan: 'static + Send + Sync {}
