// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Skipping-index interface and its implementations.
//!
//! This module also provides the session extension used to register skip index
//! implementations. Pass the bound aggregate returned by [`SkipIndex::aggregate_fn`]
//! to `WriteStrategyBuilder::with_field_aggregates` to index a field.
//!
//! # Difference from a locating index
//!
//! Unlike a locating index, a skip index summarizes a zone. It does not locate
//! matching rows. It can only prove that a zone cannot match a predicate.

use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTable;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::session::ScalarFnSessionExt;
use vortex_array::stats::StatsSessionExt;
use vortex_array::stats::rewrite::StatsRewriteRuleRef;
use vortex_session::SessionExt;

pub mod bloom;

/// A configured skip index used when writing zoned layouts.
///
/// Skip indexes rely on an aggregate function to summarize each zone. Vortex
/// builds these summaries during writes and uses them during reads to prove
/// that a zone cannot match a predicate and prune it.
///
/// # Usage
///
/// First, register the components needed to use the index through
/// [`SkipIndexSessionExt::register_skip_index`]. When writing, use
/// [`SkipIndex::aggregate_fn`] to bind the index options and pass the aggregate to
/// `WriteStrategyBuilder::with_field_aggregates` for the field to be indexed.
/// An explicit aggregate list replaces the writer's default aggregates. Zone length
/// is controlled by `WriteStrategyBuilder::with_row_block_size`.
///
/// # Logical and physical representation
///
/// [`SkipIndex`] defines the logical skip index abstraction. A skip index has no identifier
/// or serialized representation of its own. Instead, it is represented in a Vortex file by
/// the serialized [`AggregateFnRef`] it provides.
///
/// # Examples
///
/// Register the skip index implementation with the session before reading or
/// writing:
///
/// ```
/// use vortex_layout::layouts::zoned::skip_index::SkipIndexSessionExt;
/// use vortex_layout::layouts::zoned::skip_index::bloom::BloomSkipIndex;
/// use vortex_session::VortexSession;
///
/// fn register_index(session: &VortexSession) {
///     let index = BloomSkipIndex::default();
///     session.register_skip_index(&index);
/// }
/// ```
///
/// For writes, create a configured index and use [`SkipIndex::aggregate_fn`] to
/// obtain its aggregate. Pass the field path and an aggregate list containing it to
/// `WriteStrategyBuilder::with_field_aggregates`. This list replaces the default
/// aggregates for that field.
pub trait SkipIndex: Send + Sync + 'static {
    /// The concrete aggregate implementation registered for this index.
    type Aggregate: AggregateFnVTable;
    /// The concrete scalar implementation registered for this index.
    type Scalar: ScalarFnVTable;

    /// Returns the aggregate implementation, without binding write options.
    fn aggregate_vtable(&self) -> Self::Aggregate;

    /// Returns the scalar implementation.
    fn scalar_vtable(&self) -> Self::Scalar;

    /// Returns the options to bind when writing this index.
    fn options(&self) -> <Self::Aggregate as AggregateFnVTable>::Options;

    /// Returns the rules that turn query predicates into proofs over the summary.
    fn rewrite_rules(&self) -> Vec<StatsRewriteRuleRef>;

    /// Binds this index's write options into an aggregate for the zoned writer.
    ///
    /// Binding does not register components or check input dtype compatibility. The zoned writer
    /// omits aggregates that do not support its input dtype.
    fn aggregate_fn(&self) -> AggregateFnRef {
        self.aggregate_vtable().bind(self.options())
    }
}

/// Extension trait for registering skipping indexes with a Vortex session.
pub trait SkipIndexSessionExt: SessionExt {
    /// Registers the aggregate, probe scalar functions, and rewrite rules
    /// supplied by an skip index.
    ///
    /// If the aggregate ID is already registered, this
    /// method skips all components to avoid appending duplicate rewrite rules.
    ///
    /// For more information about skip indexes, see [`SkipIndex`].
    fn register_skip_index<I: SkipIndex>(&self, index: &I) {
        let session = self.session();
        let aggregate = index.aggregate_vtable();

        // The idea is to avoid duplicating rewrite rules.
        // Since neither they nor the skip index carry an ID,
        // this uses the aggregate ID instead.
        if session
            .aggregate_fns()
            .find_plugin(&aggregate.id())
            .is_some()
        {
            return;
        }

        session.aggregate_fns().register(aggregate);
        session.scalar_fns().register(index.scalar_vtable());

        for rule in index.rewrite_rules() {
            session.stats().register_rewrite_ref(rule);
        }
    }
}

impl<S: SessionExt> SkipIndexSessionExt for S {}
