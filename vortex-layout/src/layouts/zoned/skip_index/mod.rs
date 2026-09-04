// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Skipping-index interface and its implementations.
//!
//! This module also provides the session extension used to register skip index
//! implementations and extends [`ZonedLayoutOptions`] with support for adding
//! a skip index.
//!
//! # Difference from a locating index
//!
//! Unlike a locating index, a skip index summarizes a zone. It does not locate
//! matching rows. It can only prove that a zone cannot match a predicate.

use std::sync::Arc;

use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::dtype::DType;
use vortex_session::SessionExt;
use vortex_session::VortexSession;

use super::writer::ZonedLayoutOptions;

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
/// [`ZonedLayoutOptions::with_skip_index`] to add a configured skip index to the
/// zoned layout options. Pass the resulting options to
/// `WriteStrategyBuilder::with_field_zoned_options` for the field to be indexed.
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
///     session.register_skip_index::<BloomSkipIndex>();
/// }
/// ```
///
/// For writes, create a configured instance and add it to the zoned layout
/// options for the field:
///
/// ```
/// use vortex_layout::layouts::zoned::skip_index::bloom::BloomSkipIndex;
/// use vortex_layout::layouts::zoned::writer::ZonedLayoutOptions;
///
/// fn zoned_options() -> ZonedLayoutOptions {
///     ZonedLayoutOptions::default().with_skip_index(BloomSkipIndex::default().into())
/// }
/// ```
///
/// Then use `WriteStrategyBuilder::with_field_zoned_options` to apply the
/// options to the field you want to index.
pub trait SkipIndex: Send + Sync + 'static {
    /// The aggregate state to persist for `input_dtype`, or `None` when unsupported.
    fn aggregate_fn(&self, input_dtype: &DType) -> Option<AggregateFnRef>;

    /// Registers the session components for this skip-index type.
    fn register(session: &VortexSession)
    where
        Self: Sized;
}

/// Extension trait for registering skipping indexes with a Vortex session.
pub trait SkipIndexSessionExt: SessionExt {
    /// Registers the session components for the skip index implementation `T`.
    ///
    /// Register each implementation only once per session. Clones of a session
    /// share registrations and do not need to be registered again.
    ///
    /// For more information about skip indexes, see [`SkipIndex`].
    fn register_skip_index<T: SkipIndex>(&self) {
        T::register(&self.session());
    }
}

impl<S: SessionExt> SkipIndexSessionExt for S {}

/// A reference-counted, configured [`SkipIndex`] used by the writer.
///
/// Registration is type-based through
/// [`SkipIndexSessionExt::register_skip_index`]. See [`SkipIndex`] for usage
/// details and examples.
#[derive(Clone)]
pub struct SkipIndexRef(Arc<dyn SkipIndex>);

impl SkipIndexRef {
    pub fn new(index_ref: Arc<dyn SkipIndex>) -> Self {
        SkipIndexRef(index_ref)
    }

    pub fn aggregate_fn(&self, input_dtype: &DType) -> Option<AggregateFnRef> {
        self.0.aggregate_fn(input_dtype)
    }
}

impl<T> From<T> for SkipIndexRef
where
    T: SkipIndex,
{
    fn from(skip_index: T) -> Self {
        SkipIndexRef(Arc::new(skip_index))
    }
}

impl ZonedLayoutOptions {
    /// Add `skip_index` to this zoned writer while retaining the default aggregates.
    ///
    /// `WriteStrategyBuilder::with_field_zoned_options` can install the configured options for one
    /// field while retaining the default data layout pipeline.
    pub fn with_skip_index(mut self, skip_index: SkipIndexRef) -> Self {
        let mut skip_indexes = self
            .skip_indexes
            .take()
            .map(|indexes| indexes.to_vec())
            .unwrap_or_default();

        skip_indexes.push(skip_index);
        self.skip_indexes = Some(skip_indexes.into());

        self
    }
}
