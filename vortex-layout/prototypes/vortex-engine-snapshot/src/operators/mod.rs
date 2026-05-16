//! General-purpose operators that aren't tied to any specific
//! source format.
//!
//! Layout-aware sources live in [`crate::layouts`]. Synthetic
//! prototype operators (used by the worked examples in
//! `physical_plan/examples.rs`) are scoped as test fixtures and
//! stay there. Everything in this module is intended to compose
//! against any operator graph the planner builds.
//!
//! Files in this module use a `_<role>.rs` suffix where the role is
//! clear: `*_sink.rs` for sinks, plus single-file modules for
//! transforms (`filter.rs`, `aggregate.rs`, ...). Type names follow
//! the same convention (`CollectI64Sink`, `Aggregate<V>`, etc.).
//!
//! Aggregates are split across three sibling files so each operator
//! is the focus of its own module: [`aggregate`] (single-lane,
//! all-in-one), [`partial_aggregate`] (multi-lane, emits
//! partial-state batches), and [`merge_aggregate`] (single-lane,
//! folds partial-state batches into a final scalar). Shared helpers
//! (lane-safety table, partial-merge dispatch, scalar-to-array)
//! live in [`aggregate_common`].

mod aggregate;
mod aggregate_common;
mod array_collect_sink;
mod collect_i64_sink;
mod concat;
mod count_distinct_i64_sink;
mod filter;
mod k_way_merge;
mod lazy_vortex_file;
mod merge_aggregate;
mod partial_aggregate;

pub use aggregate::Aggregate;
pub use aggregate_common::is_lane_safe as aggregate_fn_is_lane_safe;
pub use array_collect_sink::ArrayCollectSink;
pub use collect_i64_sink::CollectI64Sink;
pub use concat::Concat;
pub use count_distinct_i64_sink::CountDistinctI64Sink;
pub use count_distinct_i64_sink::CountDistinctState;
pub use filter::Filter;
pub use k_way_merge::KWayMerge;
pub use lazy_vortex_file::LazyVortexFile;
pub use merge_aggregate::MergeAggregate;
pub use partial_aggregate::PartialAggregate;
