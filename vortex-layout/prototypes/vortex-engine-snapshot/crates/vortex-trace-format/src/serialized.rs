//! Serialized mirrors of engine runtime types.
//!
//! These types are decoupled from the engine's runtime types so the
//! on-disk schema can evolve independently. The recorder converts
//! engine values into these; the replayer reads them back.

use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// A row-domain span: an inclusive lower bound and exclusive upper
/// bound on a row index. Mirrors the engine's `DomainSpan`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializedDomainSpan {
    pub start: u64,
    pub end: u64,
}

/// A single interval of a `RequirementSet`: rows in `[start, end)`
/// with a per-row demand value (interpreted by the engine; opaque
/// to the format).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializedRequirementSpan {
    pub start: u64,
    pub end: u64,
    pub demand: u32,
}

/// Interval-encoded `RequirementSet`. Spans are sorted ascending by
/// `start` and do not overlap.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SerializedRequirementSet {
    pub spans: Vec<SerializedRequirementSpan>,
}

impl SerializedRequirementSet {
    pub fn new(spans: Vec<SerializedRequirementSpan>) -> Self {
        Self { spans }
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    pub fn total_rows(&self) -> u64 {
        self.spans.iter().map(|s| s.end - s.start).sum()
    }
}
