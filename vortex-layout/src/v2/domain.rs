// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Row-domain identifiers shared by the V2 scheduler prototype.

#![allow(dead_code)]

use std::ops::Range;

/// Stable identifier for an execution domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct DomainId(u32);

impl DomainId {
    /// Construct a domain identifier.
    pub(crate) const fn new(id: u32) -> Self {
        Self(id)
    }
}

/// Stable identifier for a plan node or runtime operator.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct OperatorId(u32);

impl OperatorId {
    /// Construct an operator identifier.
    pub(crate) const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Return the raw identifier.
    pub(crate) const fn raw(self) -> u32 {
        self.0
    }
}

/// The identity space an operator's rows live in.
///
/// `Sorted` is modeled as a separate variant in this prototype because
/// sortedness changes what can be lowered into ordinal demand. A later
/// design may represent sortedness as a property on `Keyed`.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub(crate) enum Domain {
    /// File/partition row ordinals. This is the domain where
    /// cardinality-preserving placeholders are meaningful.
    Ordinal { id: DomainId },
    /// Rows identified by a logical key, with no ordering promise.
    Keyed { id: DomainId, key: &'static str },
    /// Rows identified by a logical key whose order can be translated
    /// back to an ordinal row domain.
    Sorted {
        id: DomainId,
        key: &'static str,
        ordinal: DomainId,
    },
}

impl Domain {
    /// Returns the ordinal domain this domain can map to exactly.
    pub(crate) fn ordinal_mapping(&self) -> Option<DomainId> {
        match self {
            Domain::Ordinal { id } => Some(*id),
            Domain::Keyed { .. } => None,
            Domain::Sorted { ordinal, .. } => Some(*ordinal),
        }
    }
}

/// A vectorized message over a contiguous domain range.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct VectorBatch<T> {
    domain: Domain,
    range: Range<u64>,
    payload: T,
}

impl<T> VectorBatch<T> {
    /// Construct a batch.
    pub(crate) fn new(domain: Domain, range: Range<u64>, payload: T) -> Self {
        Self {
            domain,
            range,
            payload,
        }
    }

    /// Domain carried by this batch.
    pub(crate) fn domain(&self) -> &Domain {
        &self.domain
    }

    /// Range covered by this batch.
    pub(crate) fn range(&self) -> &Range<u64> {
        &self.range
    }

    /// Number of rows or positions covered by the batch range.
    pub(crate) fn row_count(&self) -> u64 {
        self.range.end.saturating_sub(self.range.start)
    }
}
