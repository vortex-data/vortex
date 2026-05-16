mod requirement;

pub use requirement::*;

use std::sync::Arc;

use crate::DomainId;
use crate::RelationId;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cardinality {
    Unknown,
    Exact(u64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DomainSpan {
    start: u64,
    len: u64,
}

impl DomainSpan {
    pub const fn new(start: u64, len: u64) -> Self {
        Self { start, len }
    }

    pub const fn from_len(len: u64) -> Self {
        Self::new(0, len)
    }

    pub const fn start(&self) -> u64 {
        self.start
    }

    pub const fn len(&self) -> u64 {
        self.len
    }

    pub const fn end(&self) -> u64 {
        self.start.saturating_add(self.len)
    }

    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Domain(Arc<DomainInner>);

#[derive(Debug, PartialEq, Eq)]
struct DomainInner {
    id: DomainId,
    cardinality: Cardinality,
}

impl Domain {
    pub fn new(id: DomainId, cardinality: Cardinality) -> Self {
        Self(Arc::new(DomainInner { id, cardinality }))
    }

    pub fn id(&self) -> &DomainId {
        &self.0.id
    }

    pub fn cardinality(&self) -> Cardinality {
        self.0.cardinality
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelationKind {
    Identity,
    SelectedSubset,
    ParentChildPrefix,
    ZoneToRows,
    DictionaryCodes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelationCardinality {
    OneToOne,
    OneToMany,
    ManyToOne,
    ManyToMany,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RelationTotality {
    Total,
    Partial,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Relation(Arc<RelationInner>);

#[derive(Debug, PartialEq, Eq)]
struct RelationInner {
    id: RelationId,
    source: Domain,
    target: Domain,
    kind: RelationKind,
    cardinality: RelationCardinality,
    totality: RelationTotality,
}

impl Relation {
    pub fn new(
        id: RelationId,
        source: Domain,
        target: Domain,
        kind: RelationKind,
        cardinality: RelationCardinality,
        totality: RelationTotality,
    ) -> Self {
        Self(Arc::new(RelationInner {
            id,
            source,
            target,
            kind,
            cardinality,
            totality,
        }))
    }

    pub fn id(&self) -> &RelationId {
        &self.0.id
    }

    pub fn source(&self) -> &Domain {
        &self.0.source
    }

    pub fn target(&self) -> &Domain {
        &self.0.target
    }

    pub fn kind(&self) -> RelationKind {
        self.0.kind
    }

    pub fn cardinality(&self) -> RelationCardinality {
        self.0.cardinality
    }

    pub fn totality(&self) -> RelationTotality {
        self.0.totality
    }
}
