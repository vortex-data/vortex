mod pruning;

pub use pruning::ZoneMapResource;

use std::collections::BTreeMap;
use std::collections::BTreeSet;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResourceValue {
    KeyIndex(BTreeMap<i64, Vec<usize>>),
    KeySet(BTreeSet<i64>),
    Scalar(i64),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResourceKind {
    KeyIndex,
    KeySet,
    Scalar,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResourceSpec {
    pub id: String,
    pub kind: ResourceKind,
}

#[derive(Clone, Debug)]
pub struct Resource {
    spec: ResourceSpec,
    value: Option<ResourceValue>,
    ready: bool,
}

impl Resource {
    pub fn new(spec: ResourceSpec) -> Self {
        Self {
            spec,
            value: None,
            ready: false,
        }
    }

    pub fn id(&self) -> &str {
        self.spec.id.as_str()
    }

    pub const fn is_ready(&self) -> bool {
        self.ready
    }

    pub fn publish(&mut self, value: ResourceValue) -> bool {
        self.value = Some(value);
        let changed = !self.ready;
        self.ready = true;
        changed
    }

    pub fn value(&self) -> Option<&ResourceValue> {
        self.value.as_ref()
    }
}
