//! Registry for extension types.
//!
//! The registry can hold both static and dynamically created types.

use std::sync::Arc;

use hashbrown::HashMap;
use itertools::Itertools;
use parking_lot::RwLock;

use crate::{ExtID, ExtensionTypeEncodingRef, ExtensionVTable};

/// Registry shared by dynamic and static extension types.
#[allow(clippy::disallowed_types)]
#[derive(Clone)]
pub struct ExtensionTypeRegistry {
    registry: Arc<RwLock<HashMap<String, ExtensionTypeEncodingRef>>>,
}

impl Default for ExtensionTypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ExtensionTypeRegistry {
    /// Create a new empty extension type registry.
    pub fn new() -> Self {
        Self {
            registry: Arc::default(),
        }
    }

    /// Register an encoding.
    pub fn register(&self, encoding: ExtensionTypeEncodingRef) -> &Self {
        self.registry
            .write()
            .insert(encoding.id().to_string(), encoding);
        self
    }

    /// Returns `true` if the provided extension VTable has its encoding registered.
    pub fn contains<V: ExtensionVTable>(&self) -> bool {
        self.registry
            .read()
            .values()
            .find_or_first(|encoding| encoding.is::<V>())
            .is_some()
    }

    /// Lookup the encoding that supports serde for the given extension type.
    pub fn find_encoding(&self, id: &ExtID) -> Option<ExtensionTypeEncodingRef> {
        self.registry
            .read()
            .values()
            .find(|decoder| decoder.supports_type(id))
            .cloned()
    }
}
