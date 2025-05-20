//! Registry for extension types.
//!
//! The registry can hold both static and dynamically created types.

use std::sync::Arc;

use hashbrown::HashMap;
use parking_lot::RwLock;

use crate::{ExtID, ExtensionTypeEncodingRef};

/// Registry shared by dynamic and static extension types.
#[allow(clippy::disallowed_types)]
#[derive(Clone)]
pub struct ExtensionTypeRegistry {
    registry: Arc<RwLock<HashMap<ExtID, ExtensionTypeEncodingRef>>>,
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
        self.registry.write().insert(encoding.id(), encoding);
        self
    }
    
    pub fn 
}
