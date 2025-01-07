use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use bytes::Bytes;
use vortex_array::aliases::hash_map::HashMap;
use vortex_error::{vortex_err, VortexExpect};

use crate::read::{LayoutPartId, MessageId};

/// A read-only cache of messages.
pub trait MessageCache {
    fn get(&self, path: &[LayoutPartId]) -> Option<Bytes>;
}

#[derive(Default, Debug, Clone)]
pub struct LayoutMessageCache {
    cache: Arc<RwLock<HashMap<MessageId, Bytes>>>,
}

impl LayoutMessageCache {
    pub fn remove(&self, path: &[LayoutPartId]) -> Option<Bytes> {
        self.cache
            .write()
            .map_err(|_| vortex_err!("Poisoned cache"))
            .vortex_expect("poisoned")
            .remove(path)
    }

    pub fn set(&self, path: MessageId, value: Bytes) {
        self.cache
            .write()
            .map_err(|_| vortex_err!("Poisoned cache"))
            .vortex_expect("poisoned")
            .insert(path, value);
    }

    pub fn set_many<I: IntoIterator<Item = (MessageId, Bytes)>>(&self, iter: I) {
        let mut guard = self
            .cache
            .write()
            .map_err(|_| vortex_err!("Poisoned cache"))
            .vortex_expect("poisoned");
        for (id, bytes) in iter.into_iter() {
            guard.insert(id, bytes);
        }
    }
}

impl MessageCache for LayoutMessageCache {
    fn get(&self, path: &[LayoutPartId]) -> Option<Bytes> {
        self.cache
            .read()
            .map_err(|_| vortex_err!("Poisoned cache"))
            .vortex_expect("poisoned")
            .get(path)
            .cloned()
    }
}
