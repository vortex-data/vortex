// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arcref::ArcRef;
use parking_lot::RwLock;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_session::registry::Registry;

use crate::vtable::DynVTable;

pub type ArrayContext = VTableContext<&'static dyn DynVTable>;

/// A collection of encodings that can be addressed by a u16 positional index.
/// This is used to map array encodings and layout encodings when reading from a file.
#[derive(Debug, Clone)]
pub struct VTableContext<T> {
    ids: Arc<RwLock<Vec<ArcRef<str>>>>,
    registry: Registry<T>,
}

impl<T: Clone> VTableContext<T> {
    pub fn try_new(ids: Vec<ArcRef<str>>, registry: Registry<T>) -> VortexResult<Self> {
        for id in &ids {
            vortex_ensure!(
                registry.find(id).is_some(),
                "Registry missing encoding with id {}",
                id
            );
        }
        Ok(Self {
            ids: Arc::new(RwLock::new(ids)),
            registry,
        })
    }

    pub fn from_registry_sorted(registry: &Registry<T>) -> Self {
        let ids: Vec<_> = registry.ids().collect();
        Self {
            ids: Arc::new(RwLock::new(ids)),
            registry: registry.clone(),
        }
    }
    //
    // pub fn with(self, encoding: T) -> Self {
    //     {
    //         let mut write = self.0.write();
    //         if write.iter().all(|e| e != &encoding) {
    //             write.push(encoding);
    //         }
    //     }
    //     self
    // }
    //
    // pub fn with_many<E: IntoIterator<Item = T>>(self, items: E) -> Self {
    //     items.into_iter().fold(self, |ctx, e| ctx.with(e))
    // }
    //
    // pub fn encodings(&self) -> Vec<T> {
    //     self.0.read().clone()
    // }

    /// Returns the index of the encoding in the context, or adds it if it doesn't exist.
    ///
    /// At write time the order encodings are registered by this method can change.
    /// See [File Format specification](https://docs.vortex.rs/specs/file-format#file-determinism-and-reproducibility)
    /// for more details.
    pub fn encoding_idx(&self, id: &ArcRef<str>) -> u16 {
        let mut write = self.ids.write();
        if let Some(idx) = write.iter().position(|e| e == id) {
            return u16::try_from(idx).vortex_expect("Cannot have more than u16::MAX encodings");
        }
        assert!(
            write.len() < u16::MAX as usize,
            "Cannot have more than u16::MAX encodings"
        );
        write.push(id.clone());
        u16::try_from(write.len() - 1).vortex_expect("checked already")
    }

    /// Find an encoding by its position.
    pub fn lookup_encoding(&self, idx: u16) -> Option<(ArcRef<str>, T)> {
        let id = self.ids.read().get(idx as usize).cloned()?;
        self.registry.find(&id).map(|entry| (id, entry))
    }
}
