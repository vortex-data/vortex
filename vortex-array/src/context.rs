// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::sync::Arc;

use itertools::Itertools;
use parking_lot::RwLock;
use vortex_error::{VortexExpect, VortexResult, vortex_bail, vortex_err};
use vortex_session::registry::Registry;

use crate::vtable::ArrayVTable;

pub type ArrayContext = VTableContext<ArrayVTable>;

/// A collection of encodings that can be addressed by a u16 positional index.
/// This is used to map array encodings and layout encodings when reading from a file.
#[derive(Debug, Clone)]
pub struct VTableContext<T>(Arc<RwLock<Vec<T>>>);

impl<T: Clone + Eq> VTableContext<T> {
    pub fn new(encodings: Vec<T>) -> Self {
        Self(Arc::new(RwLock::new(encodings)))
    }

    pub fn try_from_registry<'a>(
        registry: &Registry<T>,
        ids: impl IntoIterator<Item = &'a str>,
    ) -> VortexResult<Self>
    where
        T: Display,
    {
        let items: Vec<T> = ids
            .into_iter()
            .map(|id| {
                registry
                    .find(id)
                    .ok_or_else(|| vortex_err!("Registry missing encoding with id {}", id))
            })
            .try_collect()?;
        if items.len() > u16::MAX as usize {
            vortex_bail!(
                "Cannot create VTableContext: registry has more than u16::MAX ({}) items",
                u16::MAX
            );
        }
        Ok(Self::new(items))
    }

    pub fn empty() -> Self {
        Self(Arc::new(RwLock::new(Vec::new())))
    }

    pub fn with(self, encoding: T) -> Self {
        {
            let mut write = self.0.write();
            if write.iter().all(|e| e != &encoding) {
                write.push(encoding);
            }
        }
        self
    }

    pub fn with_many<E: IntoIterator<Item = T>>(self, items: E) -> Self {
        items.into_iter().fold(self, |ctx, e| ctx.with(e))
    }

    pub fn encodings(&self) -> Vec<T> {
        self.0.read().clone()
    }

    /// Returns the index of the encoding in the context, or adds it if it doesn't exist.
    ///
    /// At write time the order encodings are registered by this method can change.
    /// See [File Format specification](https://docs.vortex.rs/specs/file-format#file-determinism-and-reproducibility)
    /// for more details.
    pub fn encoding_idx(&self, encoding: &T) -> u16 {
        let mut write = self.0.write();
        if let Some(idx) = write.iter().position(|e| e == encoding) {
            return u16::try_from(idx).vortex_expect("Cannot have more than u16::MAX encodings");
        }
        assert!(
            write.len() < u16::MAX as usize,
            "Cannot have more than u16::MAX encodings"
        );
        write.push(encoding.clone());
        u16::try_from(write.len() - 1).vortex_expect("checked already")
    }

    /// Find an encoding by its position.
    pub fn lookup_encoding(&self, idx: u16) -> Option<T> {
        self.0.read().get(idx as usize).cloned()
    }
}
