use std::fmt::Display;
use std::sync::{Arc, RwLock, RwLockReadGuard};

use itertools::Itertools;
use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::aliases::hash_map::HashMap;
use crate::arrays::{
    BoolEncoding, ChunkedEncoding, ConstantEncoding, DecimalEncoding, ExtensionEncoding,
    ListEncoding, NullEncoding, PrimitiveEncoding, StructEncoding, VarBinEncoding,
    VarBinViewEncoding,
};
use crate::encoding::Encoding;
use crate::vtable::VTableRef;

/// A collection of array encodings.
// TODO(ngates): it feels weird that this has interior mutability. I think maybe it shouldn't.
pub type ArrayContext = VTableContext<VTableRef>;
pub type ArrayRegistry = VTableRegistry<VTableRef>;

impl ArrayRegistry {
    pub fn canonical_only() -> Self {
        let mut this = Self::empty();

        // Register the canonical encodings
        this.register_many([
            NullEncoding.vtable(),
            BoolEncoding.vtable(),
            PrimitiveEncoding.vtable(),
            DecimalEncoding.vtable(),
            StructEncoding.vtable(),
            ListEncoding.vtable(),
            VarBinEncoding.vtable(),
            VarBinViewEncoding.vtable(),
            ExtensionEncoding.vtable(),
        ]);

        // Register the utility encodings
        this.register_many([ConstantEncoding.vtable(), ChunkedEncoding.vtable()]);

        this
    }
}

/// A collection of encodings that can be addressed by a u16 positional index.
/// This is used to map array encodings and layout encodings when reading from a file.
#[derive(Debug, Clone)]
pub struct VTableContext<T>(Arc<RwLock<Vec<T>>>);

impl<T: Clone + Eq> VTableContext<T> {
    pub fn empty() -> Self {
        Self(Arc::new(RwLock::new(Vec::new())))
    }

    pub fn with(self, encoding: T) -> Self {
        {
            let mut write = self.0.write().vortex_expect("poisoned lock");
            if write.iter().all(|e| e != &encoding) {
                write.push(encoding);
            }
        }
        self
    }

    pub fn with_many<E: IntoIterator<Item = T>>(self, items: E) -> Self {
        items.into_iter().fold(self, |ctx, e| ctx.with(e))
    }

    pub fn encodings(&self) -> RwLockReadGuard<Vec<T>> {
        self.0.read().vortex_expect("poisoned lock")
    }

    /// Returns the index of the encoding in the context, or adds it if it doesn't exist.
    pub fn encoding_idx(&self, encoding: &T) -> u16 {
        let mut write = self.0.write().vortex_expect("poisoned lock");
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
        self.0
            .read()
            .vortex_expect("poisoned lock")
            .get(idx as usize)
            .cloned()
    }
}

/// A registry of encodings that can be used to construct a context for serde.
///
/// In the future, we will support loading encodings from shared libraries or even from within
/// the Vortex file itself. This registry will be used to manage the available encodings.
#[derive(Debug)]
pub struct VTableRegistry<T>(HashMap<String, T>);

impl<T: Clone + Display + Eq> VTableRegistry<T> {
    pub fn empty() -> Self {
        Self(Default::default())
    }

    /// Create a new [`VTableContext`] with the provided encodings.
    pub fn new_context<'a>(
        &self,
        encoding_ids: impl Iterator<Item = &'a str>,
    ) -> VortexResult<VTableContext<T>> {
        let mut ctx = VTableContext::<T>::empty();
        for id in encoding_ids {
            let encoding = self.0.get(id).ok_or_else(|| {
                vortex_err!(
                    "Array encoding {} not found in registry {}",
                    id,
                    self.0.values().join(", ")
                )
            })?;
            ctx = ctx.with(encoding.clone());
        }
        Ok(ctx)
    }

    /// List the vtables in the registry.
    pub fn vtables(&self) -> impl Iterator<Item = &T> + '_ {
        self.0.values()
    }

    /// Register a new encoding, replacing any existing encoding with the same ID.
    pub fn register(&mut self, encoding: T) {
        self.0.insert(encoding.to_string(), encoding);
    }

    /// Register a new encoding, replacing any existing encoding with the same ID.
    pub fn register_many<I: IntoIterator<Item = T>>(&mut self, encodings: I) {
        self.0
            .extend(encodings.into_iter().map(|e| (e.to_string(), e)));
    }
}
