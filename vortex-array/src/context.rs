use std::fmt::Display;
use std::sync::Arc;

use itertools::Itertools;
use parking_lot::RwLock;
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_utils::aliases::hash_map::HashMap;

use crate::EncodingRef;
use crate::arrays::{
    BoolEncoding, ChunkedEncoding, ConstantEncoding, DecimalEncoding, ExtensionEncoding,
    ListEncoding, NullEncoding, PrimitiveEncoding, StructEncoding, VarBinEncoding,
    VarBinViewEncoding,
};

/// A collection of array encodings.
// TODO(ngates): it feels weird that this has interior mutability. I think maybe it shouldn't.
pub type ArrayContext = VTableContext<EncodingRef>;
pub type ArrayRegistry = VTableRegistry<EncodingRef>;
pub type ArrayRegistryBuilder = RegistryBuilder<EncodingRef>;

impl ArrayRegistryBuilder {
    /// Build a new ArrayRegistry with only the [canonical][crate::Canonical] encodings.
    ///
    /// This ArrayRegistry can encode any Apache Arrow-compatible arrays, but is unaware
    /// of the compressed encodings.
    pub fn register_canonical(self) -> Self {
        // Register the canonical encodings
        self.register_many([
            EncodingRef::new_ref(NullEncoding.as_ref()) as EncodingRef,
            EncodingRef::new_ref(BoolEncoding.as_ref()),
            EncodingRef::new_ref(PrimitiveEncoding.as_ref()),
            EncodingRef::new_ref(DecimalEncoding.as_ref()),
            EncodingRef::new_ref(StructEncoding.as_ref()),
            EncodingRef::new_ref(ListEncoding.as_ref()),
            EncodingRef::new_ref(VarBinEncoding.as_ref()),
            EncodingRef::new_ref(VarBinViewEncoding.as_ref()),
            EncodingRef::new_ref(ExtensionEncoding.as_ref()),
        ])
        // Register the utility encodings
        .register_many([
            EncodingRef::new_ref(ConstantEncoding.as_ref()) as EncodingRef,
            EncodingRef::new_ref(ChunkedEncoding.as_ref()),
        ])
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

/// An immutable registry of encodings that can be used to construct a context for serde.
///
/// In the future, we will support loading encodings from shared libraries or even from within
/// the Vortex file itself. This registry will be used to manage the available encodings.
#[derive(Clone, Debug)]
pub struct VTableRegistry<T>(Arc<HashMap<String, T>>);

/// Builder for [`VTableRegistry`].
///
/// Users should construct one of these, call the `register` or `register_many` methods, and then
/// `build` to get the final registry.
#[derive(Debug, Default)]
pub struct RegistryBuilder<T>(HashMap<String, T>);

impl<T: Clone + Display + Eq> RegistryBuilder<T> {
    /// Create a new empty registry builder.
    pub fn new() -> Self {
        Self(Default::default())
    }

    /// Register a new item in the registry.
    ///
    /// The item must have some sort of string-based name as well.
    pub fn register(mut self, item: T) -> Self {
        self.0.insert(item.to_string(), item);
        self
    }

    /// Register a new encoding, replacing any existing encoding with the same ID.
    pub fn register_many<I: IntoIterator<Item = T>>(mut self, encodings: I) -> Self {
        self.0
            .extend(encodings.into_iter().map(|e| (e.to_string(), e)));
        self
    }

    pub fn build(self) -> VTableRegistry<T> {
        VTableRegistry(Arc::new(self.0))
    }
}

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
}
