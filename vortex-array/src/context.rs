use std::sync::{Arc, RwLock, RwLockReadGuard};

use vortex_error::VortexExpect;

use crate::arrays::{
    BoolEncoding, ChunkedEncoding, ConstantEncoding, ExtensionEncoding, ListEncoding, NullEncoding,
    PrimitiveEncoding, StructEncoding, VarBinEncoding, VarBinViewEncoding,
};
use crate::encoding::Encoding;
use crate::vtable::VTableRef;

/// A collection of array encodings.
// TODO(ngates): it feels weird that this has interior mutability. I think maybe it shouldn't.
pub type Context = EncodingContext<VTableRef>;

impl Default for Context {
    fn default() -> Self {
        Self(Arc::new(RwLock::new(vec![
            NullEncoding.vtable(),
            BoolEncoding.vtable(),
            PrimitiveEncoding.vtable(),
            StructEncoding.vtable(),
            ListEncoding.vtable(),
            VarBinEncoding.vtable(),
            VarBinViewEncoding.vtable(),
            ExtensionEncoding.vtable(),
            ConstantEncoding.vtable(),
            ChunkedEncoding.vtable(),
        ])))
    }
}

/// A collection of encodings that can be addressed by a u16 positional index.
/// This is used to map array encodings and layout encodings when reading from a file.
#[derive(Debug, Clone)]
pub struct EncodingContext<T>(Arc<RwLock<Vec<T>>>);

impl<T: Clone + Eq> EncodingContext<T> {
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
        if write.len() >= u16::MAX as usize {
            panic!("Cannot have more than u16::MAX encodings");
        }
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
