use std::sync::{Arc, RwLock};

use vortex_error::VortexExpect;

use crate::arrays::{
    BoolEncoding, ChunkedEncoding, ConstantEncoding, ExtensionEncoding, ListEncoding, NullEncoding,
    PrimitiveEncoding, StructEncoding, VarBinEncoding, VarBinViewEncoding,
};
use crate::encoding::Encoding;
use crate::vtable::VTableRef;

/// A set of Array encodings.
#[derive(Debug, Clone)]
pub struct Context(Arc<RwLock<Vec<VTableRef>>>);

impl Context {
    pub fn empty() -> Self {
        Self(Arc::new(RwLock::new(Vec::new())))
    }

    pub fn with_encoding(self, encoding: VTableRef) -> Self {
        {
            let mut write = self.0.write().vortex_expect("poisoned lock");
            if write.iter().all(|e| e.id() != encoding.id()) {
                write.push(encoding);
            }
        }
        self
    }

    pub fn with_encodings<E: IntoIterator<Item = VTableRef>>(self, encodings: E) -> Self {
        encodings
            .into_iter()
            .fold(self, |ctx, e| ctx.with_encoding(e))
    }

    pub fn encodings(&self) -> Vec<VTableRef> {
        self.0.read().vortex_expect("poisoned lock").clone()
    }

    /// Returns the index of the encoding in the context, or adds it if it doesn't exist.
    pub fn encoding_idx(&self, encoding: &VTableRef) -> u16 {
        let mut write = self.0.write().vortex_expect("poisoned lock");
        if let Some(idx) = write.iter().position(|e| e.id() == encoding.id()) {
            return u16::try_from(idx).vortex_expect("Cannot have more than u16::MAX encodings");
        }
        if write.len() >= u16::MAX as usize {
            panic!("Cannot have more than u16::MAX encodings");
        }
        write.push(encoding.clone());
        u16::try_from(write.len() - 1).vortex_expect("checked already")
    }

    /// Find an encoding by its position.
    pub fn lookup_encoding(&self, idx: u16) -> Option<VTableRef> {
        self.0
            .read()
            .vortex_expect("poisoned lock")
            .get(idx as usize)
            .cloned()
    }
}

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
