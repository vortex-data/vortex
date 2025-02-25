use std::sync::Arc;

use crate::aliases::hash_map::HashMap;
use crate::arrays::{
    BoolEncoding, ChunkedEncoding, ConstantEncoding, ExtensionEncoding, ListEncoding, NullEncoding,
    PrimitiveEncoding, StructEncoding, VarBinEncoding, VarBinViewEncoding,
};
use crate::encoding::Encoding;
use crate::vtable::VTableRef;

/// A mapping between an encoding's ID to an [`VTableRef`], used to have a shared view of all available encoding schemes.
#[derive(Debug, Clone)]
pub struct Context {
    encodings: HashMap<u16, VTableRef>,
}

/// An atomic shared reference to a [`Context`].
pub type ContextRef = Arc<Context>;

impl Context {
    pub fn with_encoding(mut self, encoding: VTableRef) -> Self {
        self.encodings.insert(encoding.id().code(), encoding);
        self
    }

    pub fn with_encodings<E: IntoIterator<Item = VTableRef>>(mut self, encodings: E) -> Self {
        self.encodings
            .extend(encodings.into_iter().map(|e| (e.id().code(), e)));
        self
    }

    pub fn encodings(&self) -> impl Iterator<Item = VTableRef> + '_ {
        self.encodings.values().cloned()
    }

    pub fn lookup_encoding(&self, encoding_code: u16) -> Option<VTableRef> {
        self.encodings.get(&encoding_code).cloned()
    }
}

impl Default for Context {
    fn default() -> Self {
        Self {
            encodings: [
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
            ]
            .into_iter()
            .map(|e| (e.id().code(), e))
            .collect(),
        }
    }
}
