use std::sync::Arc;

use crate::aliases::hash_map::HashMap;
use crate::array::{
    BoolEncoding, ChunkedEncoding, ConstantEncoding, ExtensionEncoding, ListEncoding, NullEncoding,
    PrimitiveEncoding, SparseEncoding, StructEncoding, VarBinEncoding, VarBinViewEncoding,
};
use crate::encoding::EncodingRef;

/// A mapping between an encoding's ID to an [`EncodingRef`], used to have a shared view of all available encoding schemes.
#[derive(Debug, Clone)]
pub struct Context {
    encodings: HashMap<u16, EncodingRef>,
}

/// An atomic shared reference to a [`Context`].
pub type ContextRef = Arc<Context>;

impl Context {
    pub fn with_encoding(mut self, encoding: EncodingRef) -> Self {
        self.encodings.insert(encoding.id().code(), encoding);
        self
    }

    pub fn with_encodings<E: IntoIterator<Item = EncodingRef>>(mut self, encodings: E) -> Self {
        self.encodings
            .extend(encodings.into_iter().map(|e| (e.id().code(), e)));
        self
    }

    pub fn encodings(&self) -> impl Iterator<Item = EncodingRef> + '_ {
        self.encodings.values().cloned()
    }

    pub fn lookup_encoding(&self, encoding_code: u16) -> Option<EncodingRef> {
        self.encodings.get(&encoding_code).cloned()
    }
}

impl Default for Context {
    fn default() -> Self {
        Self {
            encodings: [
                &NullEncoding as EncodingRef,
                &BoolEncoding,
                &PrimitiveEncoding,
                &StructEncoding,
                &ListEncoding,
                &VarBinEncoding,
                &VarBinViewEncoding,
                &ExtensionEncoding,
                &SparseEncoding,
                &ConstantEncoding,
                &ChunkedEncoding,
            ]
            .into_iter()
            .map(|e| (e.id().code(), e))
            .collect(),
        }
    }
}
