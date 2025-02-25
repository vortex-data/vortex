use std::sync::Arc;

use vortex_array::aliases::hash_map::HashMap;
use vortex_array::arrays::{
    BoolEncoding, ChunkedEncoding, ConstantEncoding, ExtensionEncoding, ListEncoding, NullEncoding,
    PrimitiveEncoding, StructEncoding, VarBinEncoding, VarBinViewEncoding,
};
use vortex_array::vtable::VTableRef;
use vortex_array::{Context, ContextRef, Encoding, EncodingId};
use vortex_error::{VortexResult, vortex_err};
use vortex_layout::layouts::chunked::ChunkedLayout;
use vortex_layout::layouts::flat::FlatLayout;
use vortex_layout::layouts::stats::StatsLayout;
use vortex_layout::layouts::struct_::StructLayout;
use vortex_layout::{LayoutContext, LayoutContextRef, LayoutId, LayoutVTableRef};

/// A registry of array and layout implementations that can be used when reading a Vortex file.
///
/// In the future, we will support loading encodings from shared libraries or even from within
/// the Vortex file itself. This registry will be used to manage the available encodings.
#[derive(Debug, Clone)]
pub struct Registry {
    array_encodings: HashMap<EncodingId, VTableRef>,
    layout_encodings: HashMap<LayoutId, LayoutVTableRef>,
}

impl Default for Registry {
    fn default() -> Self {
        let mut this = Self {
            array_encodings: Default::default(),
            layout_encodings: Default::default(),
        };

        // Register the canonical encodings
        this = this
            .register_array(NullEncoding.vtable())
            .register_array(BoolEncoding.vtable())
            .register_array(PrimitiveEncoding.vtable())
            .register_array(StructEncoding.vtable())
            .register_array(ListEncoding.vtable())
            .register_array(VarBinEncoding.vtable())
            .register_array(VarBinViewEncoding.vtable())
            .register_array(ExtensionEncoding.vtable());

        // Register the utility encodings
        this = this
            .register_array(ConstantEncoding.vtable())
            .register_array(ChunkedEncoding.vtable());

        // Register the layout encodings
        this = this
            .register_layout(LayoutVTableRef::new_ref(&FlatLayout))
            .register_layout(LayoutVTableRef::new_ref(&StructLayout))
            .register_layout(LayoutVTableRef::new_ref(&StatsLayout))
            .register_layout(LayoutVTableRef::new_ref(&ChunkedLayout));

        this
    }
}

impl Registry {
    /// Create a new [`ContextRef`] with the provided encodings.
    pub fn new_context(&self, encodings: &[EncodingId]) -> VortexResult<ContextRef> {
        let mut ctx = Context::empty();
        for encoding in encodings {
            let vtable = self
                .array_encodings
                .get(encoding)
                .ok_or_else(|| vortex_err!("Array encoding {} not found in registry", encoding))?;
            ctx = ctx.with_encoding(vtable.clone());
        }
        Ok(Arc::new(ctx))
    }

    /// Create a new [`LayoutContextRef`] with the provided encodings.
    pub fn new_layout_context(&self, encodings: &[LayoutId]) -> VortexResult<LayoutContextRef> {
        let mut ctx = LayoutContext::empty();
        for encoding in encodings {
            let vtable = self
                .layout_encodings
                .get(encoding)
                .ok_or_else(|| vortex_err!("Layout encoding {} not found in registry", encoding))?;
            ctx = ctx.with_layout(vtable.clone());
        }
        Ok(Arc::new(ctx))
    }

    /// Register a new array encoding, replacing any existing encoding with the same ID.
    pub fn register_array(mut self, encoding: VTableRef) -> Self {
        self.array_encodings.insert(encoding.id(), encoding);
        self
    }

    /// Register a new layout encoding, replacing any existing encoding with the same ID.
    pub fn register_layout(mut self, encoding: LayoutVTableRef) -> Self {
        self.layout_encodings.insert(encoding.id(), encoding);
        self
    }
}
