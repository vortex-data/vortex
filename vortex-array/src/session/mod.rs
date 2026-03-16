// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;

use crate::arrays::Bool;
use crate::arrays::Chunked;
use crate::arrays::Constant;
use crate::arrays::Decimal;
use crate::arrays::Extension;
use crate::arrays::FixedSizeList;
use crate::arrays::List;
use crate::arrays::ListView;
use crate::arrays::Masked;
use crate::arrays::Null;
use crate::arrays::Primitive;
use crate::arrays::Struct;
use crate::arrays::VarBin;
use crate::arrays::VarBinView;
use crate::vtable::ArrayId;
use crate::vtable::DynVTable;

pub type ArrayRegistry = Registry<&'static dyn DynVTable>;

#[derive(Debug)]
pub struct ArraySession {
    /// The set of registered array encodings.
    registry: ArrayRegistry,
}

impl ArraySession {
    pub fn registry(&self) -> &ArrayRegistry {
        &self.registry
    }

    /// Register a new array encoding, replacing any existing encoding with the same ID.
    pub fn register(&self, id: impl Into<ArrayId>, encoding: impl Into<&'static dyn DynVTable>) {
        self.registry.register(id.into(), encoding.into())
    }
}

impl Default for ArraySession {
    fn default() -> Self {
        let encodings = ArrayRegistry::default();

        // Register the canonical encodings.
        encodings.register(Null::ID, Null);
        encodings.register(Bool::ID, Bool);
        encodings.register(Primitive::ID, Primitive);
        encodings.register(Decimal::ID, Decimal);
        encodings.register(VarBinView::ID, VarBinView);
        encodings.register(ListView::ID, ListView);
        encodings.register(FixedSizeList::ID, FixedSizeList);
        encodings.register(Struct::ID, Struct);
        encodings.register(Extension::ID, Extension);

        // Register the utility encodings.
        encodings.register(Chunked::ID, Chunked);
        encodings.register(Constant::ID, Constant);
        encodings.register(Masked::ID, Masked);
        encodings.register(List::ID, List);
        encodings.register(VarBin::ID, VarBin);

        Self {
            registry: encodings,
        }
    }
}

/// Session data for Vortex arrays.
pub trait ArraySessionExt: SessionExt {
    /// Returns the array encoding registry.
    fn arrays(&self) -> Ref<'_, ArraySession> {
        self.get::<ArraySession>()
    }
}

impl<S: SessionExt> ArraySessionExt for S {}
