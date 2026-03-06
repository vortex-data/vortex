// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;

use crate::arrays::bool::BoolVTable;
use crate::arrays::chunked::ChunkedVTable;
use crate::arrays::constant::ConstantVTable;
use crate::arrays::decimal::DecimalVTable;
use crate::arrays::extension::ExtensionVTable;
use crate::arrays::fixed_size_list::FixedSizeListVTable;
use crate::arrays::list::ListVTable;
use crate::arrays::listview::ListViewVTable;
use crate::arrays::masked::MaskedVTable;
use crate::arrays::null::NullVTable;
use crate::arrays::primitive::PrimitiveVTable;
use crate::arrays::struct_::StructVTable;
use crate::arrays::varbin::VarBinVTable;
use crate::arrays::varbinview::VarBinViewVTable;
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
        encodings.register(NullVTable::ID, NullVTable);
        encodings.register(BoolVTable::ID, BoolVTable);
        encodings.register(PrimitiveVTable::ID, PrimitiveVTable);
        encodings.register(DecimalVTable::ID, DecimalVTable);
        encodings.register(VarBinViewVTable::ID, VarBinViewVTable);
        encodings.register(ListViewVTable::ID, ListViewVTable);
        encodings.register(FixedSizeListVTable::ID, FixedSizeListVTable);
        encodings.register(StructVTable::ID, StructVTable);
        encodings.register(ExtensionVTable::ID, ExtensionVTable);

        // Register the utility encodings.
        encodings.register(ChunkedVTable::ID, ChunkedVTable);
        encodings.register(ConstantVTable::ID, ConstantVTable);
        encodings.register(MaskedVTable::ID, MaskedVTable);
        encodings.register(ListVTable::ID, ListVTable);
        encodings.register(VarBinVTable::ID, VarBinVTable);

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
