// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;

use crate::arrays::BoolVTable;
use crate::arrays::ChunkedVTable;
use crate::arrays::ConstantVTable;
use crate::arrays::DecimalVTable;
use crate::arrays::ExtensionVTable;
use crate::arrays::FixedSizeListVTable;
use crate::arrays::ListVTable;
use crate::arrays::ListViewVTable;
use crate::arrays::MaskedVTable;
use crate::arrays::NullVTable;
use crate::arrays::PrimitiveVTable;
use crate::arrays::StructVTable;
use crate::arrays::VarBinVTable;
use crate::arrays::VarBinViewVTable;
use crate::optimizer::ArrayOptimizer;
use crate::scalar_fns::cast::array::CastArrayReduce;
use crate::vtable::ArrayVTable;
use crate::vtable::ArrayVTableExt;

pub type ArrayRegistry = Registry<ArrayVTable>;

#[derive(Debug)]
pub struct ArraySession {
    /// The set of registered array encodings.
    registry: ArrayRegistry,

    /// The array optimizer containing rules.
    optimizer: ArrayOptimizer,
}

impl ArraySession {
    pub fn registry(&self) -> &ArrayRegistry {
        &self.registry
    }

    pub fn optimizer(&self) -> &ArrayOptimizer {
        &self.optimizer
    }

    pub fn optimizer_mut(&mut self) -> &mut ArrayOptimizer {
        &mut self.optimizer
    }

    /// Register a new array encoding, replacing any existing encoding with the same ID.
    pub fn register(&self, encoding: ArrayVTable) {
        self.registry.register(encoding)
    }

    /// Register many array encodings, replacing any existing encodings with the same ID.
    pub fn register_many(&self, encodings: impl IntoIterator<Item = ArrayVTable>) {
        self.registry.register_many(encodings);
    }
}

impl Default for ArraySession {
    fn default() -> Self {
        let encodings = ArrayRegistry::default();

        // Register the canonical encodings.
        encodings.register_many([
            NullVTable.as_vtable(),
            BoolVTable.as_vtable(),
            PrimitiveVTable.as_vtable(),
            DecimalVTable.as_vtable(),
            VarBinViewVTable.as_vtable(),
            ListViewVTable.as_vtable(),
            FixedSizeListVTable.as_vtable(),
            StructVTable.as_vtable(),
            ExtensionVTable.as_vtable(),
        ]);

        // Register the utility encodings.
        encodings.register_many([
            ChunkedVTable.as_vtable(),
            ConstantVTable.as_vtable(),
            MaskedVTable.as_vtable(),
            ListVTable.as_vtable(),
            VarBinVTable.as_vtable(),
        ]);

        let mut session = Self {
            registry: encodings,
            optimizer: ArrayOptimizer::default(),
        };

        let optimizer = session.optimizer_mut();

        // Scalar function rules
        optimizer.register_reduce_rule(CastArrayReduce);

        session
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
