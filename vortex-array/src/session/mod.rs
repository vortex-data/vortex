// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::Ref;
use vortex_session::SessionExt;
use vortex_session::registry::Registry;

use crate::ArrayRef;
use crate::array::ArrayPlugin;
use crate::array::ArrayPluginRef;
use crate::arrays::Bool;
use crate::arrays::Chunked;
use crate::arrays::Constant;
use crate::arrays::Decimal;
use crate::arrays::Dict;
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
use crate::arrays::Variant;

pub type ArrayRegistry = Registry<ArrayPluginRef>;

#[derive(Debug)]
pub struct ArraySession {
    /// The set of registered array encodings.
    registry: ArrayRegistry,
}

impl ArraySession {
    pub fn empty() -> ArraySession {
        Self {
            registry: ArrayRegistry::default(),
        }
    }

    pub fn registry(&self) -> &ArrayRegistry {
        &self.registry
    }

    /// Register a new array encoding, replacing any existing encoding with the same ID.
    pub fn register<P: ArrayPlugin>(&self, plugin: P) {
        self.registry
            .register(plugin.id(), Arc::new(plugin) as ArrayPluginRef);
    }
}

impl Default for ArraySession {
    fn default() -> Self {
        let this = ArraySession {
            registry: ArrayRegistry::default(),
        };

        // Register the canonical encodings.
        this.register(Null);
        this.register(Bool);
        this.register(Primitive);
        this.register(Decimal);
        this.register(VarBinView);
        this.register(ListView);
        this.register(FixedSizeList);
        this.register(Struct);
        this.register(Variant);
        this.register(Extension);

        // Register the utility encodings.
        this.register(Chunked);
        this.register(Constant);
        this.register(Dict);
        this.register(List);
        this.register(Masked);
        this.register(VarBin);

        this
    }
}

/// Session data for Vortex arrays.
pub trait ArraySessionExt: SessionExt {
    /// Returns the array encoding registry.
    fn arrays(&self) -> Ref<'_, ArraySession> {
        self.get::<ArraySession>()
    }

    /// Serialize an array using a plugin from the overlay registry first, then the session registry.
    fn array_serialize_with_overlay(
        &self,
        array: &ArrayRef,
        overlay: Option<&ArrayRegistry>,
    ) -> VortexResult<Option<Vec<u8>>> {
        let plugin = overlay
            .and_then(|registry| registry.find(&array.encoding_id()))
            .or_else(|| self.arrays().registry.find(&array.encoding_id()))
            .ok_or_else(|| {
                vortex_err!(
                    "Array {} is not registered for serialization",
                    array.encoding_id()
                )
            })?;

        plugin.serialize(array, &self.session())
    }

    /// Serialize an array using a plugin from the registry.
    fn array_serialize(&self, array: &ArrayRef) -> VortexResult<Option<Vec<u8>>> {
        self.array_serialize_with_overlay(array, None)
    }
}

impl<S: SessionExt> ArraySessionExt for S {}
