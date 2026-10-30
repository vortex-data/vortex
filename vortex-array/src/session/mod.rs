// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_session::ArcSwapMap;
use vortex_session::SessionExt;
use vortex_session::SessionGuard;
use vortex_session::SessionVar;
use vortex_session::registry::Id;

use crate::ArrayRef;
use crate::array::ArrayId;
use crate::array::ArrayPlugin;
use crate::array::ArrayPluginRef;
use crate::array::ArraySerialization;
use crate::arrays::Bool;
use crate::arrays::Chunked;
use crate::arrays::Constant;
use crate::arrays::Decimal;
use crate::arrays::Dict;
use crate::arrays::Extension;
use crate::arrays::FixedSizeList;
use crate::arrays::List;
use crate::arrays::ListView;
use crate::arrays::Map;
use crate::arrays::Masked;
use crate::arrays::Null;
use crate::arrays::PiecewiseSequence;
use crate::arrays::Primitive;
use crate::arrays::RunEnd;
use crate::arrays::Struct;
use crate::arrays::Union;
use crate::arrays::VarBin;
use crate::arrays::VarBinView;
use crate::arrays::Variant;

/// Registry of array encodings.
pub type ArrayRegistry = ArcSwapMap<Id, ArrayPluginRef>;

#[derive(Clone, Debug)]
pub struct ArraySession {
    /// Deserializers keyed by the array ID found on the wire.
    registry: ArrayRegistry,
    /// Serializers keyed by the in-memory array encoding ID.
    serializers: ArrayRegistry,
}

impl ArraySession {
    pub fn empty() -> ArraySession {
        Self {
            registry: ArrayRegistry::default(),
            serializers: ArrayRegistry::default(),
        }
    }

    pub fn registry(&self) -> &ArrayRegistry {
        &self.registry
    }

    /// Register an in-memory array writer and its declared wire readers.
    ///
    /// This replaces any serializer with the same in-memory ID and any deserializer registered
    /// under one of [`ArrayPlugin::read_ids`].
    pub fn register<P: ArrayPlugin>(&self, plugin: P) {
        let plugin = Arc::new(plugin) as ArrayPluginRef;
        self.serializers.insert(plugin.id(), Arc::clone(&plugin));
        for serialized_id in plugin.read_ids() {
            self.registry.insert(serialized_id, Arc::clone(&plugin));
        }
    }

    fn serializer(&self, id: &ArrayId) -> Option<ArrayPluginRef> {
        self.serializers.get(id)
    }
}

impl Default for ArraySession {
    fn default() -> Self {
        let this = ArraySession {
            registry: ArrayRegistry::default(),
            serializers: ArrayRegistry::default(),
        };

        // Register the canonical encodings.
        this.register(Null);
        this.register(Bool);
        this.register(Primitive);
        this.register(Decimal);
        this.register(VarBinView);
        this.register(ListView);
        this.register(Map);
        this.register(FixedSizeList);
        this.register(Struct);
        this.register(Union);
        this.register(Variant);
        this.register(Extension);

        // Register the utility encodings.
        this.register(Chunked);
        this.register(Constant);
        this.register(Dict);
        this.register(List);
        this.register(Masked);
        this.register(PiecewiseSequence);
        this.register(RunEnd);
        this.register(VarBin);

        this
    }
}

impl SessionVar for ArraySession {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Session data for Vortex arrays.
pub trait ArraySessionExt: SessionExt {
    /// Returns the array encoding registry.
    fn arrays(&self) -> SessionGuard<'_, ArraySession> {
        self.get::<ArraySession>()
    }

    /// Serialize an array using a plugin from the registry.
    fn array_serialize(&self, array: &ArrayRef) -> VortexResult<Option<ArraySerialization>> {
        let Some(plugin) = self.arrays().serializer(&array.encoding_id()) else {
            vortex_bail!(
                "Array {} is not registered for serialization",
                array.encoding_id()
            );
        };

        let Some(serialization) = plugin.serialize(array, &self.session())? else {
            return Ok(None);
        };
        vortex_ensure!(
            plugin.write_ids().contains(&serialization.serialized_id),
            "array serializer {} produced undeclared serialized ID {}",
            array.encoding_id(),
            serialization.serialized_id,
        );
        Ok(Some(serialization))
    }
}

impl<S: SessionExt> ArraySessionExt for S {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use rstest::rstest;
    use vortex_error::VortexResult;
    use vortex_error::vortex_bail;
    use vortex_session::VortexSession;
    use vortex_session::registry::Id;

    use crate::ArrayDeserialization;
    use crate::ArrayId;
    use crate::ArrayPlugin;
    use crate::ArrayRef;
    use crate::ArraySerialization;
    use crate::ArrayVTable;
    use crate::IntoArray;
    use crate::arrays::Bool;
    use crate::arrays::Primitive;
    use crate::arrays::PrimitiveArray;
    use crate::session::ArraySession;
    use crate::session::ArraySessionExt;

    #[test]
    fn array_session_default_registers_encodings() {
        let session = VortexSession::empty().with::<ArraySession>();

        assert!(
            session
                .arrays()
                .registry()
                .contains_key(&ArrayVTable::id(&Bool))
        );
        assert!(
            session
                .arrays()
                .serializer(&ArrayVTable::id(&Bool))
                .is_some()
        );
    }

    #[test]
    fn empty_array_session_registers_no_encodings() {
        let session = VortexSession::empty().with_some(ArraySession::empty());

        assert!(
            !session
                .arrays()
                .registry()
                .contains_key(&ArrayVTable::id(&Bool))
        );
        assert!(
            session
                .arrays()
                .serializer(&ArrayVTable::id(&Bool))
                .is_none()
        );
    }

    struct WriteOnlyPrimitivePlugin {
        serialized_id: ArrayId,
    }

    impl ArrayPlugin for WriteOnlyPrimitivePlugin {
        fn id(&self) -> ArrayId {
            ArrayVTable::id(&Primitive)
        }

        fn read_ids(&self) -> Vec<ArrayId> {
            vec![Id::from("vortex.test.read_only")]
        }

        fn write_ids(&self) -> Vec<ArrayId> {
            vec![self.id()]
        }

        fn serialize(
            &self,
            array: &ArrayRef,
            _session: &VortexSession,
        ) -> VortexResult<Option<ArraySerialization>> {
            Ok(Some(ArraySerialization::from_array(
                self.serialized_id,
                array,
                vec![],
            )))
        }

        fn deserialize(
            &self,
            _parts: ArrayDeserialization<'_>,
            _session: &VortexSession,
        ) -> VortexResult<ArrayRef> {
            vortex_bail!("the registration test does not deserialize nodes")
        }
    }

    #[test]
    fn registration_preserves_the_reader_of_a_write_only_id() {
        let session = VortexSession::empty().with::<ArraySession>();
        let primitive_id = ArrayVTable::id(&Primitive);
        let reader = session.arrays().registry().get(&primitive_id).unwrap();
        let plugin = WriteOnlyPrimitivePlugin {
            serialized_id: primitive_id,
        };
        let read_id = plugin.read_ids()[0];
        session.arrays().register(plugin);

        let arrays = session.arrays();
        assert!(Arc::ptr_eq(
            &arrays.registry().get(&primitive_id).unwrap(),
            &reader
        ));
        assert!(Arc::ptr_eq(
            &arrays.registry().get(&read_id).unwrap(),
            &arrays.serializer(&primitive_id).unwrap(),
        ));
    }

    #[rstest]
    #[case::write_only(ArrayVTable::id(&Primitive), true)]
    #[case::read_only(Id::from("vortex.test.read_only"), false)]
    #[case::undeclared(Id::from("vortex.test.undeclared"), false)]
    fn serialization_checks_only_write_ids(
        #[case] serialized_id: ArrayId,
        #[case] accepted: bool,
    ) -> VortexResult<()> {
        let session = VortexSession::empty().with_some(ArraySession::empty());
        session
            .arrays()
            .register(WriteOnlyPrimitivePlugin { serialized_id });
        let array = PrimitiveArray::from_iter([1i32, 2, 3]).into_array();

        let result = session.array_serialize(&array);
        if accepted {
            assert_eq!(result?.unwrap().serialized_id, serialized_id);
            assert!(!session.arrays().registry().contains_key(&serialized_id));
        } else {
            let error = result.expect_err("only declared write IDs may be returned");
            assert!(
                error
                    .to_string()
                    .contains("produced undeclared serialized ID")
            );
        }

        Ok(())
    }
}
