// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Registration and serialization contracts for independent read and write IDs.

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
fn array_session_default_registers_encodings() {
    let session = VortexSession::empty().with::<ArraySession>();

    let id = ArrayVTable::id(&Bool);

    assert!(session.arrays().registry().contains_key(&id));
    assert!(session.arrays().serializer(&id).is_some());
}

#[test]
fn empty_array_session_registers_no_encodings() {
    let session = VortexSession::empty().with_some(ArraySession::empty());

    let id = ArrayVTable::id(&Bool);

    assert!(!session.arrays().registry().contains_key(&id));
    assert!(session.arrays().serializer(&id).is_none());
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
