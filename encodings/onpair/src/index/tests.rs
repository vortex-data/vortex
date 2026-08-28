// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use prost::Message;
use vortex_array::Array;
use vortex_array::ArrayContext;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::VarBinArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::serde::SerializeOptions;
use vortex_array::serde::SerializedArray;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_error::VortexResult;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use super::OnPairIndexSet;
use crate::DEFAULT_CONFIG;
use crate::OnPair;
use crate::OnPairArray;
use crate::OnPairArrayExt;
use crate::OnPairArraySlotsExt;
use crate::OnPairMetadata;
use crate::OnPairSlots;
use crate::build_token_frequency_index;
use crate::onpair_compress;

const FREQUENCY_INDEX: OnPairIndexSet = OnPairIndexSet::empty().with_token_frequency();

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = vortex_array::array_session();
    crate::initialize(&session);
    session
});

fn input(nullable: bool) -> ArrayRef {
    if nullable {
        VarBinArray::from_iter(
            [Some("alpha"), None, Some("beta")],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array()
    } else {
        VarBinArray::from_iter(
            [Some("alpha"), Some("alphabet"), Some("beta")],
            DType::Utf8(Nullability::NonNullable),
        )
        .into_array()
    }
}

fn encode(input: &ArrayRef, indexes: OnPairIndexSet) -> VortexResult<OnPairArray> {
    let mut ctx = SESSION.create_execution_ctx();
    let encoded = onpair_compress(input, DEFAULT_CONFIG, &mut ctx)?
        .try_downcast::<OnPair>()
        .map_err(|array| {
            vortex_error::vortex_err!("expected OnPair, got {}", array.encoding_id())
        })?;
    if indexes.has_token_frequency() {
        build_token_frequency_index(encoded, &mut ctx)
    } else {
        Ok(encoded)
    }
}

fn metadata(array: &OnPairArray) -> VortexResult<OnPairMetadata> {
    let bytes = <OnPair as VTable>::serialize(array.as_view(), &SESSION)?
        .ok_or_else(|| vortex_error::vortex_err!("OnPair metadata must be present"))?;
    Ok(OnPairMetadata::decode(bytes.as_slice())?)
}

fn deserialize(source: &OnPairArray, children: &[ArrayRef]) -> VortexResult<OnPairArray> {
    let metadata = metadata(source)?.encode_to_vec();
    let buffers = [source.dict_bytes_handle().clone()];
    let parts = <OnPair as VTable>::deserialize(
        &OnPair,
        source.dtype(),
        source.len(),
        &metadata,
        &buffers,
        &children,
        &SESSION,
    )?;
    Array::<OnPair>::try_from_parts(parts)
}

fn serde_roundtrip(array: &ArrayRef) -> VortexResult<ArrayRef> {
    let context = ArrayContext::empty();
    let bytes = array
        .serialize(&context, &SESSION, &SerializeOptions::default())?
        .into_iter()
        .flatten()
        .collect::<BufferMut<u8>>()
        .freeze();
    SerializedArray::try_from(bytes)?.decode(
        array.dtype(),
        array.len(),
        &ReadContext::new(context.to_ids()),
        &SESSION,
    )
}

#[test]
fn index_set_rejects_unknown_bits() {
    assert!(OnPairIndexSet::from_bits(1 << 63).is_err());
}

#[cfg_attr(miri, ignore)]
#[test]
fn frequency_index_is_opt_in() -> VortexResult<()> {
    let input = input(false);
    let plain = encode(&input, OnPairIndexSet::empty())?;
    let indexed = encode(&input, FREQUENCY_INDEX)?;

    assert!(plain.token_frequency_index_child().is_none());
    assert_eq!(metadata(&plain)?.index_flags, 0);
    assert_eq!(
        indexed.token_frequency_index_child().unwrap().dtype(),
        &DType::Primitive(PType::U32, Nullability::NonNullable)
    );
    assert_eq!(metadata(&indexed)?.index_flags, FREQUENCY_INDEX.bits());

    let mut ctx = SESSION.create_execution_ctx();
    let frequencies = indexed.token_frequency_index(&mut ctx)?.unwrap();
    assert_eq!(frequencies.num_tokens() + 1, indexed.dict_offsets().len());
    assert_eq!(
        frequencies.cumulative().last().copied(),
        Some(u32::try_from(indexed.codes().len())?)
    );
    Ok(())
}

#[cfg_attr(miri, ignore)]
#[test]
fn indexes_and_validity_round_trip() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    for (input, indexes) in [
        (input(true), OnPairIndexSet::empty()),
        (input(false), FREQUENCY_INDEX),
        (input(true), FREQUENCY_INDEX),
    ] {
        let encoded = encode(&input, indexes)?.into_array();
        let decoded = serde_roundtrip(&encoded)?
            .try_downcast::<OnPair>()
            .map_err(|array| {
                vortex_error::vortex_err!("expected OnPair, got {}", array.encoding_id())
            })?;

        assert_eq!(
            decoded.token_frequency_index_child().is_some(),
            indexes.has_token_frequency()
        );
        assert_arrays_eq!(decoded.into_array(), input, &mut ctx);
    }
    Ok(())
}

#[cfg_attr(miri, ignore)]
#[test]
fn extra_children_are_rejected() -> VortexResult<()> {
    let indexed = encode(&input(false), FREQUENCY_INDEX)?;
    let mut children = indexed.as_ref().children();
    children.extend([
        Buffer::from(vec![1u8]).into_array(),
        Buffer::from(vec![2u8]).into_array(),
    ]);

    assert!(deserialize(&indexed, &children).is_err());
    Ok(())
}

#[cfg_attr(miri, ignore)]
#[test]
fn stored_index_is_validated_on_first_use() -> VortexResult<()> {
    let indexed = encode(&input(false), FREQUENCY_INDEX)?;
    let mut children = indexed.as_ref().children();
    let mut cumulative = indexed
        .token_frequency_index(&mut SESSION.create_execution_ctx())?
        .unwrap()
        .cumulative()
        .to_vec();
    *cumulative.last_mut().unwrap() -= 1;
    children[OnPairSlots::VALIDITY] = Buffer::from(cumulative).into_array();

    let decoded = deserialize(&indexed, &children)?;
    let original = decoded.clone().into_array();
    let decoded = build_token_frequency_index(decoded, &mut SESSION.create_execution_ctx())?;
    assert!(ArrayRef::ptr_eq(&original, decoded.as_ref()));
    assert!(
        decoded
            .token_frequency_index(&mut SESSION.create_execution_ctx())
            .is_err()
    );
    Ok(())
}
