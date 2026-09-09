// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Serialization of decimal byte parts under the frozen and v2 format IDs.

use prost::Message as _;
use vortex_array::Array;
use vortex_array::ArrayDeserialization;
use vortex_array::ArrayId;
use vortex_array::ArrayPlugin;
use vortex_array::ArrayRef;
use vortex_array::ArraySerialization;
use vortex_array::IntoArray;
use vortex_array::vtable::VTable;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::DecimalByteParts;
use super::DecimalBytePartsArraySlotsExt;
use super::DecimalBytesPartsMetadata;

/// The `vortex.decimal_byte_parts_v2` serialized format ID: byte parts carrying lower parts.
///
/// This is a serialized format, not a second in-memory encoding. `vortex.decimal_byte_parts`
/// froze promising a single child, so an array with lower parts serializes under this ID
/// instead, and both IDs deserialize back into the same [`crate::DecimalBytePartsArray`]. A reader
/// that predates lower parts fails on this ID with an unknown-encoding error rather than
/// misreading the children.
pub fn decimal_byte_parts_v2_id() -> ArrayId {
    static ID: CachedId = CachedId::new("vortex.decimal_byte_parts_v2");
    *ID
}

/// The [`ArrayPlugin`] for [`DecimalByteParts`], owning both of its serialized formats.
///
/// An array without lower parts serializes under the frozen `vortex.decimal_byte_parts` ID,
/// byte-identical to files written before lower parts existed. An array carrying lower parts
/// serializes under [`decimal_byte_parts_v2_id`]. Reading holds each ID to its own contract:
/// the frozen ID carries no lower parts and the v2 ID carries at least one, so recognizing the
/// newer format never widens what the frozen one may mean.
///
/// Register this plugin, or call [`crate::initialize`], to enable both formats. Registering
/// [`DecimalByteParts`] directly only supports the frozen format.
#[derive(Clone, Debug)]
pub struct DecimalBytePartsPlugin;

impl ArrayPlugin for DecimalBytePartsPlugin {
    fn id(&self) -> ArrayId {
        VTable::id(&DecimalByteParts)
    }

    fn serialized_ids(&self) -> Vec<ArrayId> {
        vec![VTable::id(&DecimalByteParts), decimal_byte_parts_v2_id()]
    }

    fn serialize(
        &self,
        array: &ArrayRef,
        _session: &VortexSession,
    ) -> VortexResult<Option<ArraySerialization>> {
        let view = array.as_opt::<DecimalByteParts>().ok_or_else(|| {
            vortex_err!(
                "DecimalByteParts plugin cannot serialize {}",
                array.encoding_id()
            )
        })?;
        let serialized_id = if view.lower_parts().is_empty() {
            VTable::id(&DecimalByteParts)
        } else {
            decimal_byte_parts_v2_id()
        };
        Ok(Some(ArraySerialization::from_array(
            serialized_id,
            array,
            DecimalBytesPartsMetadata::from_array(view)?.encode_to_vec(),
        )))
    }

    fn deserialize(
        &self,
        parts: ArrayDeserialization<'_>,
        _session: &VortexSession,
    ) -> VortexResult<ArrayRef> {
        let metadata = DecimalBytesPartsMetadata::decode(parts.metadata)?;
        let lower_part_count = metadata.lower_part_count()?;
        if parts.serialized_id == decimal_byte_parts_v2_id() {
            vortex_ensure!(
                lower_part_count > 0,
                "{} must carry at least one lower part",
                parts.serialized_id
            );
        } else {
            vortex_ensure!(
                parts.serialized_id == VTable::id(&DecimalByteParts),
                "DecimalByteParts plugin does not recognize serialized ID {}",
                parts.serialized_id
            );
            vortex_ensure!(
                lower_part_count == 0,
                "{} must not carry lower parts, got {lower_part_count}",
                parts.serialized_id
            );
        }
        Ok(Array::try_from_parts(metadata.into_array_parts(
            parts.dtype,
            parts.len,
            parts.children,
        )?)?
        .into_array())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayContext;
    use vortex_array::ArrayParts;
    use vortex_array::ArraySlots;
    use vortex_array::ArrayVTable;
    use vortex_array::VortexSessionExecute;
    use vortex_array::array_session;
    use vortex_array::arrays::DecimalArray;
    use vortex_array::arrays::Primitive;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::DecimalDType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::i256;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::session::ArraySessionExt;
    use vortex_array::validity::Validity;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_session::registry::ReadContext;

    use super::*;
    use crate::DecimalBytePartsArray;
    use crate::DecimalBytePartsData;
    use crate::decimal_byte_parts::testing::encode;
    use crate::decimal_byte_parts::testing::i128_parts;
    use crate::decimal_byte_parts::testing::i256_parts;
    use crate::decimal_byte_parts::testing::wide_i128_values;
    use crate::decimal_byte_parts::testing::wide_i256_values;

    #[rstest]
    #[case::one_lower_part(i128_parts(wide_i128_values(), Validity::NonNullable))]
    #[case::three_lower_parts(i256_parts(wide_i256_values(), Validity::NonNullable))]
    #[case::nullable_three_lower_parts(i256_parts(wide_i256_values(), Validity::AllValid))]
    fn test_serde_round_trip_with_lower_parts(
        #[case] array: DecimalBytePartsArray,
    ) -> VortexResult<()> {
        test_serde_round_trip(array)
    }

    #[rstest]
    #[case::no_lower_parts(
        encode(&DecimalArray::new(buffer![1i32, 2, 3], DecimalDType::new(9, 2), Validity::NonNullable))
            .vortex_expect("valid decimal byte parts")
    )]
    fn test_serde_round_trip_flat(#[case] array: DecimalBytePartsArray) -> VortexResult<()> {
        test_serde_round_trip(array)
    }

    #[rstest]
    fn test_deserialize_frozen_with_wider_storage(
        #[values(Validity::NonNullable, Validity::from_iter([true, false, true]))]
        validity: Validity,
    ) -> VortexResult<()> {
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let decimal_dtype = DecimalDType::new(2, 0);
        let expected = DecimalArray::new(buffer![1i8, 2, 3], decimal_dtype, validity.clone());
        let children = vec![PrimitiveArray::new(buffer![1i64, 2, 3], validity).into_array()];

        // Metadata emitted by the frozen serializer for a single i64 child.
        let decoded = DecimalBytePartsPlugin.deserialize(
            ArrayDeserialization::new(
                VTable::id(&DecimalByteParts),
                expected.dtype(),
                expected.len(),
                &[8, 7],
                &[],
                &children,
            ),
            &session,
        )?;
        assert_arrays_eq!(expected, decoded, &mut ctx);
        test_serde_round_trip(decoded.as_::<DecimalByteParts>().into_owned())
    }

    #[rstest]
    #[case::i64(DecimalArray::new(
        buffer![-99i64, 0, 99], DecimalDType::new(2, 0), Validity::NonNullable,
    ))]
    #[case::i128(DecimalArray::new(
        buffer![-99i128, 0, 99], DecimalDType::new(2, 0), Validity::NonNullable,
    ))]
    #[case::i256(DecimalArray::new(
        buffer![i256::from_i128(-99), i256::ZERO, i256::from_i128(99)],
        DecimalDType::new(2, 0), Validity::NonNullable,
    ))]
    fn test_serde_round_trip_wider_storage(#[case] decimal: DecimalArray) -> VortexResult<()> {
        let mut ctx = array_session().create_execution_ctx();
        let encoded = encode(&decimal)?;
        assert_arrays_eq!(decimal, encoded, &mut ctx);
        assert_eq!(
            encoded.execute_scalar(0, &mut ctx)?,
            decimal.execute_scalar(0, &mut ctx)?,
        );
        test_serde_round_trip(encoded)
    }

    fn test_serde_round_trip(array: DecimalBytePartsArray) -> VortexResult<()> {
        let session = array_session();
        // Both serialized formats must be registered: an array with lower parts comes back
        // under the v2 format id.
        crate::initialize(&session);

        let array = array.into_array();
        let dtype = array.dtype().clone();
        let len = array.len();
        let lower_part_count = array
            .as_opt::<DecimalByteParts>()
            .vortex_expect("byte parts array")
            .lower_parts()
            .len();

        let expected_id = if lower_part_count == 0 {
            VTable::id(&DecimalByteParts)
        } else {
            decimal_byte_parts_v2_id()
        };
        assert_eq!(
            session
                .array_serialize(&array)?
                .vortex_expect("byte parts arrays are serializable")
                .serialized_id,
            expected_id
        );

        let array_ctx = ArrayContext::empty();
        let serialized = array.serialize(&array_ctx, &session, &SerializeOptions::default())?;
        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let parts = SerializedArray::try_from(concat.freeze())?;
        let decoded = parts.decode(&dtype, len, &ReadContext::new(array_ctx.to_ids()), &session)?;

        assert_eq!(
            decoded
                .as_opt::<DecimalByteParts>()
                .vortex_expect("byte parts array")
                .lower_parts()
                .len(),
            lower_part_count,
            "lower parts must survive serde"
        );

        let mut ctx = session.create_execution_ctx();
        assert_arrays_eq!(array, decoded, &mut ctx);
        Ok(())
    }

    fn deserialize_with(
        lower_part_count: u32,
        children: Vec<ArrayRef>,
    ) -> VortexResult<DecimalBytePartsArray> {
        let serialized_id = if lower_part_count == 0 {
            VTable::id(&DecimalByteParts)
        } else {
            decimal_byte_parts_v2_id()
        };
        plugin_deserialize_with(serialized_id, lower_part_count, children)
            .map(|array| array.as_::<DecimalByteParts>().into_owned())
    }

    #[test]
    fn test_deserialize_reads_lower_parts() -> VortexResult<()> {
        let array = deserialize_with(1, vec![msp(), lower_part()])?;
        assert_eq!(array.lower_parts().len(), 1);

        let mut ctx = array_session().create_execution_ctx();
        let canonical = array.into_array().execute::<DecimalArray>(&mut ctx)?;
        assert_eq!(
            canonical.buffer::<i128>().as_slice(),
            &[(1i128 << 64) | 1, (2i128 << 64) | 2, (3i128 << 64) | 3]
        );
        Ok(())
    }

    /// An array read from a file can be handed straight back to a writer, bypassing both the
    /// constructor and the compressor. Its serialized id must still be the v2 format, so a
    /// writer whose permitted encodings predate the v2 format refuses it.
    #[test]
    fn read_lower_parts_serialize_under_the_wide_format() -> VortexResult<()> {
        let session = array_session();
        crate::initialize(&session);

        let array = deserialize_with(1, vec![msp(), lower_part()])?.into_array();

        let serialization = session
            .array_serialize(&array)?
            .vortex_expect("byte parts arrays are serializable");
        assert_eq!(serialization.serialized_id, decimal_byte_parts_v2_id());

        let restricted = ArrayContext::empty()
            .with_allowed_ids([VTable::id(&DecimalByteParts)].into_iter().collect());
        let err = array
            .serialize(&restricted, &session, &SerializeOptions::default())
            .expect_err("expected the permitted-encoding check to refuse the v2 format");
        assert!(
            err.to_string().contains("not permitted"),
            "error should name the permitted-encoding check, got: {err}"
        );
        Ok(())
    }

    /// Reading back an array that already carries lower parts, and computing over it, must
    /// always work: the v2 format only restricts which writers may emit it. If reading or
    /// the rebuild that every compute kernel does were blocked, a session whose editions
    /// predate the v2 format could not read a file written by one that includes it.
    #[test]
    fn compute_over_existing_lower_parts_is_not_gated() -> VortexResult<()> {
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();

        // Stands in for an array materialized from a file: the parts already exist.
        let array = deserialize_with(1, vec![msp(), lower_part()])?.into_array();

        let sliced = array.slice(0..2)?;
        assert_eq!(sliced.execute::<DecimalArray>(&mut ctx)?.len(), 2);
        Ok(())
    }

    #[rstest]
    fn test_deserialize_redundant_lower_parts(
        #[values(2, 3)] lower_part_count: u32,
    ) -> VortexResult<()> {
        let mut children = vec![buffer![0i64; 3].into_array()];
        children.extend((1..lower_part_count).map(|_| buffer![0u64; 3].into_array()));
        children.push(lower_part());
        let array = deserialize_with(lower_part_count, children)?;
        let expected = DecimalArray::new(
            buffer![1i128, 2, 3],
            DecimalDType::new(38, 2),
            Validity::NonNullable,
        );
        let mut ctx = array_session().create_execution_ctx();
        assert_arrays_eq!(expected, array, &mut ctx);
        test_serde_round_trip(array)
    }

    #[test]
    fn test_deserialize_rejects_child_count_mismatch() {
        // Metadata claiming a lower part that was not serialized.
        assert!(deserialize_with(1, vec![msp()]).is_err());
        // Metadata claiming fewer lower parts than there are children.
        assert!(deserialize_with(0, vec![msp(), lower_part()]).is_err());
        // Metadata claiming more lower parts than the encoding supports.
        assert!(
            deserialize_with(
                4,
                vec![
                    msp(),
                    lower_part(),
                    lower_part(),
                    lower_part(),
                    lower_part()
                ]
            )
            .is_err()
        );
    }

    fn plugin_deserialize_with(
        serialized_id: ArrayId,
        lower_part_count: u32,
        children: Vec<ArrayRef>,
    ) -> VortexResult<ArrayRef> {
        let metadata = DecimalBytesPartsMetadata {
            zeroth_child_ptype: PType::I64 as i32,
            lower_part_count,
        }
        .encode_to_vec();
        let dtype = DType::Decimal(DecimalDType::new(38, 2), Nullability::NonNullable);
        DecimalBytePartsPlugin.deserialize(
            ArrayDeserialization::new(serialized_id, &dtype, 3, &metadata, &[], &children),
            &array_session(),
        )
    }

    /// Each serialized ID keeps its own contract: the frozen ID never carries lower parts, and
    /// the v2 ID is never written without them.
    #[rstest]
    #[case::frozen_without_lower_parts(VTable::id(&DecimalByteParts), 0, vec![msp()], true)]
    #[case::frozen_with_lower_parts(
        VTable::id(&DecimalByteParts),
        1,
        vec![msp(), lower_part()],
        false
    )]
    #[case::v2_with_lower_parts(decimal_byte_parts_v2_id(), 1, vec![msp(), lower_part()], true)]
    #[case::v2_without_lower_parts(decimal_byte_parts_v2_id(), 0, vec![msp()], false)]
    #[case::unknown_id(ArrayVTable::id(&Primitive), 0, vec![msp()], false)]
    fn plugin_holds_each_id_to_its_contract(
        #[case] serialized_id: ArrayId,
        #[case] lower_part_count: u32,
        #[case] children: Vec<ArrayRef>,
        #[case] accepted: bool,
    ) {
        let result = plugin_deserialize_with(serialized_id, lower_part_count, children);
        assert_eq!(result.is_ok(), accepted, "{serialized_id}: {result:?}");
    }

    fn msp() -> ArrayRef {
        buffer![1i64, 2, 3].into_array()
    }

    fn lower_part() -> ArrayRef {
        buffer![1u64, 2, 3].into_array()
    }

    fn session() -> VortexSession {
        let session = array_session();
        crate::initialize(&session);
        session
    }

    /// The wire ID the session's plugin picks for `array`.
    fn serialized_id(session: &VortexSession, array: &ArrayRef) -> VortexResult<ArrayId> {
        Ok(session
            .array_serialize(array)?
            .ok_or_else(|| vortex_err!("byte parts arrays are serializable"))?
            .serialized_id)
    }

    /// A single-child array is the stable shape and is always constructible.
    #[test]
    fn single_child_is_always_allowed() {
        assert!(DecimalByteParts::try_new(msp(), DecimalDType::new(19, 2)).is_ok());
        assert!(
            DecimalByteParts::try_new_with_lower_parts(msp(), vec![], DecimalDType::new(19, 2))
                .is_ok()
        );
    }

    /// Building lower parts in memory is always allowed — reading a file requires it. What
    /// changes is the serialized format, not what can be constructed.
    #[test]
    fn lower_parts_can_always_be_constructed() {
        assert!(
            DecimalByteParts::try_new_with_lower_parts(
                msp(),
                vec![lower_part()],
                DecimalDType::new(38, 2),
            )
            .is_ok()
        );
    }

    /// A single-child array keeps the frozen format id, byte-compatible with every reader since
    /// the format froze; lower parts move the array onto the v2 format id.
    #[test]
    fn serialized_id_tracks_lower_parts() -> VortexResult<()> {
        let session = session();

        let flat = DecimalByteParts::try_new(msp(), DecimalDType::new(19, 2))?.into_array();
        assert_eq!(
            serialized_id(&session, &flat)?,
            ArrayVTable::id(&DecimalByteParts)
        );

        let wide = DecimalByteParts::try_new_with_lower_parts(
            msp(),
            vec![lower_part()],
            DecimalDType::new(38, 2),
        )?
        .into_array();
        assert_eq!(serialized_id(&session, &wide)?, decimal_byte_parts_v2_id());

        Ok(())
    }

    /// The permitted-encoding check applies to the serialized id. A context restricted to the
    /// frozen format — a writer whose enabled editions predate the v2 format — must refuse an
    /// array carrying lower parts, however it was obtained.
    ///
    /// `ArrayParts` is public and `DecimalBytePartsData` is a public unit struct, so a caller can
    /// assemble slots by hand and go straight to `Array::try_from_parts`, bypassing
    /// `try_new_with_lower_parts` entirely. That back door is left open on purpose — it is the
    /// same path `deserialize` uses. What must hold is that the resulting array cannot become
    /// bytes under the frozen id.
    #[test]
    fn wide_format_is_refused_where_not_permitted() -> VortexResult<()> {
        let session = session();

        let mut slots = ArraySlots::with_capacity(2);
        slots.push(Some(msp()));
        slots.push(Some(lower_part()));

        // Assembling the array by hand succeeds: this is the shape a file read produces.
        let array = Array::try_from_parts(
            ArrayParts::new(
                DecimalByteParts,
                DType::Decimal(DecimalDType::new(38, 2), Nullability::NonNullable),
                3,
                DecimalBytePartsData,
            )
            .with_slots(slots),
        )?
        .into_array();
        assert_eq!(array.nchildren(), 2, "expected two limbs");

        // A context permitting only the frozen format refuses to write it.
        let restricted = ArrayContext::empty()
            .with_allowed_ids([ArrayVTable::id(&DecimalByteParts)].into_iter().collect());
        let err = array
            .serialize(&restricted, &session, &SerializeOptions::default())
            .expect_err("expected the permitted-encoding check to refuse the v2 format");
        assert!(
            err.to_string().contains("not permitted"),
            "error should name the permitted-encoding check, got: {err}"
        );

        // Permitting the v2 format id is exactly what allows the same array through.
        let permissive = ArrayContext::empty().with_allowed_ids(
            [
                ArrayVTable::id(&DecimalByteParts),
                decimal_byte_parts_v2_id(),
                ArrayVTable::id(&Primitive),
            ]
            .into_iter()
            .collect(),
        );
        let serialized = array.serialize(&permissive, &session, &SerializeOptions::default())?;
        assert!(!serialized.is_empty());
        assert!(
            permissive.to_ids().contains(&decimal_byte_parts_v2_id()),
            "the file's encoding table must carry the v2 format id"
        );

        Ok(())
    }

    #[test]
    fn bare_vtable_refuses_wide_serialization() -> VortexResult<()> {
        let session = array_session();
        session.arrays().register(DecimalByteParts);
        let array = DecimalByteParts::try_new_with_lower_parts(
            msp(),
            vec![lower_part()],
            DecimalDType::new(38, 2),
        )?
        .into_array();
        let restricted = ArrayContext::empty().with_allowed_ids(
            [
                ArrayVTable::id(&DecimalByteParts),
                ArrayVTable::id(&Primitive),
            ]
            .into_iter()
            .collect(),
        );

        assert!(
            array
                .serialize(&restricted, &session, &SerializeOptions::default())
                .is_err(),
            "bare VTable registration must not write lower parts under the frozen ID"
        );
        Ok(())
    }

    #[test]
    fn bare_vtable_refuses_lower_parts_on_frozen_id() -> VortexResult<()> {
        let session = array_session();
        session.arrays().register(DecimalByteParts);
        let array = DecimalByteParts::try_new_with_lower_parts(
            msp(),
            vec![lower_part()],
            DecimalDType::new(38, 2),
        )?
        .into_array();
        let id = ArrayVTable::id(&DecimalByteParts);
        let plugin = session
            .arrays()
            .registry()
            .get(&id)
            .ok_or_else(|| vortex_err!("missing decimal plugin"))?;
        let children = array.children();

        // i64 MSP and one lower part, mislabeled as the frozen format.
        let parts = ArrayDeserialization::new(
            id,
            array.dtype(),
            array.len(),
            &[8, 7, 16, 1],
            &[],
            &children,
        );
        assert!(plugin.deserialize(parts, &session).is_err());
        Ok(())
    }

    #[rstest]
    #[case::vtable(false)]
    #[case::plugin(true)]
    fn frozen_serde_is_compatible(#[case] use_plugin: bool) -> VortexResult<()> {
        let session = array_session();
        if use_plugin {
            session.arrays().register(DecimalBytePartsPlugin);
        } else {
            session.arrays().register(DecimalByteParts);
        }
        let array = DecimalByteParts::try_new(msp(), DecimalDType::new(2, 0))?.into_array();
        let serialized = session
            .array_serialize(&array)?
            .ok_or_else(|| vortex_err!("missing decimal serialization"))?;
        assert_eq!(serialized.serialized_id, ArrayVTable::id(&DecimalByteParts));
        assert_eq!(serialized.metadata, [8, 7]);
        let plugin = session
            .arrays()
            .registry()
            .get(&serialized.serialized_id)
            .ok_or_else(|| vortex_err!("missing decimal plugin"))?;
        let decoded = plugin.deserialize(
            ArrayDeserialization::new(
                serialized.serialized_id,
                array.dtype(),
                array.len(),
                &serialized.metadata,
                &[],
                &serialized.children,
            ),
            &session,
        )?;
        assert_arrays_eq!(array, decoded, &mut session.create_execution_ctx());
        Ok(())
    }
}
