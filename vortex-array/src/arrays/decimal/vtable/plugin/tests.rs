// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reads of hand-built `vortex.decimal` nodes through [`DecimalPlugin`].
//!
//! Each node is written as raw bytes with hand-written [`DecimalMetadata`], so no test builds a
//! decimal array at a non-canonical width. A node lists the raw value of every slot as `i128`,
//! including nonzero values in null slots, and its validity is `None` for a non-nullable node.

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;
use std::sync::LazyLock;

use futures::future::BoxFuture;
use prost::Message;
use rstest::rstest;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use super::DecimalPlugin;
use crate::ArrayContext;
use crate::ArrayRef;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array::ArrayDeserialization;
use crate::array::ArrayPlugin;
use crate::array::VTable;
use crate::arrays::BoolArray;
use crate::arrays::Decimal;
use crate::arrays::DecimalArray;
use crate::arrays::DictArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::Widen;
use crate::arrays::WidenArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::arrays::decimal::vtable::DecimalMetadata;
use crate::arrays::widen::WidenArraySlotsExt;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::buffer::DeviceBuffer;
use crate::dtype::BigCast;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::match_each_decimal_value_type;
use crate::serde::SerializeOptions;
use crate::serde::SerializedArray;
use crate::validity::Validity;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// An array read from a node, with the address of the buffer of the node to check its reuse.
struct ReadNode {
    array: ArrayRef,
    stored_ptr: *const u8,
}

/// Reads a node that stores `slots` as `values_type` integers, with the dtype `decimal_dtype`.
fn read_node(
    values_type: DecimalType,
    decimal_dtype: DecimalDType,
    slots: &[i128],
    validity: Option<&[bool]>,
) -> VortexResult<ReadNode> {
    let values = stored_values(values_type, slots);
    let stored_ptr = values.as_host().as_ptr();
    let (nullability, children) = match validity {
        Some(validity) => (
            Nullability::Nullable,
            vec![BoolArray::from_iter(validity.iter().copied()).into_array()],
        ),
        None => (Nullability::NonNullable, Vec::new()),
    };

    let array = deserialize(
        &DType::Decimal(decimal_dtype, nullability),
        slots.len(),
        &metadata(values_type),
        &[values],
        &children,
    )?;

    Ok(ReadNode { array, stored_ptr })
}

/// Deserializes the parts of a node through [`DecimalPlugin`].
fn deserialize(
    dtype: &DType,
    len: usize,
    metadata: &[u8],
    buffers: &[BufferHandle],
    children: &[ArrayRef],
) -> VortexResult<ArrayRef> {
    let id = DecimalPlugin.id();
    let parts = ArrayDeserialization::new(id, dtype, len, metadata, buffers, &children);

    DecimalPlugin.deserialize(parts, &SESSION)
}

fn metadata(values_type: DecimalType) -> Vec<u8> {
    DecimalMetadata {
        values_type: values_type as i32,
    }
    .encode_to_vec()
}

/// Returns `slots` as a host buffer of `values_type` integers.
fn stored_values(values_type: DecimalType, slots: &[i128]) -> BufferHandle {
    let bytes = match_each_decimal_value_type!(values_type, |T| {
        slots
            .iter()
            .map(|&slot| <T as BigCast>::from(slot).vortex_expect("a slot fits its stored width"))
            .collect::<Buffer<T>>()
            .into_byte_buffer()
    });

    BufferHandle::new_host(bytes)
}

/// Returns the canonical array of the valid values of a node, with zeros in its null slots.
fn canonical_reference(
    decimal_dtype: DecimalDType,
    slots: &[i128],
    validity: Option<&[bool]>,
) -> VortexResult<DecimalArray> {
    let valid_slots = slots
        .iter()
        .enumerate()
        .map(|(row, &slot)| match validity {
            Some(validity) if !validity[row] => 0,
            _ => slot,
        })
        .collect::<Vec<_>>();
    let validity = match validity {
        Some(validity) => Validity::from_iter(validity.iter().copied()),
        None => Validity::NonNullable,
    };

    let canonical_type = decimal_dtype.canonical_values_type();

    DecimalArray::try_new_handle(
        stored_values(canonical_type, &valid_slots),
        canonical_type,
        decimal_dtype,
        validity,
    )
}

/// Returns the decimal leaf that holds the values of `array`: the child of a `Widen`, or `array`.
fn decimal_leaf(array: &ArrayRef) -> DecimalArray {
    let leaf = match array.as_opt::<Widen>() {
        Some(widen) => widen.values().clone(),
        None => array.clone(),
    };

    leaf.as_::<Decimal>().into_owned()
}

/// Checks the encoding, the leaf dtype and width, the reuse of the stored buffer and the values.
///
/// The cases cover reads at the canonical width, old nodes narrower and wider than it, and the
/// child precision rule of [`WidenArray::from_stored_decimal`]: the min and the max both count,
/// and a positive scale is the smallest precision.
///
/// [`WidenArray::from_stored_decimal`]: crate::arrays::WidenArray::from_stored_decimal
#[rstest]
#[case::canonical_reuse(
    DecimalType::I32, DecimalDType::new(9, 2),
    vec![-999_999_999, 0, 999_999_999], None,
    None, true,
)]
#[case::narrow_reuse(
    DecimalType::I16, DecimalDType::new(18, 2),
    vec![-9_999, 0, 9_999], None,
    Some(4), true,
)]
#[case::gap_gives_i32_child(
    DecimalType::I16, DecimalDType::new(18, 2),
    vec![0, 32_767], None,
    Some(5), false,
)]
#[case::narrow_gives_i8_child(
    DecimalType::I32, DecimalDType::new(18, 2),
    vec![-99, 0, 99], None,
    Some(2), false,
)]
#[case::gap_to_canonical_i16(
    DecimalType::I8, DecimalDType::new(4, 2),
    vec![-99, 100, 127], None,
    None, false,
)]
#[case::scale_bound_to_canonical(
    DecimalType::I8, DecimalDType::new(18, 10),
    vec![-99, 0, 99], None,
    None, false,
)]
#[case::negative_minimum(
    DecimalType::I8, DecimalDType::new(10, 0),
    vec![-128, 0, 1], None,
    Some(3), false,
)]
#[case::p40_i128_child(
    DecimalType::I128, DecimalDType::new(40, 2),
    vec![1 - 10i128.pow(38), 0, 10i128.pow(38) - 1], None,
    Some(38), true,
)]
#[case::p40_needs_39_digits(
    DecimalType::I128, DecimalDType::new(40, 0),
    vec![0, i128::MAX], None,
    None, false,
)]
#[case::wider_to_canonical(
    DecimalType::I128, DecimalDType::new(5, 2),
    vec![-99_999, 0, 99_999], None,
    None, false,
)]
#[case::wider_to_widen(
    DecimalType::I128, DecimalDType::new(18, 2),
    vec![-99, 0, 99], None,
    Some(2), false,
)]
#[case::nullable_reuse(
    DecimalType::I16, DecimalDType::new(18, 2),
    vec![-9_999, 32_767, 9_999], Some(vec![true, false, true]),
    Some(4), true,
)]
#[case::all_null(
    DecimalType::I32, DecimalDType::new(18, 2),
    vec![5, -5], Some(vec![false, false]),
    Some(2), false,
)]
#[case::empty(
    DecimalType::I32, DecimalDType::new(18, 2),
    vec![], None,
    Some(2), false,
)]
#[case::negative_scale(
    DecimalType::I32, DecimalDType::new(18, -3),
    vec![-99, 0, 99], None,
    Some(2), false,
)]
fn reads_node_at_any_width(
    #[case] values_type: DecimalType,
    #[case] decimal_dtype: DecimalDType,
    #[case] slots: Vec<i128>,
    #[case] validity: Option<Vec<bool>>,
    #[case] child_precision: Option<u8>,
    #[case] reuses_buffer: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();

    let node = read_node(values_type, decimal_dtype, &slots, validity.as_deref())?;

    assert_eq!(node.array.is::<Widen>(), child_precision.is_some());
    let leaf = decimal_leaf(&node.array);
    let leaf_dtype = match child_precision {
        Some(precision) => DecimalDType::new(precision, decimal_dtype.scale()),
        None => decimal_dtype,
    };
    assert_eq!(leaf.decimal_dtype(), leaf_dtype);
    assert_eq!(leaf.values_type(), leaf_dtype.canonical_values_type());

    // A leaf at another width than the stored width is a copy. Two empty buffers can share an
    // address, so only a reuse is checked by address.
    assert_eq!(leaf.values_type() == values_type, reuses_buffer);
    if reuses_buffer {
        assert_eq!(leaf.buffer_handle().as_host().as_ptr(), node.stored_ptr);
    }

    assert_arrays_eq!(
        node.array,
        canonical_reference(decimal_dtype, &slots, validity.as_deref())?,
        &mut ctx
    );

    Ok(())
}

/// A node at the canonical width is trusted, so a valid value outside the precision is no error.
#[test]
fn canonical_node_trusts_precision() -> VortexResult<()> {
    let node = read_node(DecimalType::I8, DecimalDType::new(2, 0), &[127], None)?;

    let leaf = node.array.as_::<Decimal>();
    assert_eq!(leaf.buffer_handle().as_host().as_ptr(), node.stored_ptr);
    assert_eq!(leaf.buffer::<i8>().as_slice(), [127]);

    Ok(())
}

#[test]
fn wider_node_rejects_value_outside_precision() {
    let result = read_node(
        DecimalType::I32,
        DecimalDType::new(4, 2),
        &[1, 20_000],
        None,
    );

    assert!(result.is_err_and(|error| error.to_string().contains("20000")));
}

/// A null slot does not count for the child precision, and the reused buffer keeps its value.
#[test]
fn narrower_node_ignores_null_slots() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let slots = [9_999, 32_767];
    let validity = [true, false];

    let node = read_node(
        DecimalType::I16,
        DecimalDType::new(18, 2),
        &slots,
        Some(&validity),
    )?;

    assert!(node.array.is::<Widen>());
    let leaf = decimal_leaf(&node.array);
    assert_eq!(leaf.decimal_dtype(), DecimalDType::new(4, 2));
    assert_eq!(leaf.buffer_handle().as_host().as_ptr(), node.stored_ptr);
    assert_eq!(leaf.buffer::<i16>().as_slice(), [9_999, 32_767]);
    assert_arrays_eq!(
        node.array,
        canonical_reference(DecimalDType::new(18, 2), &slots, Some(&validity))?,
        &mut ctx
    );

    Ok(())
}

#[test]
fn narrowing_copy_zeroes_null_slots() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let slots: [i128; 3] = [99, i32::MAX.into(), -99];
    let validity = [true, false, true];

    let node = read_node(
        DecimalType::I32,
        DecimalDType::new(18, 2),
        &slots,
        Some(&validity),
    )?;

    assert!(node.array.is::<Widen>());
    let leaf = decimal_leaf(&node.array);
    assert_eq!(leaf.decimal_dtype(), DecimalDType::new(2, 2));
    assert_eq!(leaf.buffer::<i8>().as_slice(), [99, 0, -99]);
    assert_arrays_eq!(
        node.array,
        canonical_reference(DecimalDType::new(18, 2), &slots, Some(&validity))?,
        &mut ctx
    );

    Ok(())
}

#[test]
fn unknown_values_type_fails() {
    let metadata = DecimalMetadata { values_type: 6 }.encode_to_vec();
    let dtype = DType::Decimal(DecimalDType::new(18, 2), Nullability::NonNullable);

    let result = deserialize(
        &dtype,
        1,
        &metadata,
        &[stored_values(DecimalType::I64, &[1])],
        &[],
    );

    assert!(result.is_err_and(|error| error.to_string().contains("got 6")));
}

#[test]
fn short_buffer_fails() {
    let dtype = DType::Decimal(DecimalDType::new(18, 2), Nullability::NonNullable);
    let values = stored_values(DecimalType::I32, &[1, 2, 3]);

    let result = deserialize(&dtype, 4, &metadata(DecimalType::I32), &[values], &[]);

    assert!(result.is_err_and(|error| error.to_string().contains("must have 16 bytes, got 12")));
}

/// The byte length of the node does not fit in `usize`, so it must not wrap to the empty buffer.
#[test]
fn overflowing_length_fails() {
    let dtype = DType::Decimal(DecimalDType::new(18, 2), Nullability::NonNullable);
    let values = stored_values(DecimalType::I16, &[]);

    let result = deserialize(
        &dtype,
        usize::MAX / 2 + 1,
        &metadata(DecimalType::I16),
        &[values],
        &[],
    );

    assert!(result.is_err_and(|error| error.to_string().contains("fits in usize")));
}

/// A buffer of the right length that is aligned only to bytes cannot be read as `i32` values.
#[test]
fn misaligned_buffer_fails() {
    let dtype = DType::Decimal(DecimalDType::new(9, 2), Nullability::NonNullable);
    let values = BufferHandle::new_host(ByteBuffer::copy_from([0u8; 8]));

    let result = deserialize(&dtype, 2, &metadata(DecimalType::I32), &[values], &[]);

    assert!(result.is_err_and(|error| error.to_string().contains("alignment")));
}

#[test]
fn non_decimal_dtype_fails() {
    let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
    let values = stored_values(DecimalType::I32, &[1]);

    let result = deserialize(&dtype, 1, &metadata(DecimalType::I32), &[values], &[]);

    assert!(result.is_err_and(|error| error.to_string().contains("requires a decimal dtype")));
}

#[test]
fn extra_validity_child_fails() {
    let dtype = DType::Decimal(DecimalDType::new(18, 2), Nullability::Nullable);
    let values = stored_values(DecimalType::I32, &[1, 2]);
    let validity = BoolArray::from_iter([true, false]).into_array();

    let result = deserialize(
        &dtype,
        2,
        &metadata(DecimalType::I32),
        &[values],
        &[validity.clone(), validity],
    );

    assert!(result.is_err_and(|error| error.to_string().contains("got 2")));
}

/// A validity child would make the result nullable, and the deserializer asserts the dtype.
#[test]
fn validity_child_of_non_nullable_node_fails() {
    let dtype = DType::Decimal(DecimalDType::new(18, 2), Nullability::NonNullable);
    let values = stored_values(DecimalType::I32, &[1, 2]);
    let validity = BoolArray::from_iter([true, false]).into_array();

    let result = deserialize(
        &dtype,
        2,
        &metadata(DecimalType::I32),
        &[values],
        &[validity],
    );

    assert!(result.is_err_and(|error| error.to_string().contains("nullability")));
}

#[test]
fn decimal_vtable_rejects_old_node() {
    let dtype = DType::Decimal(DecimalDType::new(18, 2), Nullability::NonNullable);
    let buffers = [stored_values(DecimalType::I16, &[1, 2])];
    let children: Vec<ArrayRef> = Vec::new();

    let result = VTable::deserialize(
        &Decimal,
        &dtype,
        2,
        &metadata(DecimalType::I16),
        &buffers,
        &children,
        &SESSION,
    );

    assert!(result.is_err_and(|error| {
        let message = error.to_string();
        message.contains("DecimalPlugin") && message.contains("got i16")
    }));
}

/// Writes a canonical array of `written_dtype` and decodes it as `read_dtype`, which makes it an
/// old node of `read_dtype`.
///
/// This goes through the deserializer, which asserts that the plugin supports the encoding of the
/// result.
#[rstest]
#[case::narrower_node(
    DecimalDType::new(2, 0), DecimalDType::new(10, 0),
    [-99, 0, 99],
    true,
)]
#[case::wider_node(
    DecimalDType::new(38, 0), DecimalDType::new(5, 0),
    [-12_345, 0, 12_345],
    false,
)]
fn decode_through_serialized_array(
    #[case] written_dtype: DecimalDType,
    #[case] read_dtype: DecimalDType,
    #[case] values: [i128; 3],
    #[case] reads_widen: bool,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let written = canonical_reference(written_dtype, &values, None)?.into_array();
    let array_ctx = ArrayContext::empty();
    let serialized = written.serialize(&array_ctx, &SESSION, &SerializeOptions::default())?;
    let mut bytes = ByteBufferMut::empty();
    for buffer in serialized {
        bytes.extend_from_slice(buffer.as_ref());
    }

    let dtype = DType::Decimal(read_dtype, Nullability::NonNullable);
    let decoded = SerializedArray::try_from(bytes.freeze())?.decode(
        &dtype,
        values.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    )?;

    assert_eq!(decoded.is::<Widen>(), reads_widen);
    assert_arrays_eq!(
        decoded,
        canonical_reference(read_dtype, &values, None)?,
        &mut ctx
    );

    Ok(())
}

/// A device buffer that holds its bytes on the host. A copy to the host succeeds unless
/// `fails_host_copy` is set.
#[derive(Debug, PartialEq, Eq, Hash)]
struct TestDeviceBuffer {
    bytes: ByteBuffer,
    fails_host_copy: bool,
}

impl DeviceBuffer for TestDeviceBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.bytes.len()
    }

    fn alignment(&self) -> Alignment {
        self.bytes.alignment()
    }

    fn copy_to_host_sync(&self, alignment: Alignment) -> VortexResult<ByteBuffer> {
        if self.fails_host_copy {
            vortex_bail!("no host copy");
        }

        Ok(ByteBuffer::copy_from_aligned(
            self.bytes.as_slice(),
            alignment,
        ))
    }

    fn copy_to_host(
        &self,
        _alignment: Alignment,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ByteBuffer>>> {
        vortex_bail!("a decimal read copies device buffers synchronously")
    }

    fn slice(&self, range: Range<usize>) -> Arc<dyn DeviceBuffer> {
        Arc::new(Self {
            bytes: self.bytes.slice(range),
            fails_host_copy: self.fails_host_copy,
        })
    }

    fn aligned(self: Arc<Self>, alignment: Alignment) -> VortexResult<Arc<dyn DeviceBuffer>> {
        assert!(self.alignment().is_aligned_to(alignment));
        Ok(self)
    }
}

fn on_device(bytes: ByteBuffer) -> BufferHandle {
    BufferHandle::new_device(Arc::new(TestDeviceBuffer {
        bytes,
        fails_host_copy: false,
    }))
}

fn on_failing_device(bytes: ByteBuffer) -> BufferHandle {
    BufferHandle::new_device(Arc::new(TestDeviceBuffer {
        bytes,
        fails_host_copy: true,
    }))
}

/// A canonical node keeps its device buffer. An old node and its validity are read on the host.
#[test]
fn device_buffers_read_on_host() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = DType::Decimal(DecimalDType::new(18, 2), Nullability::Nullable);
    let validity = [true, false, true];
    let validity_bits = on_device(ByteBuffer::copy_from([0b101u8]));
    let validity_child =
        BoolArray::try_new_from_handle(validity_bits, 0, 3, Validity::NonNullable)?;
    let children = [validity_child.into_array()];

    let canonical_values = on_device(buffer![-99i64, 7, 99].into_byte_buffer());
    let canonical = deserialize(
        &dtype,
        3,
        &metadata(DecimalType::I64),
        &[canonical_values],
        &children,
    )?;
    assert!(canonical.as_::<Decimal>().buffer_handle().is_on_device());

    let old_values = on_device(buffer![-99i32, 7, 99].into_byte_buffer());
    let old = deserialize(
        &dtype,
        3,
        &metadata(DecimalType::I32),
        &[old_values],
        &children,
    )?;
    assert!(old.is_host());
    assert_eq!(decimal_leaf(&old).decimal_dtype(), DecimalDType::new(2, 2));
    assert_arrays_eq!(
        old,
        canonical_reference(DecimalDType::new(18, 2), &[-99, 7, 99], Some(&validity))?,
        &mut ctx
    );

    Ok(())
}

/// A device buffer one slot below the validity array is also read on the host.
#[test]
fn nested_device_validity_reads_on_host() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let dtype = DType::Decimal(DecimalDType::new(18, 2), Nullability::Nullable);
    let codes = PrimitiveArray::from_buffer_handle(
        on_device(buffer![0u8, 1, 0].into_byte_buffer()),
        PType::U8,
        Validity::NonNullable,
    );
    let validity_child = DictArray::try_new(
        codes.into_array(),
        BoolArray::from_iter([true, false]).into_array(),
    )?
    .into_array();
    assert!(!validity_child.is_host());

    let old = deserialize(
        &dtype,
        3,
        &metadata(DecimalType::I32),
        &[stored_values(DecimalType::I32, &[-99, 7, 99])],
        &[validity_child],
    )?;
    assert!(old.is_host());
    assert_eq!(decimal_leaf(&old).decimal_dtype(), DecimalDType::new(2, 2));
    assert_arrays_eq!(
        old,
        canonical_reference(
            DecimalDType::new(18, 2),
            &[-99, 7, 99],
            Some(&[true, false, true])
        )?,
        &mut ctx
    );

    Ok(())
}

/// An old node fails when its values or its validity cannot be copied to the host. A canonical
/// node copies nothing, so it reads even when no copy would succeed.
#[test]
fn failed_host_copy_fails() -> VortexResult<()> {
    let dtype = DType::Decimal(DecimalDType::new(18, 2), Nullability::Nullable);
    let device_validity = [BoolArray::try_new_from_handle(
        on_failing_device(ByteBuffer::copy_from([0b101u8])),
        0,
        3,
        Validity::NonNullable,
    )?
    .into_array()];
    let host_validity = BoolArray::from_iter([true, false, true]).into_array();

    let error = deserialize(
        &dtype,
        3,
        &metadata(DecimalType::I32),
        &[stored_values(DecimalType::I32, &[-99, 7, 99])],
        &device_validity,
    )
    .err()
    .vortex_expect("an old node with device validity that cannot be copied fails");
    assert!(error.to_string().contains("no host copy"));

    let error = deserialize(
        &dtype,
        3,
        &metadata(DecimalType::I32),
        &[on_failing_device(buffer![-99i32, 7, 99].into_byte_buffer())],
        &[host_validity],
    )
    .err()
    .vortex_expect("an old node with device values that cannot be copied fails");
    assert!(error.to_string().contains("no host copy"));

    let canonical = deserialize(
        &dtype,
        3,
        &metadata(DecimalType::I64),
        &[on_failing_device(buffer![-99i64, 7, 99].into_byte_buffer())],
        &device_validity,
    )?;
    assert!(canonical.as_::<Decimal>().buffer_handle().is_on_device());

    Ok(())
}

#[test]
fn decimal_widen_round_trip_reads_as_widen() -> VortexResult<()> {
    let values = DecimalArray::from_option_iter(
        [Some(-999_999_999i32), None, Some(42)],
        DecimalDType::new(9, 2),
    );
    let dtype = DType::Decimal(DecimalDType::new(38, 2), Nullability::Nullable);
    let written = WidenArray::try_new(values.into_array(), dtype)?.into_array();
    let array_ctx = ArrayContext::empty();
    let mut bytes = ByteBufferMut::empty();
    for buffer in written.serialize(&array_ctx, &SESSION, &SerializeOptions::default())? {
        bytes.extend_from_slice(buffer.as_ref());
    }
    let serialized = SerializedArray::try_from(bytes.freeze())?;
    let decoded = serialized.decode(
        written.dtype(),
        written.len(),
        &ReadContext::new(array_ctx.to_ids()),
        &SESSION,
    )?;

    assert!(decoded.is::<Widen>());
    assert_arrays_eq!(decoded, written, &mut SESSION.create_execution_ctx());

    Ok(())
}
