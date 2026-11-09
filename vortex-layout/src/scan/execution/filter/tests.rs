// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;

use futures::future::BoxFuture;
use rstest::rstest;
use vortex_array::IntoArray;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Slice;
use vortex_array::buffer::BufferHandle;
use vortex_array::buffer::DeviceBuffer;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::vortex_bail;

use super::*;
use crate::test::SESSION;

fn assert_piece_result(array: ArrayRef, expected: Mask) -> VortexResult<()> {
    let reference = array
        .fill_null(false)?
        .execute::<Mask>(&mut SESSION.create_execution_ctx())?;
    assert_eq!(reference, expected);
    let len = array.len();
    // Piece rows number outputs; they are not the array's logical row range.
    let actual = piece_mask(
        &SESSION,
        Piece {
            rows: 41..42,
            array,
        },
    )?;
    assert_eq!(actual.len(), len);
    assert_eq!(actual, expected);
    Ok(())
}

#[rstest]
#[case(Scalar::from(true), true)]
#[case(Scalar::from(false), false)]
#[case(Scalar::from(Some(true)), true)]
#[case(Scalar::from(Some(false)), false)]
#[case(Scalar::null(DType::Bool(Nullability::Nullable)), false)]
fn test_ready_constant_mask(
    #[case] scalar: Scalar,
    #[case] value: bool,
    #[values(0, 1, 9)] len: usize,
) -> VortexResult<()> {
    let array = ConstantArray::new(scalar, len).into_array();
    assert_eq!(try_ready_piece_mask(&array)?, Some(Mask::new(len, value)));
    assert_piece_result(array, Mask::new(len, value))
}

#[rstest]
#[case(Validity::NonNullable)]
#[case(Validity::AllValid)]
#[case(Validity::AllInvalid)]
fn test_ready_bool_validity(#[case] validity: Validity) -> VortexResult<()> {
    let expected = if matches!(validity, Validity::AllInvalid) {
        Mask::new_false(3)
    } else {
        Mask::from_iter([true, false, true])
    };
    let array = BoolArray::new(BitBuffer::from_iter([true, false, true]), validity).into_array();
    assert_eq!(try_ready_piece_mask(&array)?, Some(expected.clone()));
    assert_piece_result(array, expected)
}

#[rstest]
fn test_ready_bool_preserves_offsets_and_padding(
    #[values(0, 1, 7, 8, 9, 65)] len: usize,
) -> VortexResult<()> {
    // Set padding and invalid value bits, and slice values and validity at different offsets.
    let values = BitBuffer::from_iter(
        (0..len + 8).map(|index| index < 3 || index >= len + 3 || !(index - 3).is_multiple_of(3)),
    )
    .slice(3..len + 3);
    let validity = BitBuffer::from_iter(
        (0..len + 8).map(|index| index < 5 || index >= len + 5 || (index - 5).is_multiple_of(2)),
    )
    .slice(5..len + 5);
    let array = BoolArray::new(
        values,
        Validity::Array(BoolArray::new(validity, Validity::NonNullable).into_array()),
    )
    .into_array();
    let expected =
        Mask::from_iter((0..len).map(|index| !index.is_multiple_of(3) && index.is_multiple_of(2)));
    assert_eq!(try_ready_piece_mask(&array)?, Some(expected.clone()));
    assert_piece_result(array, expected)
}

#[test]
fn test_encoded_piece_and_validity_keep_fallback() -> VortexResult<()> {
    let codes = buffer![0_u8, 1, 0].into_array();
    let encoded = DictArray::try_new(
        codes.clone(),
        BoolArray::from_iter([Some(true), None]).into_array(),
    )?
    .into_array();
    assert!(try_ready_piece_mask(&encoded)?.is_none());
    assert_piece_result(encoded, Mask::from_iter([true, false, true]))?;

    let validity =
        DictArray::try_new(codes, BoolArray::from_iter([true, false]).into_array())?.into_array();
    let array = BoolArray::new(BitBuffer::new_set(3), Validity::Array(validity)).into_array();
    assert!(try_ready_piece_mask(&array)?.is_none());
    assert_piece_result(array, Mask::from_iter([true, false, true]))?;

    let non_boolean = ConstantArray::new(1_i32, 3).into_array();
    assert!(try_ready_piece_mask(&non_boolean)?.is_none());
    Ok(())
}

#[test]
fn test_lazy_boolean_piece_keeps_fallback() -> VortexResult<()> {
    let array = BoolArray::from_iter([Some(false), Some(true), None, Some(false), Some(true)])
        .into_array()
        .slice(1..4)?;
    assert!(array.is::<Slice>());
    assert!(try_ready_piece_mask(&array)?.is_none());
    assert_piece_result(array, Mask::from_iter([true, false, false]))
}

#[test]
fn test_lazy_validity_keeps_fallback() -> VortexResult<()> {
    let validity = BoolArray::from_iter([false, true, false, true, false])
        .into_array()
        .slice(1..4)?;
    assert!(validity.is::<Slice>());
    let array = BoolArray::new(
        BitBuffer::from_iter([true, true, false]),
        Validity::Array(validity),
    )
    .into_array();
    assert!(try_ready_piece_mask(&array)?.is_none());
    assert_piece_result(array, Mask::from_iter([true, false, false]))
}

#[test]
fn test_non_boolean_piece_preserves_fallback_error() -> VortexResult<()> {
    let array = ConstantArray::new("not a predicate", 3).into_array();
    assert!(try_ready_piece_mask(&array)?.is_none());
    let reference = array
        .fill_null(false)
        .and_then(|array| array.execute::<Mask>(&mut SESSION.create_execution_ctx()));
    assert!(reference.is_err());
    assert!(
        piece_mask(
            &SESSION,
            Piece {
                rows: 41..42,
                array
            }
        )
        .is_err()
    );
    Ok(())
}

// A fast-path eligibility check must not access device memory or copy it to the host.
#[derive(Debug, PartialEq, Eq, Hash)]
struct MetadataOnlyDeviceBuffer(usize);

impl DeviceBuffer for MetadataOnlyDeviceBuffer {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn len(&self) -> usize {
        self.0
    }

    fn alignment(&self) -> Alignment {
        Alignment::of::<u8>()
    }

    fn copy_to_host_sync(&self, _alignment: Alignment) -> VortexResult<ByteBuffer> {
        panic!("mask eligibility must not read device values");
    }

    fn copy_to_host(
        &self,
        _alignment: Alignment,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ByteBuffer>>> {
        panic!("mask eligibility must not read device values");
    }

    fn slice(&self, _range: Range<usize>) -> Arc<dyn DeviceBuffer> {
        panic!("mask eligibility must not slice device values");
    }

    fn aligned(self: Arc<Self>, alignment: Alignment) -> VortexResult<Arc<dyn DeviceBuffer>> {
        if alignment != self.alignment() {
            vortex_bail!("mask eligibility must not realign device values");
        }
        Ok(self)
    }
}

#[test]
fn test_device_piece_and_validity_decline_without_reading() -> VortexResult<()> {
    let device = BufferHandle::new_device(Arc::new(MetadataOnlyDeviceBuffer(1)));
    for validity in [Validity::NonNullable, Validity::AllInvalid] {
        let array = BoolArray::try_new_from_handle(device.clone(), 3, 5, validity)?.into_array();
        assert!(try_ready_piece_mask(&array)?.is_none());
    }
    let validity = BoolArray::try_new_from_handle(device, 2, 5, Validity::NonNullable)?;
    let array = BoolArray::new(
        BitBuffer::new_set(5),
        Validity::Array(validity.into_array()),
    )
    .into_array();
    assert!(try_ready_piece_mask(&array)?.is_none());
    Ok(())
}
