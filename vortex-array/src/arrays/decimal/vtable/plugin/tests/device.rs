// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Reads of canonical and legacy nodes backed by device buffers.

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use rstest::rstest;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::deserialize;
use super::metadata;
use super::stored_values;
use crate::ArrayRef;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::Decimal;
use crate::arrays::DictArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::buffer::BufferHandle;
use crate::buffer::DeviceBuffer;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::validity::Validity;

/// A device buffer that holds its bytes on the host and fails every copy to the host, so a read
/// that copies it fails.
#[derive(Debug, PartialEq, Eq, Hash)]
struct TestDeviceBuffer {
    bytes: ByteBuffer,
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

    fn copy_to_host_sync(&self, _alignment: Alignment) -> VortexResult<ByteBuffer> {
        vortex_bail!("a decimal read never copies device buffers to the host")
    }

    fn copy_to_host(
        &self,
        _alignment: Alignment,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ByteBuffer>>> {
        vortex_bail!("a decimal read never copies device buffers to the host")
    }

    fn slice(&self, range: Range<usize>) -> Arc<dyn DeviceBuffer> {
        Arc::new(Self {
            bytes: self.bytes.slice(range),
        })
    }

    fn aligned(self: Arc<Self>, alignment: Alignment) -> VortexResult<Arc<dyn DeviceBuffer>> {
        assert!(self.alignment().is_aligned_to(alignment));
        Ok(self)
    }
}

fn on_device(bytes: ByteBuffer) -> BufferHandle {
    BufferHandle::new_device(Arc::new(TestDeviceBuffer { bytes }))
}

/// Returns the validity of `[true, false, true]` as a bool array in device memory.
fn device_validity() -> VortexResult<ArrayRef> {
    let bits = on_device(ByteBuffer::copy_from([0b101u8]));

    Ok(BoolArray::try_new_from_handle(bits, 0, 3, Validity::NonNullable)?.into_array())
}

/// Returns the validity of `[true, false, true]` as a dict whose codes are in device memory.
fn nested_device_validity() -> VortexResult<ArrayRef> {
    let codes = PrimitiveArray::from_buffer_handle(
        on_device(buffer![0u8, 1, 0].into_byte_buffer()),
        PType::U8,
        Validity::NonNullable,
    );
    let validity = DictArray::try_new(
        codes.into_array(),
        BoolArray::from_iter([true, false]).into_array(),
    )?;

    Ok(validity.into_array())
}

fn host_validity() -> VortexResult<ArrayRef> {
    Ok(BoolArray::from_iter([true, false, true]).into_array())
}

/// A canonical node keeps its device buffer, because its read never scans the values.
#[test]
fn canonical_node_keeps_device_buffer() -> VortexResult<()> {
    let dtype = DType::Decimal(DecimalDType::new(18, 2), Nullability::Nullable);

    let canonical = deserialize(
        &dtype,
        3,
        &metadata(DecimalType::I64),
        &[on_device(buffer![-99i64, 7, 99].into_byte_buffer())],
        &[device_validity()?],
    )?;

    assert!(canonical.as_::<Decimal>().buffer_handle().is_on_device());

    Ok(())
}

/// A legacy node fails when its values or any buffer of its validity are in device memory, because
/// choosing its width scans the values on the host.
#[rstest]
#[case::device_values(true, host_validity)]
#[case::device_validity(false, device_validity)]
#[case::nested_device_validity(false, nested_device_validity)]
fn legacy_node_in_device_memory_fails(
    #[case] values_on_device: bool,
    #[case] validity: fn() -> VortexResult<ArrayRef>,
) -> VortexResult<()> {
    let dtype = DType::Decimal(DecimalDType::new(18, 2), Nullability::Nullable);
    let values = if values_on_device {
        on_device(buffer![-99i32, 7, 99].into_byte_buffer())
    } else {
        stored_values(DecimalType::I32, &[-99, 7, 99])
    };

    let result = deserialize(
        &dtype,
        3,
        &metadata(DecimalType::I32),
        &[values],
        &[validity()?],
    );

    assert!(result.is_err_and(|error| error.to_string().contains("must be in host memory")));

    Ok(())
}
