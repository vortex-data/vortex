// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Host-copy behavior for canonical nodes and non-canonical nodes backed by device buffers.

use std::any::Any;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use super::SESSION;
use super::canonical_reference;
use super::decimal_leaf;
use super::deserialize;
use super::metadata;
use super::stored_values;
use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::arrays::BoolArray;
use crate::arrays::Decimal;
use crate::arrays::DictArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::assert_arrays_eq;
use crate::buffer::BufferHandle;
use crate::buffer::DeviceBuffer;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::Nullability;
use crate::dtype::PType;
use crate::validity::Validity;

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
