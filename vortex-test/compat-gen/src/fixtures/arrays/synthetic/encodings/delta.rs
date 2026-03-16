// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::array::vtable::ArrayId;
use vortex::buffer::Buffer;
use vortex::encodings::fastlanes::Delta;
use vortex::encodings::fastlanes::DeltaArray;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::ArrayFixture;

pub struct DeltaFixture;

impl ArrayFixture for DeltaFixture {
    fn name(&self) -> &str {
        "delta.vortex"
    }

    fn description(&self) -> &str {
        "Monotonically increasing unsigned integers for Delta encoding"
    }

    fn expected_encodings(&self) -> Vec<ArrayId> {
        vec![Delta::ID]
    }

    fn build(&self) -> VortexResult<ArrayRef> {
        let monotonic_u64: Vec<u64> = (0..N as u64).map(|i| i * 3 + 1000).collect();
        let constant_delta_u32: Vec<u32> = (0..N as u32).collect();
        let large_stride_u64: Vec<u64> = (0..N as u64).map(|i| i * 1_000_000).collect();
        let monotonic_u16: Vec<u16> = (0..N as u16).map(|i| i / 2).collect();
        let monotonic_u8: Vec<u8> = (0..N).map(|i| (i / 4) as u8).collect();
        let large_base_u64: Vec<u64> = (0..N as u64).map(|i| 10_000_000_000 + i * 10).collect();

        let arr = StructArray::try_new(
            FieldNames::from([
                "monotonic_u64",
                "constant_delta_u32",
                "large_stride_u64",
                "monotonic_u16",
                "monotonic_u8",
                "large_base_u64",
            ]),
            vec![
                DeltaArray::try_from_primitive_array(&PrimitiveArray::new(
                    Buffer::from(monotonic_u64),
                    Validity::NonNullable,
                ))?
                .into_array(),
                DeltaArray::try_from_primitive_array(&PrimitiveArray::new(
                    Buffer::from(constant_delta_u32),
                    Validity::NonNullable,
                ))?
                .into_array(),
                DeltaArray::try_from_primitive_array(&PrimitiveArray::new(
                    Buffer::from(large_stride_u64),
                    Validity::NonNullable,
                ))?
                .into_array(),
                DeltaArray::try_from_primitive_array(&PrimitiveArray::new(
                    Buffer::from(monotonic_u16),
                    Validity::NonNullable,
                ))?
                .into_array(),
                DeltaArray::try_from_primitive_array(&PrimitiveArray::new(
                    Buffer::from(monotonic_u8),
                    Validity::NonNullable,
                ))?
                .into_array(),
                DeltaArray::try_from_primitive_array(&PrimitiveArray::new(
                    Buffer::from(large_base_u64),
                    Validity::NonNullable,
                ))?
                .into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
