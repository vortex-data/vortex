// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::dtype::FieldNames;
use vortex::array::validity::Validity;
use vortex::array::vtable::ArrayId;
use vortex::encodings::fastlanes::Delta;
use vortex::encodings::fastlanes::DeltaArray;
use vortex::error::VortexResult;

use super::N;
use crate::fixtures::FlatLayoutFixture;

pub struct DeltaFixture;

impl FlatLayoutFixture for DeltaFixture {
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
        let monotonic_u64: PrimitiveArray = (0..N as u64).map(|i| i * 3 + 1000).collect();
        let constant_delta_u32: PrimitiveArray = (0..N as u32).collect();
        let large_stride_u64: PrimitiveArray = (0..N as u64).map(|i| i * 1_000_000).collect();
        let monotonic_u16: PrimitiveArray = (0..N as u16).map(|i| i / 2).collect();
        let monotonic_u8: PrimitiveArray = (0..N).map(|i| (i / 4) as u8).collect();
        let large_base_u64: PrimitiveArray =
            (0..N as u64).map(|i| 10_000_000_000 + i * 10).collect();
        let all_zero_deltas: PrimitiveArray = std::iter::repeat_n(777u64, N).collect();
        let irregular_monotone: PrimitiveArray = (0..N as u64)
            .scan(0u64, |state, i| {
                *state += (i % 7) + 1;
                Some(*state)
            })
            .collect();
        let near_overflow_base: PrimitiveArray = (0..N as u64)
            .map(|i| u64::MAX - ((N as u64 - i) * 2))
            .collect();
        let nullable_monotone = PrimitiveArray::from_option_iter((0..N as u64).map(|i| {
            if i < 8 || i >= N as u64 - 8 {
                None
            } else {
                Some(50_000 + i * 4)
            }
        }));

        let arr = StructArray::try_new(
            FieldNames::from([
                "monotonic_u64",
                "constant_delta_u32",
                "large_stride_u64",
                "monotonic_u16",
                "monotonic_u8",
                "large_base_u64",
                "all_zero_deltas",
                "irregular_monotone",
                "near_overflow_base",
                "nullable_monotone",
            ]),
            vec![
                DeltaArray::try_from_primitive_array(&monotonic_u64)?.into_array(),
                DeltaArray::try_from_primitive_array(&constant_delta_u32)?.into_array(),
                DeltaArray::try_from_primitive_array(&large_stride_u64)?.into_array(),
                DeltaArray::try_from_primitive_array(&monotonic_u16)?.into_array(),
                DeltaArray::try_from_primitive_array(&monotonic_u8)?.into_array(),
                DeltaArray::try_from_primitive_array(&large_base_u64)?.into_array(),
                DeltaArray::try_from_primitive_array(&all_zero_deltas)?.into_array(),
                DeltaArray::try_from_primitive_array(&irregular_monotone)?.into_array(),
                DeltaArray::try_from_primitive_array(&near_overflow_base)?.into_array(),
                DeltaArray::try_from_primitive_array(&nullable_monotone)?.into_array(),
            ],
            N,
            Validity::NonNullable,
        )?;
        Ok(arr.into_array())
    }
}
