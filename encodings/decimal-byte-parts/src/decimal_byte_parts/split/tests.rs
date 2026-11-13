// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DecimalDType;
use vortex_array::dtype::PType;
use vortex_array::dtype::i256;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use super::split_decimal;
use crate::decimal_byte_parts::LOWER_PART_DTYPE;
use crate::decimal_byte_parts::MAX_LOWER_PARTS;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    crate::initialize(&session);
    session
});

#[test]
fn test_split_i256_part_count_and_types() -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal = DecimalArray::new(
        Buffer::from(vec![i256::from_i128(i128::MAX), i256::MIN]),
        DecimalDType::new(76, 0),
        Validity::NonNullable,
    );
    let parts = split_decimal(&decimal, &mut ctx)?;
    assert_eq!(parts.lower_parts.len(), MAX_LOWER_PARTS);
    assert_eq!(parts.msp.dtype().as_ptype(), PType::I64);
    for part in &parts.lower_parts {
        assert_eq!(part.dtype(), &LOWER_PART_DTYPE);
    }
    Ok(())
}

#[rstest]
fn test_split_i256_part_order(
    #[values(Validity::NonNullable, Validity::from_iter([true, false, true]))] validity: Validity,
) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let decimal = DecimalArray::new(
        buffer![
            i256::from_parts((2u128 << 64) | 3, (1i128 << 64) | 4),
            i256::ZERO,
            i256::from_parts((6u128 << 64) | 7, (-2i128 << 64) | 5),
        ],
        DecimalDType::new(76, 0),
        validity.clone(),
    );
    let parts = split_decimal(&decimal, &mut ctx)?;
    assert_arrays_eq!(
        parts.msp,
        PrimitiveArray::new(buffer![1i64, 0, -2], validity),
        &mut ctx
    );
    assert_eq!(parts.lower_parts.len(), 3);
    for (part, expected) in parts.lower_parts.into_iter().zip([
        buffer![4u64, 0, 5],
        buffer![2u64, 0, 6],
        buffer![3u64, 0, 7],
    ]) {
        assert_arrays_eq!(
            part,
            PrimitiveArray::new(expected, Validity::NonNullable),
            &mut ctx
        );
    }

    Ok(())
}
