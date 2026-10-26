// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Public width dispatch and bit-preserving structural operations.

use std::sync::LazyLock;

use rstest::rstest;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::InterleaveArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::assert_arrays_eq;
use vortex_array::dtype::DecimalType;
use vortex_array::dtype::PType;
use vortex_array::match_each_decimal_type_width;
use vortex_array::match_each_power_of_two_width;
use vortex_array::match_each_ptype_width;
use vortex_array::patches::Patches;
use vortex_array::scalar_fn::fns::fill_null::FillNullKernel;
use vortex_array::scalar_fn::fns::zip::ZipKernel;
use vortex_array::validity::Validity;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_session::VortexSession;

static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

/// Builds over-aligned storage from the low bytes of each supplied bit pattern.
fn primitive_from_bits(ptype: PType, bits: &[u64], validity: Validity) -> PrimitiveArray {
    let width = ptype.byte_width();
    let values = bits
        .iter()
        .flat_map(|bits| bits.to_le_bytes().into_iter().take(width));
    let buffer = Buffer::from_iter(values).aligned(Alignment::new(64));

    PrimitiveArray::from_byte_buffer(buffer, ptype, validity)
}

#[track_caller]
fn assert_records(actual: &PrimitiveArray, ptype: PType, bits: &[u64], ctx: &mut ExecutionCtx) {
    let expected = primitive_from_bits(ptype, bits, Validity::NonNullable);
    assert_arrays_eq!((*actual).clone(), expected, ctx);
    assert_eq!(
        actual.buffer_handle().as_host_opt().unwrap().as_slice(),
        expected.buffer_handle().as_host_opt().unwrap().as_slice(),
    );
}

#[rstest]
fn accepts_owned_and_borrowed_ptypes(
    #[values(
        PType::U8,
        PType::U16,
        PType::U32,
        PType::U64,
        PType::I8,
        PType::I16,
        PType::I32,
        PType::I64,
        PType::F16,
        PType::F32,
        PType::F64
    )]
    ptype: PType,
) {
    let owned = match_each_ptype_width!(ptype, |W| { W }, _ => { unreachable!() });
    let borrowed = match_each_ptype_width!(&ptype, |W| { W }, _ => { unreachable!() });
    assert_eq!(owned, ptype.byte_width());
    assert_eq!(borrowed, owned);
}

#[rstest]
fn accepts_owned_and_borrowed_decimal_types(
    #[values(
        DecimalType::I8,
        DecimalType::I16,
        DecimalType::I32,
        DecimalType::I64,
        DecimalType::I128,
        DecimalType::I256
    )]
    values_type: DecimalType,
) {
    let owned = match_each_decimal_type_width!(values_type, |W| { W }, _ => { unreachable!() });
    let borrowed = match_each_decimal_type_width!(&values_type, |W| { W }, _ => { unreachable!() });
    assert_eq!(owned, values_type.byte_width());
    assert_eq!(borrowed, owned);
}

#[test]
fn forwards_specializations_and_evaluates_physical_types_once() {
    let mut ptype_calls = 0;
    let primitive = match_each_ptype_width!(
        {
            ptype_calls += 1;
            PType::F32
        },
        widths = [1, 8],
        |W| { Ok(W) },
        other => { Err(other) },
    );
    let mut decimal_calls = 0;
    let decimal = match_each_decimal_type_width!(
        {
            decimal_calls += 1;
            DecimalType::I128
        },
        widths = [1, 8],
        |W| { Ok(W) },
        other => { Err(other) },
    );
    assert_eq!(primitive, Err(4));
    assert_eq!(decimal, Err(16));
    assert_eq!((ptype_calls, decimal_calls), (1, 1));

    let width = match_each_power_of_two_width!(16, |W| { W }, _ => { unreachable!() });
    assert_eq!(width, 16);
}

#[rstest]
fn records_survive_structural_kernels(
    #[values(
        PType::U8,
        PType::U16,
        PType::U32,
        PType::U64,
        PType::I8,
        PType::I16,
        PType::I32,
        PType::I64,
        PType::F16,
        PType::F32,
        PType::F64
    )]
    ptype: PType,
    #[values(false, true)] shared: bool,
) -> VortexResult<()> {
    let bits = match ptype {
        PType::F16 => [0, 0x8000, 0x7c00, 0xfc00, 0x7e01, 0x7e02],
        PType::F32 => [
            0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc00001, 0x7fc00002,
        ],
        PType::F64 => [
            0,
            0x8000000000000000,
            0x7ff0000000000000,
            0xfff0000000000000,
            0x7ff8000000000001,
            0x7ff8000000000002,
        ],
        _ => [0, u64::MAX, 1, 2, 0x8080808080808080, 0x7f7f7f7f7f7f7f7f],
    };
    let mut ctx = SESSION.create_execution_ctx();
    let input = primitive_from_bits(ptype, &bits, Validity::NonNullable);
    let input_ptr = input.buffer_handle().as_host_opt().unwrap().as_ptr();
    let retained = shared.then(|| input.clone());
    let patches = Patches::new(
        bits.len(),
        7,
        PrimitiveArray::from_iter([8u32, 11]).into_array(),
        primitive_from_bits(ptype, &[bits[5], bits[0]], Validity::NonNullable).into_array(),
        None,
        &mut ctx,
    )?;
    let patched = input.patch(&patches, &mut ctx)?;
    assert_records(
        &patched,
        ptype,
        &[bits[0], bits[5], bits[2], bits[3], bits[0], bits[5]],
        &mut ctx,
    );
    assert!(patched.buffer_handle().is_aligned_to(Alignment::new(64)));
    let output_ptr = patched.buffer_handle().as_host_opt().unwrap().as_ptr();
    if let Some(retained) = retained {
        assert_ne!(output_ptr, input_ptr);
        assert_records(&retained, ptype, &bits, &mut ctx);
    } else {
        assert_eq!(output_ptr, input_ptr);
    }

    let if_false = primitive_from_bits(ptype, &bits, Validity::NonNullable).into_array();
    let mask = Mask::from_iter([true, false, true, false, true, false]).into_array();
    let zipped = <Primitive as ZipKernel>::zip(patched.as_view(), &if_false, &mask, &mut ctx)?
        .vortex_expect("mixed primitive zip uses the primitive kernel")
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert_records(
        &zipped,
        ptype,
        &[bits[0], bits[1], bits[2], bits[3], bits[0], bits[5]],
        &mut ctx,
    );

    let nullable = PrimitiveArray::from_buffer_handle(
        zipped.buffer_handle().clone(),
        ptype,
        Validity::from_iter([true, false, true, false, true, true]),
    );
    let fill = if_false.execute_scalar(5, &mut ctx)?;
    let filled = <Primitive as FillNullKernel>::fill_null(nullable.as_view(), &fill, &mut ctx)?
        .vortex_expect("mixed primitive validity uses the fill-null kernel")
        .execute::<PrimitiveArray>(&mut ctx)?;
    assert_records(
        &filled,
        ptype,
        &[bits[0], bits[5], bits[2], bits[5], bits[0], bits[5]],
        &mut ctx,
    );

    let interleaved = InterleaveArray::try_new(
        vec![ConstantArray::new(fill, 3).into_array(), filled.into_array()],
        PrimitiveArray::from_iter([0u8, 1, 0, 1]).into_array(),
        PrimitiveArray::from_iter([2u32, 2, 0, 0]).into_array(),
    )?
    .into_array()
    .execute::<PrimitiveArray>(&mut ctx)?;
    assert_records(
        &interleaved,
        ptype,
        &[bits[5], bits[2], bits[5], bits[0]],
        &mut ctx,
    );
    Ok(())
}
