// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::LazyLock;

use hegel::TestCase;
use hegel::generators as gs;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::conformance::hegel::EncodingCheckOptions;
use vortex_array::compute::conformance::hegel::check_encoding_roundtrip_and_ops;
use vortex_array::compute::conformance::hegel::draw_validity_for;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::BitPackedData;
use crate::Delta;
use crate::FL_CHUNK_SIZE;
use crate::FoR;
use crate::RLE;
use crate::bitpack_compress::bitpack_encode;

static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
    let session = VortexSession::empty();
    vortex_array::initialize(&session);
    crate::initialize(&session);
    session
});

const INT_PTYPES: [PType; 8] = [
    PType::U8,
    PType::U16,
    PType::U32,
    PType::U64,
    PType::I8,
    PType::I16,
    PType::I32,
    PType::I64,
];

fn draw_fl_len(tc: &TestCase) -> usize {
    if tc.draw(gs::integers::<u8>().max_value(5)) == 0 {
        tc.draw(
            gs::integers::<usize>()
                .min_value(FL_CHUNK_SIZE)
                .max_value(FL_CHUNK_SIZE + 56),
        )
    } else {
        tc.draw(gs::integers::<usize>().max_value(64))
    }
}

fn draw_nullability(tc: &TestCase) -> Nullability {
    if tc.draw(gs::booleans()) {
        Nullability::Nullable
    } else {
        Nullability::NonNullable
    }
}

fn draw_typed_ints<T>(
    tc: &TestCase,
    len: usize,
    validity: Validity,
    signed: bool,
    non_negative_signed: bool,
    run_heavy: bool,
) -> PrimitiveArray
where
    T: NativePType + gs::Integer + Ord + Copy,
{
    let zero = T::default();
    let max_narrow = num_traits::cast::<u8, T>(63).vortex_expect("63 fits integer ptype");
    let sorted = tc.draw(gs::integers::<u8>().max_value(3)) == 0;
    let narrow_range = tc.draw(gs::booleans());
    let draw_one = |tc: &TestCase| -> T {
        if narrow_range {
            tc.draw(gs::integers::<T>().min_value(zero).max_value(max_narrow))
        } else if signed && non_negative_signed {
            tc.draw(gs::integers::<T>().min_value(zero))
        } else {
            tc.draw(gs::integers::<T>())
        }
    };

    let mut vals = Vec::with_capacity(len);
    if run_heavy && len > 0 {
        let mut cur = zero;
        let mut run_left = 0usize;
        for _ in 0..len {
            if run_left == 0 {
                cur = draw_one(tc);
                run_left = tc.draw(gs::integers::<usize>().min_value(1).max_value(12));
            }
            vals.push(cur);
            run_left -= 1;
        }
    } else {
        for _ in 0..len {
            vals.push(draw_one(tc));
        }
    }
    if sorted {
        vals.sort_unstable();
    }
    PrimitiveArray::new(Buffer::from(vals), validity)
}

fn draw_int_primitive(
    tc: &TestCase,
    ptype: PType,
    len: usize,
    non_negative_signed: bool,
    run_heavy: bool,
) -> PrimitiveArray {
    let validity = draw_validity_for(tc, draw_nullability(tc), len);
    match ptype {
        PType::U8 => {
            draw_typed_ints::<u8>(tc, len, validity, false, non_negative_signed, run_heavy)
        }
        PType::U16 => {
            draw_typed_ints::<u16>(tc, len, validity, false, non_negative_signed, run_heavy)
        }
        PType::U32 => {
            draw_typed_ints::<u32>(tc, len, validity, false, non_negative_signed, run_heavy)
        }
        PType::U64 => {
            draw_typed_ints::<u64>(tc, len, validity, false, non_negative_signed, run_heavy)
        }
        PType::I8 => draw_typed_ints::<i8>(tc, len, validity, true, non_negative_signed, run_heavy),
        PType::I16 => {
            draw_typed_ints::<i16>(tc, len, validity, true, non_negative_signed, run_heavy)
        }
        PType::I32 => {
            draw_typed_ints::<i32>(tc, len, validity, true, non_negative_signed, run_heavy)
        }
        PType::I64 => {
            draw_typed_ints::<i64>(tc, len, validity, true, non_negative_signed, run_heavy)
        }
        _ => unreachable!(),
    }
}

fn check_bitpacked(tc: &TestCase) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let ptype = tc.draw(gs::sampled_from(&INT_PTYPES));
    let len = draw_fl_len(tc);
    // BitPacked requires non-negative signed values.
    let prim = draw_int_primitive(tc, ptype, len, true, false);
    let max_valid_bw = (ptype.byte_width() * 8) as u8 - 1;

    let encoded = if len > 0 && tc.draw(gs::integers::<u8>().max_value(3)) == 0 {
        let num_blocks = len.div_ceil(FL_CHUNK_SIZE);
        let bws: Vec<u8> = (0..num_blocks)
            .map(|_| tc.draw(gs::integers::<u8>().max_value(max_valid_bw)))
            .collect();
        BitPackedData::encode_blocked(&prim.clone().into_array(), &bws, &mut ctx)?.into_array()
    } else {
        let bw = tc.draw(gs::integers::<u8>().max_value(max_valid_bw));
        bitpack_encode(&prim, bw, None, &mut ctx)?.into_array()
    };

    check_encoding_roundtrip_and_ops(
        tc,
        &encoded,
        &prim.into_array(),
        &EncodingCheckOptions::default(),
        &mut ctx,
    )
}

fn check_delta(tc: &TestCase) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let ptype = tc.draw(gs::sampled_from(&INT_PTYPES));
    let len = draw_fl_len(tc);
    let prim = draw_int_primitive(tc, ptype, len, false, false);
    let encoded = Delta::try_from_primitive_array(&prim, &mut ctx)?.into_array();

    check_encoding_roundtrip_and_ops(
        tc,
        &encoded,
        &prim.into_array(),
        &EncodingCheckOptions::default(),
        &mut ctx,
    )
}

fn check_for(tc: &TestCase) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let ptype = tc.draw(gs::sampled_from(&INT_PTYPES));
    let len = draw_fl_len(tc);
    let prim = draw_int_primitive(tc, ptype, len, false, false);
    let has_valid = prim.validity()?.execute_mask(len, &mut ctx)?.true_count() > 0;
    let encoded = if !has_valid {
        let zero = Scalar::default_value(&DType::Primitive(ptype, Nullability::NonNullable))
            .cast(prim.dtype())?;
        FoR::try_new(prim.clone().into_array(), zero)?.into_array()
    } else if tc.draw(gs::booleans()) {
        FoR::encode(prim.clone(), &mut ctx)?.into_array()
    } else {
        FoR::encode_chunked(prim.clone(), &mut ctx)?.into_array()
    };

    check_encoding_roundtrip_and_ops(
        tc,
        &encoded,
        &prim.into_array(),
        &EncodingCheckOptions::default(),
        &mut ctx,
    )
}

fn check_rle(tc: &TestCase) -> VortexResult<()> {
    let mut ctx = SESSION.create_execution_ctx();
    let len = draw_fl_len(tc);
    let prim = match tc.draw(gs::integers::<u8>().max_value(9)) {
        0 => {
            let validity = draw_validity_for(tc, draw_nullability(tc), len);
            let mut v = tc.draw(gs::vecs(gs::floats::<f32>()).min_size(len).max_size(len));
            if tc.draw(gs::integers::<u8>().max_value(3)) == 0 {
                v.sort_by(f32::total_cmp);
            }
            PrimitiveArray::new(Buffer::from(v), validity)
        }
        1 => {
            let validity = draw_validity_for(tc, draw_nullability(tc), len);
            let mut v = tc.draw(gs::vecs(gs::floats::<f64>()).min_size(len).max_size(len));
            if tc.draw(gs::integers::<u8>().max_value(3)) == 0 {
                v.sort_by(f64::total_cmp);
            }
            PrimitiveArray::new(Buffer::from(v), validity)
        }
        _ => {
            let ptype = tc.draw(gs::sampled_from(&INT_PTYPES));
            draw_int_primitive(tc, ptype, len, false, true)
        }
    };
    let encoded = RLE::encode(prim.as_view(), &mut ctx)?.into_array();

    check_encoding_roundtrip_and_ops(
        tc,
        &encoded,
        &prim.into_array(),
        &EncodingCheckOptions::default(),
        &mut ctx,
    )
}

#[hegel::test(test_cases = 35, print_blob = true)]
fn test_bitpacked_hegel_encoding_harness(tc: TestCase) {
    check_bitpacked(&tc).vortex_expect("BitPacked harness check");
}

#[hegel::test(test_cases = 35, print_blob = true)]
fn test_delta_hegel_encoding_harness(tc: TestCase) {
    check_delta(&tc).vortex_expect("Delta harness check");
}

#[hegel::test(test_cases = 35, print_blob = true)]
fn test_for_hegel_encoding_harness(tc: TestCase) {
    check_for(&tc).vortex_expect("FoR harness check");
}

#[hegel::test(test_cases = 35, print_blob = true)]
fn test_rle_hegel_encoding_harness(tc: TestCase) {
    check_rle(&tc).vortex_expect("RLE harness check");
}
