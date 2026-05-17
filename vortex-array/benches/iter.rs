// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compare `ArrayAccessor::with_iterator` against the fastest hand-written
//! buffer iteration we can express today, for the three array types that
//! currently implement an iterator (primitive, varbin, varbinview), and
//! against the best column-wise / scalar-at paths for a complex array
//! (struct).
//!
//! Each benchmark performs a tiny reduction (sum / byte-len-sum / row count)
//! so the loop body cannot be eliminated.

#![expect(clippy::unwrap_used)]
#![expect(clippy::many_single_char_names)]
#![expect(deprecated)]

use divan::Bencher;
use divan::black_box;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::ToCanonical as _;
use vortex_array::VortexSessionExecute;
use vortex_array::accessor::ArrayAccessor;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::VarBinArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::bool::BoolArrayExt;
use vortex_array::arrays::dict_test::gen_primitive_for_dict;
use vortex_array::arrays::dict_test::gen_varbin_words;
use vortex_array::arrays::struct_::StructArrayExt;
use vortex_array::arrays::varbin::{VarBinArrayExt, iter_offsets};
use vortex_array::dtype::Nullability;
use vortex_array::iter_array::IterArray;
use vortex_array::iter_array::IterArrayValue;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;

fn main() {
    divan::main();
}

const LENGTHS: &[usize] = &[10_000, 100_000];

// ---------- builders -------------------------------------------------------

fn build_primitive_i32(len: usize, with_nulls: bool) -> PrimitiveArray {
    let arr = gen_primitive_for_dict::<i32>(len, 256);
    let buffer: Buffer<i32> = arr.as_slice::<i32>().iter().copied().collect();
    if with_nulls {
        let mut mask = BitBufferMut::new_set(len);
        for i in (0..len).step_by(7) {
            mask.set_to(i, false);
        }
        let validity = Validity::from_bit_buffer(mask.freeze(), Nullability::Nullable);
        PrimitiveArray::new(buffer, validity)
    } else {
        PrimitiveArray::new(buffer, Validity::NonNullable)
    }
}

fn build_varbin(len: usize, with_nulls: bool) -> VarBinArray {
    let strings = gen_varbin_words(len, 256);
    if with_nulls {
        let with_opts: Vec<Option<&[u8]>> = strings
            .iter()
            .enumerate()
            .map(|(i, s)| if i % 7 == 0 { None } else { Some(s.as_bytes()) })
            .collect();
        VarBinArray::from_nullable_bytes(with_opts)
    } else {
        VarBinArray::from(strings)
    }
}

fn build_varbinview(len: usize, with_nulls: bool) -> VarBinViewArray {
    let strings = gen_varbin_words(len, 256);
    if with_nulls {
        VarBinViewArray::from_iter_nullable_str(
            strings
                .into_iter()
                .enumerate()
                .map(|(i, s)| if i % 7 == 0 { None } else { Some(s) }),
        )
    } else {
        VarBinViewArray::from_iter_str(strings)
    }
}

fn build_struct(len: usize) -> StructArray {
    let ints = gen_primitive_for_dict::<i32>(len, 16).into_array();
    let strs = VarBinViewArray::from_iter_str(gen_varbin_words(len, 64)).into_array();
    StructArray::try_from_iter([("a", ints), ("b", strs)]).unwrap()
}

fn build_bool(len: usize, with_nulls: bool) -> BoolArray {
    let mut bits = BitBufferMut::new_unset(len);
    for i in 0..len {
        if i % 3 == 0 {
            bits.set(i);
        }
    }
    let validity = if with_nulls {
        let mut mask = BitBufferMut::new_set(len);
        for i in (0..len).step_by(7) {
            mask.set_to(i, false);
        }
        Validity::from_bit_buffer(mask.freeze(), Nullability::Nullable)
    } else {
        Validity::NonNullable
    };
    BoolArray::new(bits.freeze(), validity)
}

// ---------- primitive: with_iterator vs raw slice --------------------------

#[divan::bench(args = LENGTHS)]
fn primitive_i32_current_nonnull(b: Bencher, len: usize) {
    let arr = build_primitive_i32(len, false);
    b.bench_local(|| {
        let s: i64 = arr.with_iterator(|it: &mut dyn Iterator<Item = Option<&i32>>| {
            it.flatten().map(|v| *v as i64).sum()
        });
        black_box(s)
    });
}

#[divan::bench(args = LENGTHS)]
fn primitive_i32_new_nonnull(b: Bencher, len: usize) {
    let arr = build_primitive_i32(len, false);
    b.bench_local(|| {
        let s: i64 = IterArray::<i32>::iter(&arr)
            .flatten()
            .map(|v| *v as i64)
            .sum();
        black_box(s)
    });
}

#[divan::bench(args = LENGTHS)]
fn primitive_i32_manual_nonnull(b: Bencher, len: usize) {
    let arr = build_primitive_i32(len, false);
    b.bench_local(|| {
        let s: i64 = arr.as_slice::<i32>().iter().map(|v| *v as i64).sum();
        black_box(s)
    });
}

#[divan::bench(args = LENGTHS)]
fn primitive_i32_current_nullable(b: Bencher, len: usize) {
    let arr = build_primitive_i32(len, true);
    b.bench_local(|| {
        let s: i64 = arr.with_iterator(|it: &mut dyn Iterator<Item = Option<&i32>>| {
            it.map(|opt| opt.copied().unwrap_or(0) as i64).sum()
        });
        black_box(s)
    });
}

#[divan::bench(args = LENGTHS)]
fn primitive_i32_new_nullable(b: Bencher, len: usize) {
    let arr = build_primitive_i32(len, true);
    b.bench_local(|| {
        let s: i64 = IterArray::<i32>::iter(&arr)
            .map(|opt| opt.copied().unwrap_or(0) as i64)
            .sum();
        black_box(s)
    });
}

#[divan::bench(args = LENGTHS)]
fn primitive_i32_manual_nullable(b: Bencher, len: usize) {
    let arr = build_primitive_i32(len, true);
    // Resolve validity to a bit buffer ONCE, outside the timed region. The
    // existing `with_iterator` does this every call.
    let validity = arr.validity().vortex_expect("validity");
    let bits = match validity {
        Validity::Array(v) => Some(v.to_bool().into_bit_buffer()),
        _ => None,
    };
    b.bench_local(|| {
        let values = arr.as_slice::<i32>();
        let s: i64 = match &bits {
            Some(bits) => values
                .iter()
                .zip(bits.iter())
                .map(|(v, valid)| if valid { *v as i64 } else { 0 })
                .sum(),
            None => values.iter().map(|v| *v as i64).sum(),
        };
        black_box(s)
    });
}

// ---------- varbin: with_iterator vs raw offsets+bytes ---------------------

#[divan::bench(args = LENGTHS)]
fn varbin_current_nonnull(b: Bencher, len: usize) {
    let arr = build_varbin(len, false);
    b.bench_local(|| {
        let s: usize = arr.with_iterator(|it: &mut dyn Iterator<Item = Option<&[u8]>>| {
            it.flatten().map(|b| b.len()).sum()
        });
        black_box(s)
    });
}

#[divan::bench(args = LENGTHS)]
fn varbin_new_nonnull(b: Bencher, len: usize) {
    let arr = build_varbin(len, false);
    b.bench_local(|| {
        let s: usize = IterArray::<[u8]>::iter(&arr)
            .flatten()
            .map(|b| b.len())
            .sum();
        black_box(s)
    });
}

#[divan::bench(args = LENGTHS)]
fn varbin_new_typed_u32_nonnull(b: Bencher, len: usize) {
    // Typed escape hatch: caller knows offsets are u32, no upfront
    // conversion. Per-tick reads u32 and casts to usize (free on x86).
    let arr = build_varbin(len, false);
    b.bench_local(|| {
        let s: usize = iter_offsets::<u32>(&arr).flatten().map(|b| b.len()).sum();
        black_box(s)
    });
}

#[divan::bench(args = LENGTHS)]
fn varbin_manual_nonnull(b: Bencher, len: usize) {
    let arr = build_varbin(len, false);
    // Materialize offsets once. The existing `with_iterator` does
    // `offsets().to_primitive()` on every call.
    let offsets_array = arr.offsets().to_primitive();
    let offsets: Buffer<u32> = offsets_array.as_slice::<u32>().iter().copied().collect();
    let bytes_buf = arr.bytes().clone();
    b.bench_local(|| {
        let bytes = bytes_buf.as_slice();
        // Build the actual &[u8] per element to match what `with_iterator`
        // hands the consumer.
        let s: usize = offsets
            .as_slice()
            .windows(2)
            .map(|w| {
                let slice = &bytes[w[0] as usize..w[1] as usize];
                slice.len()
            })
            .sum();
        black_box(s)
    });
}

// ---------- varbinview: with_iterator vs raw views walk --------------------

#[divan::bench(args = LENGTHS)]
fn varbinview_current_nonnull(b: Bencher, len: usize) {
    let arr = build_varbinview(len, false);
    b.bench_local(|| {
        let s: usize = arr.with_iterator(|it: &mut dyn Iterator<Item = Option<&[u8]>>| {
            it.flatten().map(|b| b.len()).sum()
        });
        black_box(s)
    });
}

#[divan::bench(args = LENGTHS)]
fn varbinview_new_nonnull(b: Bencher, len: usize) {
    let arr = build_varbinview(len, false);
    b.bench_local(|| {
        let s: usize = IterArray::<[u8]>::iter(&arr)
            .flatten()
            .map(|b| b.len())
            .sum();
        black_box(s)
    });
}

#[divan::bench(args = LENGTHS)]
fn varbinview_manual_nonnull(b: Bencher, len: usize) {
    let arr = build_varbinview(len, false);
    b.bench_local(|| {
        let views = arr.views();
        let s: usize = views
            .iter()
            .map(|v| {
                if v.is_inlined() {
                    v.as_inlined().value().len()
                } else {
                    v.as_view().as_range().len()
                }
            })
            .sum();
        black_box(s)
    });
}

// ---------- complex: struct row iteration ----------------------------------
//
// Workload: count rows where field `a` (i32) is even AND field `b` (str)
// has length > 8. Touches both columns once per row.

#[divan::bench(args = LENGTHS)]
fn struct_row_scalar_at(b: Bencher, len: usize) {
    // Naive: scalar_at per field per row. What most consumers reach for
    // today when there is no row iterator.
    let s = build_struct(len);
    let int_col = s.unmasked_field(0).clone();
    let str_col = s.unmasked_field(1).clone();
    b.bench_local(|| {
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let mut count = 0u64;
        for i in 0..len {
            let a = int_col.execute_scalar(i, &mut ctx).unwrap();
            let bs = str_col.execute_scalar(i, &mut ctx).unwrap();
            let a_v: i32 = (&a).try_into().unwrap();
            let b_v: String = (&bs).try_into().unwrap();
            if a_v % 2 == 0 && b_v.len() > 8 {
                count += 1;
            }
        }
        black_box(count)
    });
}

#[divan::bench(args = LENGTHS)]
fn struct_row_with_iterator_zip(b: Bencher, len: usize) {
    // Column-wise: canonicalize each field once, then use the existing
    // with_iterator API for the str column and zip with the primitive slice.
    let s = build_struct(len);
    let int_col = s.unmasked_field(0).clone().to_primitive();
    let str_col = s.unmasked_field(1).clone().to_varbinview();
    b.bench_local(|| {
        let ints = int_col.as_slice::<i32>();
        let mut count = 0u64;
        str_col.with_iterator(|str_iter: &mut dyn Iterator<Item = Option<&[u8]>>| {
            for (a, b_opt) in ints.iter().zip(str_iter) {
                if let Some(b_v) = b_opt
                    && *a % 2 == 0
                    && b_v.len() > 8
                {
                    count += 1;
                }
            }
        });
        black_box(count)
    });
}

// ---------- bool: new iter vs raw BitBuffer ------------------------------
//
// BoolArray had NO with_iterator impl before, so the comparison is between
// the new IterArrayValue and the most direct raw-BitBuffer walk.

#[divan::bench(args = LENGTHS)]
fn bool_new_nonnull(b: Bencher, len: usize) {
    let arr = build_bool(len, false);
    b.bench_local(|| {
        let count: u64 = arr.iter_value().filter(|opt| opt == &Some(true)).count() as u64;
        black_box(count)
    });
}

#[divan::bench(args = LENGTHS)]
fn bool_manual_nonnull(b: Bencher, len: usize) {
    let arr = build_bool(len, false);
    let bits: BitBuffer = arr.to_bit_buffer();
    b.bench_local(|| {
        let count: u64 = bits.iter().filter(|b| *b).count() as u64;
        black_box(count)
    });
}

#[divan::bench(args = LENGTHS)]
fn bool_new_nullable(b: Bencher, len: usize) {
    let arr = build_bool(len, true);
    b.bench_local(|| {
        let count: u64 = arr.iter_value().filter(|opt| opt == &Some(true)).count() as u64;
        black_box(count)
    });
}

#[divan::bench(args = LENGTHS)]
fn struct_row_new_iter_zip(b: Bencher, len: usize) {
    // New IterArray: zip per-field concrete iterators. This is the
    // recommended row-iteration pattern.
    let s = build_struct(len);
    let int_col = s.unmasked_field(0).clone().to_primitive();
    let str_col = s.unmasked_field(1).clone().to_varbinview();
    b.bench_local(|| {
        let int_iter = IterArray::<i32>::iter(&int_col);
        let str_iter = IterArray::<[u8]>::iter(&str_col);
        let mut count = 0u64;
        for (a_opt, b_opt) in int_iter.zip(str_iter) {
            if let (Some(a), Some(bv)) = (a_opt, b_opt)
                && *a % 2 == 0
                && bv.len() > 8
            {
                count += 1;
            }
        }
        black_box(count)
    });
}

#[divan::bench(args = LENGTHS)]
fn struct_row_manual_zip(b: Bencher, len: usize) {
    // Lower bound: column slices for both, raw view walk for the str column.
    let s = build_struct(len);
    let int_col = s.unmasked_field(0).clone().to_primitive();
    let str_col = s.unmasked_field(1).clone().to_varbinview();
    b.bench_local(|| {
        let ints = int_col.as_slice::<i32>();
        let views = str_col.views();
        let mut count = 0u64;
        for (a, v) in ints.iter().zip(views.iter()) {
            let blen = if v.is_inlined() {
                v.as_inlined().value().len()
            } else {
                v.as_view().as_range().len()
            };
            if *a % 2 == 0 && blen > 8 {
                count += 1;
            }
        }
        black_box(count)
    });
}
