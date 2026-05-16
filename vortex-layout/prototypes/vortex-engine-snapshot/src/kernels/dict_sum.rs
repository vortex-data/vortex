//! Dict-aware `Sum` aggregate kernel — fused gather-sum.
//!
//! Computes `Σ values[codes[i]]` directly: walk the codes array,
//! gather from the (small) dictionary `values` per code, accumulate.
//! Skips the intermediate N-row materialisation that the canonical
//! `Dict → take(codes, values) → SIMD sum` path performs.
//!
//! Why fused, not histogram-and-multiply? Earlier this kernel used
//! `Σ values[k] * freq[k]` with a per-batch histogram walk over the
//! codes (K multiplies after one O(N) histogram). For simple-numeric
//! value dtypes the histogram phase is bottlenecked at ~2 cycles per
//! code by the scalar read-modify-write through `freq[c]`, while the
//! canonical path's SIMD sum runs at ~32-64 lanes/cycle. The
//! "savings" on the multiply step never recover. The fused path
//! still pays one gather per code (same as canonical's take), but
//! avoids the N-element materialisation write+read.
//!
//! Registered against `(Dict, Sum)`. Used directly by `Sum`
//! aggregates and — via the per-child dispatch in
//! `Combined<V>::try_accumulate` (vortex develop) — reused by
//! `Mean` / `Avg` through `Combined<Mean>`'s inner `Sum` child.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::sum::Sum;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::Dict;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;

#[derive(Debug)]
pub struct DictSumKernel;

impl DynAggregateKernel for DictSumKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<Sum>() {
            return Ok(None);
        }
        let Some(dict_view) = batch.as_opt::<Dict>() else {
            return Ok(None);
        };
        let codes = dict_view.codes();
        let values = dict_view.values();

        let values_ptype = match values.dtype() {
            DType::Primitive(p, _) => *p,
            _ => return Ok(None),
        };

        // Canonicalise both children. Values is small (K rows, often
        // cached in the dict layout's SharedArray); codes is the
        // large one — that's where the work is.
        let values_canonical = values.clone().execute::<Canonical>(ctx)?;
        let codes_canonical = codes.clone().execute::<Canonical>(ctx)?;
        let Canonical::Primitive(values_primitive) = values_canonical else {
            return Ok(None);
        };
        let Canonical::Primitive(codes_primitive) = codes_canonical else {
            return Ok(None);
        };
        let n_values = values_primitive.len();
        let values_validity = values_primitive.validity()?.execute_mask(n_values, ctx)?;
        let codes_validity = codes_primitive
            .validity()?
            .execute_mask(codes_primitive.len(), ctx)?;

        // Dispatch on (codes ptype, values ptype). For each
        // combination we run a fused `acc += values[codes[i]]` loop.
        // Codes are always unsigned (Dict invariant).
        match values_ptype {
            PType::U8 => dispatch_codes_unsigned::<u8>(
                &codes_primitive,
                &values_primitive,
                &codes_validity,
                &values_validity,
            ),
            PType::U16 => dispatch_codes_unsigned::<u16>(
                &codes_primitive,
                &values_primitive,
                &codes_validity,
                &values_validity,
            ),
            PType::U32 => dispatch_codes_unsigned::<u32>(
                &codes_primitive,
                &values_primitive,
                &codes_validity,
                &values_validity,
            ),
            PType::U64 => dispatch_codes_unsigned::<u64>(
                &codes_primitive,
                &values_primitive,
                &codes_validity,
                &values_validity,
            ),
            PType::I8 => dispatch_codes_signed::<i8>(
                &codes_primitive,
                &values_primitive,
                &codes_validity,
                &values_validity,
            ),
            PType::I16 => dispatch_codes_signed::<i16>(
                &codes_primitive,
                &values_primitive,
                &codes_validity,
                &values_validity,
            ),
            PType::I32 => dispatch_codes_signed::<i32>(
                &codes_primitive,
                &values_primitive,
                &codes_validity,
                &values_validity,
            ),
            PType::I64 => dispatch_codes_signed::<i64>(
                &codes_primitive,
                &values_primitive,
                &codes_validity,
                &values_validity,
            ),
            PType::F16 | PType::F32 => dispatch_codes_float::<f32>(
                &codes_primitive,
                &values_primitive,
                &codes_validity,
                &values_validity,
                values_ptype,
            ),
            PType::F64 => dispatch_codes_float::<f64>(
                &codes_primitive,
                &values_primitive,
                &codes_validity,
                &values_validity,
                values_ptype,
            ),
        }
    }
}

// ---- codes-ptype dispatch ---------------------------------------------------

/// Given a known values-dtype path (V), dispatch on the codes' ptype so we
/// can monomorphise the gather-sum loop on both code and value primitive
/// types. The macros below keep the per-codes-ptype call sites short.
macro_rules! dispatch_on_codes {
    ($f:ident, $codes:expr, $values:expr, $cmask:expr, $vmask:expr $(, $extra:expr)*) => {
        match $codes.ptype() {
            PType::U8 => $f::<u8, _>($codes.as_slice::<u8>(), $values, $cmask, $vmask $(, $extra)*),
            PType::U16 => $f::<u16, _>($codes.as_slice::<u16>(), $values, $cmask, $vmask $(, $extra)*),
            PType::U32 => $f::<u32, _>($codes.as_slice::<u32>(), $values, $cmask, $vmask $(, $extra)*),
            PType::U64 => $f::<u64, _>($codes.as_slice::<u64>(), $values, $cmask, $vmask $(, $extra)*),
            _ => Ok(None),
        }
    };
}

fn dispatch_codes_unsigned<V>(
    codes: &PrimitiveArray,
    values: &PrimitiveArray,
    codes_validity: &Mask,
    values_validity: &Mask,
) -> VortexResult<Option<Scalar>>
where
    V: NativePType + Into<u64>,
{
    let values_slice = values.as_slice::<V>();
    dispatch_on_codes!(
        fused_sum_unsigned,
        codes,
        values_slice,
        codes_validity,
        values_validity
    )
}

fn dispatch_codes_signed<V>(
    codes: &PrimitiveArray,
    values: &PrimitiveArray,
    codes_validity: &Mask,
    values_validity: &Mask,
) -> VortexResult<Option<Scalar>>
where
    V: NativePType + Into<i64>,
{
    let values_slice = values.as_slice::<V>();
    dispatch_on_codes!(
        fused_sum_signed,
        codes,
        values_slice,
        codes_validity,
        values_validity
    )
}

fn dispatch_codes_float<V>(
    codes: &PrimitiveArray,
    values: &PrimitiveArray,
    codes_validity: &Mask,
    values_validity: &Mask,
    values_ptype: PType,
) -> VortexResult<Option<Scalar>>
where
    V: NativePType + Into<f64>,
{
    let values_slice = values.as_slice::<V>();
    dispatch_on_codes!(
        fused_sum_float,
        codes,
        values_slice,
        codes_validity,
        values_validity,
        values_ptype
    )
}

// ---- fused inner loops ------------------------------------------------------

/// 4-way unrolled fused gather-sum for floats. Breaks the FP-add
/// dependency chain so the CPU can issue ~4 loads + 4 FP adds per
/// cycle. For ResolutionWidth-shaped inputs (K small, values cached
/// in L1) the per-code cost drops from ~4 cycles (single-accumulator
/// FP add latency) to ~1 cycle.
fn fused_sum_float<C, V>(
    codes: &[C],
    values: &[V],
    codes_validity: &Mask,
    values_validity: &Mask,
    _values_ptype: PType,
) -> VortexResult<Option<Scalar>>
where
    C: NativePType + Copy + TryInto<usize>,
    V: NativePType + Copy + Into<f64>,
{
    let acc = match (codes_validity.bit_buffer(), values_validity.bit_buffer()) {
        (AllOr::None, _) | (_, AllOr::None) => 0.0_f64,
        (AllOr::All, AllOr::All) => fused_sum_float_all(codes, values),
        (AllOr::Some(cm), AllOr::All) => {
            let mut acc = 0.0_f64;
            for i in cm.set_indices() {
                let Ok(idx) = codes[i].try_into() else { continue };
                if idx < values.len() {
                    acc += values[idx].into();
                }
            }
            acc
        }
        (AllOr::All, AllOr::Some(vm)) => {
            let mut acc = 0.0_f64;
            for &c in codes {
                let Ok(idx) = c.try_into() else { continue };
                if idx < values.len() && vm.value(idx) {
                    acc += values[idx].into();
                }
            }
            acc
        }
        (AllOr::Some(cm), AllOr::Some(vm)) => {
            let mut acc = 0.0_f64;
            for i in cm.set_indices() {
                let Ok(idx) = codes[i].try_into() else { continue };
                if idx < values.len() && vm.value(idx) {
                    acc += values[idx].into();
                }
            }
            acc
        }
    };
    Ok(Some(Scalar::primitive(acc, Nullability::Nullable)))
}

#[inline(always)]
fn fused_sum_float_all<C, V>(codes: &[C], values: &[V]) -> f64
where
    C: NativePType + Copy + TryInto<usize>,
    V: NativePType + Copy + Into<f64>,
{
    let n = codes.len();
    let n_values = values.len();
    let mut a0 = 0.0_f64;
    let mut a1 = 0.0_f64;
    let mut a2 = 0.0_f64;
    let mut a3 = 0.0_f64;
    let chunks = n / 4;
    for chunk in 0..chunks {
        let base = chunk * 4;
        // Trust Dict invariant: 0 <= codes[i] < values.len(). If the
        // input violates the invariant `try_into` may fail or the
        // bounds check below clamps; either way we don't OOB.
        let i0: usize = codes[base].try_into().unwrap_or(0);
        let i1: usize = codes[base + 1].try_into().unwrap_or(0);
        let i2: usize = codes[base + 2].try_into().unwrap_or(0);
        let i3: usize = codes[base + 3].try_into().unwrap_or(0);
        if i0 < n_values {
            a0 += values[i0].into();
        }
        if i1 < n_values {
            a1 += values[i1].into();
        }
        if i2 < n_values {
            a2 += values[i2].into();
        }
        if i3 < n_values {
            a3 += values[i3].into();
        }
    }
    let mut acc = a0 + a1 + a2 + a3;
    for i in (chunks * 4)..n {
        let Ok(idx) = codes[i].try_into() else { continue };
        if idx < n_values {
            acc += values[idx].into();
        }
    }
    acc
}

fn fused_sum_unsigned<C, V>(
    codes: &[C],
    values: &[V],
    codes_validity: &Mask,
    values_validity: &Mask,
) -> VortexResult<Option<Scalar>>
where
    C: NativePType + Copy + TryInto<usize>,
    V: NativePType + Copy + Into<u64>,
{
    let (acc, overflowed) = match (codes_validity.bit_buffer(), values_validity.bit_buffer()) {
        (AllOr::None, _) | (_, AllOr::None) => (0u64, false),
        (AllOr::All, AllOr::All) => fused_sum_unsigned_all(codes, values),
        (codes_state, values_state) => {
            let mut acc = 0u64;
            let iter = code_iter(codes, codes_state);
            let mut overflowed = false;
            for idx in iter {
                if !value_valid(&values_state, idx) {
                    continue;
                }
                if idx >= values.len() {
                    continue;
                }
                match acc.checked_add(values[idx].into()) {
                    Some(next) => acc = next,
                    None => {
                        overflowed = true;
                        break;
                    }
                }
            }
            (acc, overflowed)
        }
    };
    let scalar = if overflowed {
        Scalar::null(DType::Primitive(PType::U64, Nullability::Nullable))
    } else {
        Scalar::primitive(acc, Nullability::Nullable)
    };
    Ok(Some(scalar))
}

#[inline(always)]
fn fused_sum_unsigned_all<C, V>(codes: &[C], values: &[V]) -> (u64, bool)
where
    C: NativePType + Copy + TryInto<usize>,
    V: NativePType + Copy + Into<u64>,
{
    let n_values = values.len();
    let mut acc = 0u64;
    for &c in codes {
        let idx: usize = c.try_into().unwrap_or(0);
        if idx >= n_values {
            continue;
        }
        match acc.checked_add(values[idx].into()) {
            Some(next) => acc = next,
            None => return (0, true),
        }
    }
    (acc, false)
}

fn fused_sum_signed<C, V>(
    codes: &[C],
    values: &[V],
    codes_validity: &Mask,
    values_validity: &Mask,
) -> VortexResult<Option<Scalar>>
where
    C: NativePType + Copy + TryInto<usize>,
    V: NativePType + Copy + Into<i64>,
{
    let (acc, overflowed) = match (codes_validity.bit_buffer(), values_validity.bit_buffer()) {
        (AllOr::None, _) | (_, AllOr::None) => (0i64, false),
        (AllOr::All, AllOr::All) => fused_sum_signed_all(codes, values),
        (codes_state, values_state) => {
            let mut acc = 0i64;
            let iter = code_iter(codes, codes_state);
            let mut overflowed = false;
            for idx in iter {
                if !value_valid(&values_state, idx) {
                    continue;
                }
                if idx >= values.len() {
                    continue;
                }
                match acc.checked_add(values[idx].into()) {
                    Some(next) => acc = next,
                    None => {
                        overflowed = true;
                        break;
                    }
                }
            }
            (acc, overflowed)
        }
    };
    let scalar = if overflowed {
        Scalar::null(DType::Primitive(PType::I64, Nullability::Nullable))
    } else {
        Scalar::primitive(acc, Nullability::Nullable)
    };
    Ok(Some(scalar))
}

#[inline(always)]
fn fused_sum_signed_all<C, V>(codes: &[C], values: &[V]) -> (i64, bool)
where
    C: NativePType + Copy + TryInto<usize>,
    V: NativePType + Copy + Into<i64>,
{
    let n_values = values.len();
    let mut acc = 0i64;
    for &c in codes {
        let idx: usize = c.try_into().unwrap_or(0);
        if idx >= n_values {
            continue;
        }
        match acc.checked_add(values[idx].into()) {
            Some(next) => acc = next,
            None => return (0, true),
        }
    }
    (acc, false)
}

// ---- validity helpers --------------------------------------------------------

/// Iterator over code positions to visit, given a codes-validity state.
fn code_iter<'a, C>(
    codes: &'a [C],
    state: AllOr<&'a vortex_buffer::BitBuffer>,
) -> Box<dyn Iterator<Item = usize> + 'a>
where
    C: NativePType + Copy + TryInto<usize>,
{
    match state {
        AllOr::All => Box::new((0..codes.len()).filter_map(|i| codes[i].try_into().ok())),
        AllOr::None => Box::new(std::iter::empty()),
        AllOr::Some(mask) => Box::new(
            mask.set_indices()
                .filter_map(move |i| codes[i].try_into().ok()),
        ),
    }
}

fn value_valid(state: &AllOr<&vortex_buffer::BitBuffer>, index: usize) -> bool {
    match state {
        AllOr::All => true,
        AllOr::None => false,
        AllOr::Some(m) => m.value(index),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions;
    use vortex_array::arrays::DictArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_session::VortexSession;

    fn dict<C, V>(codes: &[C], values: &[V]) -> DictArray
    where
        C: NativePType,
        V: NativePType,
    {
        let codes_arr =
            PrimitiveArray::new(Buffer::copy_from(codes), Validity::NonNullable).into_array();
        let values_arr =
            PrimitiveArray::new(Buffer::copy_from(values), Validity::NonNullable).into_array();
        DictArray::try_new(codes_arr, values_arr).expect("dict")
    }

    fn run_kernel(dict_array: DictArray) -> Scalar {
        use vortex::VortexSessionDefault;
        let session = VortexSession::default();
        let mut exec = session.create_execution_ctx();
        let agg = Sum.bind(EmptyOptions);
        DictSumKernel
            .aggregate(&agg, &dict_array.into_array(), &mut exec)
            .expect("kernel ok")
            .expect("kernel produced partial")
    }

    #[test]
    fn signed_i64_sum() {
        // codes=[0,1,2,3,0,1] values=[10,20,30,40]
        // sum = 10+20+30+40+10+20 = 130
        let d = dict::<u32, i64>(&[0, 1, 2, 3, 0, 1], &[10, 20, 30, 40]);
        let scalar = run_kernel(d);
        let v = scalar.as_primitive().typed_value::<i64>().expect("i64");
        assert_eq!(v, 130);
    }

    #[test]
    fn unsigned_u32_sum() {
        // codes=[0,0,1] values=[100, 200]
        // sum = 100+100+200 = 400 (u64 partial)
        let d = dict::<u32, u32>(&[0, 0, 1], &[100, 200]);
        let scalar = run_kernel(d);
        let v = scalar.as_primitive().typed_value::<u64>().expect("u64");
        assert_eq!(v, 400);
    }

    #[test]
    fn float_f64_sum() {
        // codes=[0,1,1] values=[1.5, 2.5]
        // sum = 1.5+2.5+2.5 = 6.5
        let d = dict::<u32, f64>(&[0, 1, 1], &[1.5, 2.5]);
        let scalar = run_kernel(d);
        let v = scalar.as_primitive().typed_value::<f64>().expect("f64");
        assert!((v - 6.5).abs() < 1e-9);
    }

    #[test]
    fn unreferenced_values_excluded() {
        // codes=[0] values=[1, 999]; only the referenced value contributes.
        let d = dict::<u32, i64>(&[0], &[1, 999]);
        let scalar = run_kernel(d);
        let v = scalar.as_primitive().typed_value::<i64>().expect("i64");
        assert_eq!(v, 1);
    }

    #[test]
    fn unrolled_float_path_matches() {
        // Exercise the 4-way unrolled all-valid float path with a
        // length that's not a multiple of 4.
        let codes: Vec<u8> = (0..17).map(|i| (i % 3) as u8).collect();
        let values: Vec<f64> = vec![1.0, 10.0, 100.0];
        // Expected: count code by occurrence × value.
        // 17 codes, indices 0..2 cycling. Count: 0→6, 1→6, 2→5.
        // sum = 6*1 + 6*10 + 5*100 = 6 + 60 + 500 = 566
        let d = dict::<u8, f64>(&codes, &values);
        let scalar = run_kernel(d);
        let v = scalar.as_primitive().typed_value::<f64>().expect("f64");
        assert!((v - 566.0).abs() < 1e-9);
    }
}
