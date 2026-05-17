// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Sort-merge join built on [`BinaryKernel`].
//!
//! Output is a `StructArray` of two index columns
//! `(left_idx, right_idx)`. Duplicates fan out via Cartesian product
//! within each run of equal keys.
//!
//! The Constant×Constant kernel emits chunked index columns
//! (`ChunkedArray<ConstantArray>` + `ChunkedArray<ProgressionArray>`)
//! to keep the L*R Cartesian in O(L+R) memory.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

use crate::binary_kernel::BinaryKernel;
use crate::binary_kernel::BinaryKernelSet;
use crate::progression::Progression;

fn smj_output(left_idx: ArrayRef, right_idx: ArrayRef) -> ArrayRef {
    StructArray::from_fields(&[("left_idx", left_idx), ("right_idx", right_idx)])
        .expect("struct array")
        .into_array()
}

#[inline(never)]
fn smj_primitive(l: &[u64], r: &[u64]) -> ArrayRef {
    let cap = l.len().min(r.len());
    let mut left_idx = Vec::<u32>::with_capacity(cap);
    let mut right_idx = Vec::<u32>::with_capacity(cap);
    let (mut i, mut j) = (0usize, 0usize);
    while i < l.len() && j < r.len() {
        if l[i] < r[j] {
            i += 1;
        } else if l[i] > r[j] {
            j += 1;
        } else {
            let key = l[i];
            let i0 = i;
            while i < l.len() && l[i] == key {
                i += 1;
            }
            let j0 = j;
            while j < r.len() && r[j] == key {
                j += 1;
            }
            for li in i0..i {
                for rj in j0..j {
                    left_idx.push(li as u32);
                    right_idx.push(rj as u32);
                }
            }
        }
    }
    let l_arr =
        PrimitiveArray::new(Buffer::<u32>::copy_from(&left_idx), Validity::NonNullable)
            .into_array();
    let r_arr =
        PrimitiveArray::new(Buffer::<u32>::copy_from(&right_idx), Validity::NonNullable)
            .into_array();
    smj_output(l_arr, r_arr)
}

#[derive(Debug)]
pub struct PrimitivePrimitiveSmj;

impl BinaryKernel<Primitive, Primitive> for PrimitivePrimitiveSmj {
    fn execute(
        &self,
        left: ArrayView<'_, Primitive>,
        right: ArrayView<'_, Primitive>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(smj_primitive(
            left.as_slice::<u64>(),
            right.as_slice::<u64>(),
        )))
    }
}

pub const PRIM_PRIM_SMJ: BinaryKernelSet<Primitive, Primitive> =
    BinaryKernelSet::new(&[BinaryKernelSet::lift(&PrimitivePrimitiveSmj)]);

/// Encoding-aware Cartesian: L*R pairs in O(L+R) memory.
///
/// `left_idx[i*R + j] = i` becomes L chunks of `ConstantArray(i, R)`.
/// `right_idx[i*R + j] = j` becomes L chunks of `Progression(0, 1, R)`.
#[inline(never)]
fn smj_constant_constant(lv: u64, ln: usize, rv: u64, rn: usize) -> ArrayRef {
    if lv != rv || ln == 0 || rn == 0 {
        let empty = ConstantArray::new(0u64, 0).into_array();
        return smj_output(empty.clone(), empty);
    }
    let dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
    let left_chunks: Vec<ArrayRef> = (0..ln)
        .map(|i| ConstantArray::new(i as u64, rn).into_array())
        .collect();
    let right_chunks: Vec<ArrayRef> = (0..ln)
        .map(|_| Progression::new(0, 1, rn).into_array())
        .collect();
    let left_idx = ChunkedArray::try_new(left_chunks, dtype.clone())
        .expect("chunked left")
        .into_array();
    let right_idx = ChunkedArray::try_new(right_chunks, dtype)
        .expect("chunked right")
        .into_array();
    smj_output(left_idx, right_idx)
}

#[derive(Debug)]
pub struct ConstantConstantSmj;

impl BinaryKernel<Constant, Constant> for ConstantConstantSmj {
    fn execute(
        &self,
        left: ArrayView<'_, Constant>,
        right: ArrayView<'_, Constant>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let lv: u64 = u64::try_from(left.scalar()).expect("u64");
        let rv: u64 = u64::try_from(right.scalar()).expect("u64");
        Ok(Some(smj_constant_constant(
            lv,
            left.array().len(),
            rv,
            right.array().len(),
        )))
    }
}

pub const CONST_CONST_SMJ: BinaryKernelSet<Constant, Constant> =
    BinaryKernelSet::new(&[BinaryKernelSet::lift(&ConstantConstantSmj)]);

/// Row-by-row SMJ baseline via the polymorphic `scalar_at`.
#[inline(never)]
#[allow(deprecated)]
pub fn smj_naive(l: &ArrayRef, r: &ArrayRef) -> ArrayRef {
    let lv: Vec<u64> = (0..l.len())
        .map(|i| u64::try_from(&l.scalar_at(i).expect("l")).expect("u64"))
        .collect();
    let rv: Vec<u64> = (0..r.len())
        .map(|i| u64::try_from(&r.scalar_at(i).expect("r")).expect("u64"))
        .collect();
    smj_primitive(&lv, &rv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;

    #[test]
    fn disjoint() {
        assert_eq!(smj_primitive(&[1, 3, 5], &[2, 4, 6]).len(), 0);
    }

    #[test]
    fn cartesian_within_run() {
        assert_eq!(smj_primitive(&[5, 5, 5], &[5, 5]).len(), 6);
    }

    #[test]
    fn via_kernel_set() -> VortexResult<()> {
        let l =
            PrimitiveArray::new(Buffer::<u64>::copy_from(&[1u64, 3, 5]), Validity::NonNullable);
        let r =
            PrimitiveArray::new(Buffer::<u64>::copy_from(&[3u64, 5, 7]), Validity::NonNullable);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let out = PRIM_PRIM_SMJ
            .execute(l.as_view(), &r.clone().into_array(), &mut ctx)?
            .expect("Some");
        assert_eq!(out.len(), 2);
        Ok(())
    }

    /// 1M-pair Cartesian materialised flat would be ~8MB; the
    /// encoding-aware output should be much smaller.
    #[test]
    fn constant_cartesian_is_compact() -> VortexResult<()> {
        let l = ConstantArray::new(7u64, 1000);
        let r = ConstantArray::new(7u64, 1000).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let out = CONST_CONST_SMJ
            .execute(l.as_view(), &r, &mut ctx)?
            .expect("Some");
        assert_eq!(out.len(), 1_000_000);
        assert!(out.nbytes() < 1_000_000);
        Ok(())
    }

    #[test]
    fn constant_unequal_is_empty() -> VortexResult<()> {
        let l = ConstantArray::new(1u64, 100);
        let r = ConstantArray::new(2u64, 100).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let out = CONST_CONST_SMJ
            .execute(l.as_view(), &r, &mut ctx)?
            .expect("Some");
        assert_eq!(out.len(), 0);
        Ok(())
    }
}
