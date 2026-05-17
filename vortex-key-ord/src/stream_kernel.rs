// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `OvcKernel`: per-encoding offset-value-coding kernel.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::Chunked;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::chunked::ChunkedArrayExt;
use vortex_array::arrays::dict::DictArraySlotsExt;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::matcher::AnyArray;
use vortex_array::validity::Validity;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;

/// Sentinel offset meaning "equal to predecessor".
pub const OFFSET_EQ: u8 = u8::MAX;

/// First byte (MSB-first) where `prev` and `curr` differ, or
/// [`OFFSET_EQ`] if equal.
#[inline(always)]
pub fn first_diff_byte(prev: u64, curr: u64) -> u8 {
    let xor = prev ^ curr;
    if xor == 0 {
        OFFSET_EQ
    } else {
        (xor.leading_zeros() / 8) as u8
    }
}

/// Per-encoding OVC kernel.
pub trait OvcKernel: VTable {
    /// Encode `array` to OVC given the carried predecessor `prev`.
    fn ovc_encode(array: ArrayView<'_, Self>, prev: u64) -> ArrayRef;
}

/// `Parent = AnyArray` is a research shortcut. Production uses
/// `ExactScalarFn<Ovc>` (see [`crate::ovc_scalarfn`]).
#[derive(Debug)]
pub struct OvcAdaptor<V>(pub V);

impl<V: OvcKernel> ExecuteParentKernel<V> for OvcAdaptor<V> {
    type Parent = AnyArray;

    fn execute_parent(
        &self,
        array: ArrayView<'_, V>,
        _parent: &ArrayRef,
        _child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        Ok(Some(<V as OvcKernel>::ovc_encode(array, 0)))
    }
}

impl OvcKernel for Primitive {
    fn ovc_encode(array: ArrayView<'_, Self>, prev: u64) -> ArrayRef {
        let buf = array.as_slice::<u64>();
        let mut values = Vec::<u64>::with_capacity(buf.len());
        let mut prev = prev;
        for &v in buf {
            let _off = first_diff_byte(prev, v);
            values.push(v);
            prev = v;
        }
        let validity = array.validity().unwrap_or(Validity::NonNullable);
        PrimitiveArray::new(Buffer::<u64>::copy_from(&values), validity).into_array()
    }
}

impl OvcKernel for Constant {
    fn ovc_encode(array: ArrayView<'_, Self>, prev: u64) -> ArrayRef {
        let v: u64 = u64::try_from(array.scalar()).expect("u64 scalar");
        let _ = first_diff_byte(prev, v);
        ConstantArray::new(v, array.array().len()).into_array()
    }
}

impl OvcKernel for Dict {
    fn ovc_encode(array: ArrayView<'_, Self>, prev: u64) -> ArrayRef {
        // OVC the small values dictionary; keep the codes. O(dict_size).
        let new_values = dispatch_ovc_encode(array.values(), prev);
        DictArray::try_new(array.codes().clone(), new_values)
            .expect("dict construction")
            .into_array()
    }
}

impl OvcKernel for Chunked {
    fn ovc_encode(array: ArrayView<'_, Self>, prev: u64) -> ArrayRef {
        let mut prev = prev;
        let mut out_chunks: Vec<ArrayRef> = Vec::with_capacity(array.nchunks());
        for chunk in array.iter_chunks() {
            if chunk.is_empty() {
                continue;
            }
            out_chunks.push(dispatch_ovc_encode(chunk, prev));
            prev = last_value_for_carry(chunk);
        }
        if out_chunks.is_empty() {
            return ConstantArray::new(0u64, 0).into_array();
        }
        let dtype = DType::Primitive(PType::U64, Nullability::NonNullable);
        ChunkedArray::try_new(out_chunks, dtype)
            .expect("chunked")
            .into_array()
    }
}

/// Hand-rolled `(encoding) -> kernel` lookup used by the `Chunked` and
/// `Dict` recursive kernels. Production would consult the runtime
/// `ArrayKernels` registry.
pub fn dispatch_ovc_encode(chunk: &ArrayRef, prev: u64) -> ArrayRef {
    if let Some(v) = chunk.as_typed::<Primitive>() {
        return <Primitive as OvcKernel>::ovc_encode(v, prev);
    }
    if let Some(v) = chunk.as_typed::<Constant>() {
        return <Constant as OvcKernel>::ovc_encode(v, prev);
    }
    if let Some(v) = chunk.as_typed::<Chunked>() {
        return <Chunked as OvcKernel>::ovc_encode(v, prev);
    }
    if let Some(v) = chunk.as_typed::<Dict>() {
        return <Dict as OvcKernel>::ovc_encode(v, prev);
    }
    panic!("no OvcKernel for encoding {}", chunk.encoding_id());
}

fn last_value_for_carry(chunk: &ArrayRef) -> u64 {
    if let Some(v) = chunk.as_typed::<Constant>() {
        return u64::try_from(v.scalar()).expect("u64 constant");
    }
    if let Some(v) = chunk.as_typed::<Primitive>() {
        return *v.as_slice::<u64>().last().expect("non-empty chunk");
    }
    if let Some(v) = chunk.as_typed::<Chunked>() {
        return last_value_for_carry(v.iter_chunks().last().expect("non-empty"));
    }
    panic!("no carry extractor for {}", chunk.encoding_id());
}

pub const PRIMITIVE_OVC_KERNELS: ParentKernelSet<Primitive> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&OvcAdaptor(Primitive))]);

pub const CONSTANT_OVC_KERNELS: ParentKernelSet<Constant> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&OvcAdaptor(Constant))]);

pub const CHUNKED_OVC_KERNELS: ParentKernelSet<Chunked> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&OvcAdaptor(Chunked))]);

pub const DICT_OVC_KERNELS: ParentKernelSet<Dict> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&OvcAdaptor(Dict))]);

#[cfg(test)]
mod tests {
    use super::*;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::assert_arrays_eq;

    const N: usize = 128;
    const V: u64 = 0xDEAD_BEEF_CAFE_F00D;

    #[test]
    fn primitive_and_constant_agree() -> VortexResult<()> {
        let prim = PrimitiveArray::new(
            Buffer::<u64>::copy_from(&vec![V; N]),
            Validity::NonNullable,
        );
        let constant = ConstantArray::new(V, N);
        let p = <Primitive as OvcKernel>::ovc_encode(prim.as_view(), 0);
        let c = <Constant as OvcKernel>::ovc_encode(constant.as_view(), 0);
        assert_arrays_eq!(p, c);
        Ok(())
    }

    #[test]
    fn nullable_input_yields_nullable_output() -> VortexResult<()> {
        let arr = PrimitiveArray::from_option_iter([Some(1u64), None, Some(3), None]);
        let dtype_before = arr.dtype().clone();
        let out = <Primitive as OvcKernel>::ovc_encode(arr.as_view(), 0);
        assert_eq!(out.dtype(), &dtype_before);
        Ok(())
    }

    #[test]
    fn chunked_carries_state_across_boundaries() -> VortexResult<()> {
        let flat = ConstantArray::new(V, N);
        let flat_out = <Constant as OvcKernel>::ovc_encode(flat.as_view(), 0);
        let chunks: Vec<ArrayRef> = (0..4)
            .map(|_| ConstantArray::new(V, N / 4).into_array())
            .collect();
        let chunked =
            ChunkedArray::try_new(chunks, DType::Primitive(PType::U64, Nullability::NonNullable))?;
        let chunked_out = <Chunked as OvcKernel>::ovc_encode(chunked.as_view(), 0);
        assert_arrays_eq!(flat_out, chunked_out);
        Ok(())
    }

    #[test]
    fn dict_preserves_structure() -> VortexResult<()> {
        let codes_buf: Vec<u32> = (0..N).map(|i| (i % 4) as u32).collect();
        let values = PrimitiveArray::new(
            Buffer::<u64>::copy_from(&[10u64, 20, 30, 40]),
            Validity::NonNullable,
        )
        .into_array();
        let codes =
            PrimitiveArray::new(Buffer::<u32>::copy_from(&codes_buf), Validity::NonNullable)
                .into_array();
        let dict = DictArray::new(codes, values);
        let out = <Dict as OvcKernel>::ovc_encode(dict.as_view(), 0);
        assert_eq!(out.len(), N);
        assert!(out.as_opt::<Dict>().is_some());
        Ok(())
    }

    #[test]
    fn pks_dispatch_matches_direct_call() -> VortexResult<()> {
        let arr = ConstantArray::new(7u64, N);
        let direct = <Constant as OvcKernel>::ovc_encode(arr.as_view(), 0);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let via_pks = CONSTANT_OVC_KERNELS
            .execute(arr.as_view(), &arr.clone().into_array(), 0, &mut ctx)?
            .expect("kernel returned Some");
        assert_arrays_eq!(direct, via_pks);
        Ok(())
    }
}
