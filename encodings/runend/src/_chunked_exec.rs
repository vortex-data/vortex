// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `RunEnd<Primitive>` chunked decoder + registration with `vortex_array::_chunked_exec`.
//!
//! Lives in this crate because `RunEnd` is defined here; the producer itself
//! ([`vortex_array::_chunked_exec::primitive::RunEndPrimitiveProducer`]) is generic and
//! lives in `vortex-array`. The kernel here just bridges Vortex's encoding view to the
//! generic streaming primitive.

use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::VTable;
use vortex_array::_chunked_exec::primitive::PrimitiveChunkKernel;
use vortex_array::_chunked_exec::primitive::PrimitiveChunkKernelDispatcher;
use vortex_array::_chunked_exec::primitive::PrimitiveChunkProducer;
use vortex_array::_chunked_exec::primitive::build_runend_producer;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::array::RunEndArrayExt as _;

/// `RunEnd<Primitive>` chunked kernel.
pub struct RunEndKernel<T: NativePType> {
    _marker: std::marker::PhantomData<fn() -> T>,
}

impl<T: NativePType> RunEndKernel<T> {
    /// Construct an empty kernel marker.
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: NativePType> Default for RunEndKernel<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: NativePType> PrimitiveChunkKernel<T> for RunEndKernel<T> {
    fn build(
        &self,
        array: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Box<dyn PrimitiveChunkProducer<T>>>> {
        let Some(re) = array.as_opt::<RunEnd>() else {
            return Ok(None);
        };
        if !matches!(array.dtype().nullability(), Nullability::NonNullable) {
            return Ok(None);
        }
        let values = re.values();
        let ends = re.ends();
        if !matches!(values.dtype().nullability(), Nullability::NonNullable)
            || !matches!(ends.dtype().nullability(), Nullability::NonNullable)
        {
            return Ok(None);
        }
        let DType::Primitive(values_ptype, _) = *values.dtype() else {
            return Ok(None);
        };
        if values_ptype != T::PTYPE {
            return Ok(None);
        }
        let offset = re.offset();
        let len = array.len();
        let values_canonical = values.clone().execute::<PrimitiveArray>(ctx)?;
        let ends_canonical = ends.clone().execute::<PrimitiveArray>(ctx)?;
        Ok(Some(build_runend_producer::<T>(
            values_canonical,
            ends_canonical,
            offset,
            len,
        )?))
    }
}

/// Register the `RunEnd` chunk kernel onto `dispatcher` for every primitive output type
/// that we currently care about.
///
/// This is the runend-side equivalent of `vortex_array::_chunked_exec::register_defaults`;
/// downstream crates that want the fused `Dict<RunEnd<P>>` path should call both.
pub fn register_chunk_kernels(dispatcher: &mut PrimitiveChunkKernelDispatcher) {
    macro_rules! register_all_for {
        ($($T:ty),*) => {
            $(
                dispatcher.register::<$T>(RunEnd.id(), Arc::new(RunEndKernel::<$T>::new()));
            )*
        };
    }
    register_all_for!(u8, u16, u32, u64, i8, i16, i32, i64, f32, f64);
}

#[cfg(test)]
mod tests {
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::_chunked_exec::primitive::decode_to_buffer;
    use vortex_array::_chunked_exec::primitive::default_dispatcher;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;

    use super::register_chunk_kernels;
    use crate::RunEnd;

    fn ctx() -> ExecutionCtx {
        LEGACY_SESSION.create_execution_ctx()
    }

    fn dispatcher() -> vortex_array::_chunked_exec::primitive::PrimitiveChunkKernelDispatcher {
        let mut d = default_dispatcher();
        register_chunk_kernels(&mut d);
        d
    }

    /// Construct a RunEnd<Primitive> from an iterator of values; useful in tests.
    fn make_runend_i32(values: &[i32]) -> VortexResult<vortex_array::ArrayRef> {
        let prim = PrimitiveArray::new(
            Buffer::<i32>::from_iter(values.iter().copied()),
            Validity::NonNullable,
        );
        let re = RunEnd::encode(prim.into_array(), &mut ctx())?;
        Ok(re.into_array())
    }

    #[test]
    fn runend_chunked() -> VortexResult<()> {
        let mut values = Vec::with_capacity(4000);
        values.extend(std::iter::repeat(1i32).take(100));
        values.extend(std::iter::repeat(2i32).take(150));
        values.extend(std::iter::repeat(3i32).take(850));
        values.extend(std::iter::repeat(4i32).take(2900));
        let re = make_runend_i32(&values)?;
        let buf = decode_to_buffer::<i32>(re, &dispatcher(), &mut ctx())?;
        assert_eq!(buf.as_slice(), values.as_slice());
        Ok(())
    }

    #[test]
    fn fused_dict_runend_chunked() -> VortexResult<()> {
        // Inner dictionary VALUES are RunEnd-encoded (so the dict has 12 logical entries).
        let inner_values = vec![100i32, 100, 100, 200, 200, 200, 300, 300, 300, 400, 400, 400];
        let inner_re = make_runend_i32(&inner_values)?;

        let codes_vec: Vec<u8> = (0..8192u32).map(|i| (i % 12) as u8).collect();
        let codes_arr = PrimitiveArray::new(
            Buffer::<u8>::from_iter(codes_vec.iter().copied()),
            Validity::NonNullable,
        );
        let dict = DictArray::try_new(codes_arr.into_array(), inner_re)?;

        let buf = decode_to_buffer::<i32>(dict.into_array(), &dispatcher(), &mut ctx())?;
        let expected: Vec<i32> = codes_vec
            .iter()
            .map(|c| inner_values[*c as usize])
            .collect();
        assert_eq!(buf.as_slice(), expected.as_slice());
        Ok(())
    }

    #[test]
    fn runend_sliced_chunked() -> VortexResult<()> {
        let mut values = Vec::with_capacity(2000);
        for run_idx in 0..20 {
            values.extend(std::iter::repeat(run_idx as i32 + 1).take(100));
        }
        let re_full = make_runend_i32(&values)?;
        let sliced = re_full.slice(50..1500)?;
        let buf = decode_to_buffer::<i32>(sliced, &dispatcher(), &mut ctx())?;
        assert_eq!(buf.as_slice(), &values[50..1500]);
        Ok(())
    }

}
