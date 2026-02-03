// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::FilterVTable;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::SliceVTable;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::matchers::Exact;
use vortex_error::VortexResult;

use crate::ALPArray;
use crate::ALPVTable;

pub(super) const PARENT_KERNELS: ParentKernelSet<ALPVTable> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&ALPFilterKernel),
    ParentKernelSet::lift(&ALPSliceKernel),
]);

#[derive(Debug)]
struct ALPFilterKernel;

impl ExecuteParentKernel<ALPVTable> for ALPFilterKernel {
    type Parent = Exact<FilterVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::new()
    }

    fn execute_parent(
        &self,
        array: &ALPArray,
        parent: &FilterArray,
        _child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let mask = parent.filter_mask();
        let patches = array
            .patches()
            .map(|p| p.filter(mask))
            .transpose()?
            .flatten();

        // SAFETY: filtering the values does not change correctness
        let filtered_alp = unsafe {
            ALPArray::new_unchecked(
                array.encoded().filter(mask.clone())?,
                array.exponents(),
                patches,
                array.dtype().clone(),
            )
        }
        .into_array();

        Ok(Some(filtered_alp))
    }
}

/// CPU-only slice kernel that performs slicing of the buffer and any patches.
/// Note that this triggers compute (binary searching Patches) which we cannot do when the
/// buffers live in GPU memory.
#[derive(Debug)]
struct ALPSliceKernel;

impl ExecuteParentKernel<ALPVTable> for ALPSliceKernel {
    type Parent = Exact<SliceVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::new()
    }

    fn execute_parent(
        &self,
        array: &ALPArray,
        parent: &SliceArray,
        _child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let range = parent.slice_range().clone();
        let sliced_alp = ALPArray::new(
            array.encoded().slice(range.clone())?,
            array.exponents(),
            array
                .patches()
                .map(|p| p.slice(range))
                .transpose()?
                .flatten(),
        )
        .into_array();

        Ok(Some(sliced_alp))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::IntoArray;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::conformance::filter::test_filter_conformance;
    use vortex_buffer::buffer;

    use crate::alp_encode;

    #[rstest]
    #[case(buffer![1.23f32, 4.56, 7.89, 10.11, 12.13].into_array())]
    #[case(buffer![100.1f64, 200.2, 300.3, 400.4, 500.5].into_array())]
    #[case(PrimitiveArray::from_option_iter([Some(1.1f32), None, Some(2.2), Some(3.3), None]).into_array())]
    #[case(buffer![42.42f64].into_array())]
    #[case(buffer![
        1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
        11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0
    ].into_array())]
    fn test_filter_alp_conformance(#[case] array: ArrayRef) {
        let alp = alp_encode(&array.to_primitive(), None).unwrap();
        test_filter_conformance(alp.as_ref());
    }
}
