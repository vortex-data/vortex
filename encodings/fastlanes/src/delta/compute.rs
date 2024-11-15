use std::cmp::min;

use vortex_array::compute::unary::ScalarAtFn;
use vortex_array::compute::{slice, ArrayCompute, SliceFn};
use vortex_array::{ArrayData, IntoArrayData, IntoArrayVariant};
use vortex_error::{VortexExpect, VortexResult};
use vortex_scalar::Scalar;

use crate::DeltaArray;

impl ArrayCompute for DeltaArray {
    fn scalar_at(&self) -> Option<&dyn ScalarAtFn> {
        Some(self)
    }

    fn slice(&self) -> Option<&dyn SliceFn> {
        Some(self)
    }
}

impl ScalarAtFn for DeltaArray {
    fn scalar_at(&self, index: usize) -> VortexResult<Scalar> {
        let decompressed = slice(self, index, index + 1)?.into_primitive()?;
        ScalarAtFn::scalar_at(&decompressed, 0)
    }

    fn scalar_at_unchecked(&self, index: usize) -> Scalar {
        let decompressed = slice(self, index, index + 1)
            .vortex_expect("DeltaArray slice for one value should work")
            .into_primitive()
            .vortex_expect("Converting slice into primitive should work");
        ScalarAtFn::scalar_at_unchecked(&decompressed, 0)
    }
}

impl SliceFn for DeltaArray {
    fn slice(&self, start: usize, stop: usize) -> VortexResult<ArrayData> {
        let physical_start = start + self.offset();
        let physical_stop = stop + self.offset();

        let start_chunk = physical_start / 1024;
        let stop_chunk = physical_stop.div_ceil(1024);

        let bases = self.bases();
        let deltas = self.deltas();
        let validity = self.validity();
        let lanes = self.lanes();

        let new_bases = slice(
            bases,
            min(start_chunk * lanes, self.bases_len()),
            min(stop_chunk * lanes, self.bases_len()),
        )?;

        let new_deltas = slice(
            deltas,
            min(start_chunk * 1024, self.deltas_len()),
            min(stop_chunk * 1024, self.deltas_len()),
        )?;

        let new_validity = validity.slice(start, stop)?;

        let logical_len = stop - start;

        let arr = DeltaArray::try_new(
            new_bases,
            new_deltas,
            new_validity,
            physical_start % 1024,
            logical_len,
        )?;

        Ok(arr.into_array())
    }
}

#[cfg(test)]
mod test {
    use vortex_array::compute::slice;
    use vortex_array::compute::unary::{scalar_at, scalar_at_unchecked};
    use vortex_array::IntoArrayVariant;
    use vortex_error::VortexError;

    use super::*;

    #[test]
    fn test_scalar_at_non_jagged_array() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect())
            .unwrap()
            .into_array();

        assert_eq!(scalar_at(&delta, 0).unwrap(), 0_u32.into());
        assert_eq!(scalar_at_unchecked(&delta, 0), 0_u32.into());

        assert_eq!(scalar_at(&delta, 1).unwrap(), 1_u32.into());
        assert_eq!(scalar_at_unchecked(&delta, 1), 1_u32.into());

        assert_eq!(scalar_at(&delta, 10).unwrap(), 10_u32.into());
        assert_eq!(scalar_at_unchecked(&delta, 10), 10_u32.into());

        assert_eq!(scalar_at(&delta, 1023).unwrap(), 1023_u32.into());
        assert_eq!(scalar_at_unchecked(&delta, 1023), 1023_u32.into());

        assert_eq!(scalar_at(&delta, 1024).unwrap(), 1024_u32.into());
        assert_eq!(scalar_at_unchecked(&delta, 1024), 1024_u32.into());

        assert_eq!(scalar_at(&delta, 1025).unwrap(), 1025_u32.into());
        assert_eq!(scalar_at_unchecked(&delta, 1025), 1025_u32.into());

        assert_eq!(scalar_at(&delta, 2047).unwrap(), 2047_u32.into());
        assert_eq!(scalar_at_unchecked(&delta, 2047), 2047_u32.into());

        assert!(matches!(
            scalar_at(&delta, 2048),
            Err(VortexError::OutOfBounds(2048, 0, 2048, _))
        ));

        assert!(matches!(
            scalar_at(&delta, 2049),
            Err(VortexError::OutOfBounds(2049, 0, 2048, _))
        ));
    }

    #[test]
    fn test_scalar_at_jagged_array() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect())
            .unwrap()
            .into_array();

        assert_eq!(scalar_at(&delta, 0).unwrap(), 0_u32.into());
        assert_eq!(scalar_at_unchecked(&delta, 0), 0_u32.into());

        assert_eq!(scalar_at(&delta, 1).unwrap(), 1_u32.into());
        assert_eq!(scalar_at_unchecked(&delta, 1), 1_u32.into());

        assert_eq!(scalar_at(&delta, 10).unwrap(), 10_u32.into());
        assert_eq!(scalar_at_unchecked(&delta, 10), 10_u32.into());

        assert_eq!(scalar_at(&delta, 1023).unwrap(), 1023_u32.into());
        assert_eq!(scalar_at_unchecked(&delta, 1023), 1023_u32.into());

        assert_eq!(scalar_at(&delta, 1024).unwrap(), 1024_u32.into());
        assert_eq!(scalar_at_unchecked(&delta, 1024), 1024_u32.into());

        assert_eq!(scalar_at(&delta, 1025).unwrap(), 1025_u32.into());
        assert_eq!(scalar_at_unchecked(&delta, 1025), 1025_u32.into());

        assert_eq!(scalar_at(&delta, 1999).unwrap(), 1999_u32.into());
        assert_eq!(scalar_at_unchecked(&delta, 1999), 1999_u32.into());

        assert!(matches!(
            scalar_at(&delta, 2000),
            Err(VortexError::OutOfBounds(2000, 0, 2000, _))
        ));

        assert!(matches!(
            scalar_at(&delta, 2001),
            Err(VortexError::OutOfBounds(2001, 0, 2000, _))
        ));
    }

    #[test]
    fn test_slice_non_jagged_array_first_chunk_of_two() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        assert_eq!(
            SliceFn::slice(&delta, 10, 250)
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u32>(),
            (10u32..250).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slice_non_jagged_array_second_chunk_of_two() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        assert_eq!(
            SliceFn::slice(&delta, 1024 + 10, 1024 + 250)
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u32>(),
            ((1024 + 10u32)..(1024 + 250)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slice_non_jagged_array_span_two_chunks_chunk_of_two() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        assert_eq!(
            SliceFn::slice(&delta, 1000, 1048)
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u32>(),
            (1000u32..1048).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slice_non_jagged_array_span_two_chunks_chunk_of_four() {
        let delta = DeltaArray::try_from_vec((0u32..4096).collect()).unwrap();

        assert_eq!(
            SliceFn::slice(&delta, 2040, 2050)
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u32>(),
            (2040u32..2050).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slice_non_jagged_array_whole() {
        let delta = DeltaArray::try_from_vec((0u32..4096).collect()).unwrap();

        assert_eq!(
            SliceFn::slice(&delta, 0, 4096)
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u32>(),
            (0u32..4096).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slice_non_jagged_array_empty() {
        let delta = DeltaArray::try_from_vec((0u32..4096).collect()).unwrap();

        assert_eq!(
            SliceFn::slice(&delta, 0, 0)
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u32>(),
            Vec::<u32>::new(),
        );

        assert_eq!(
            SliceFn::slice(&delta, 4096, 4096)
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u32>(),
            Vec::<u32>::new(),
        );

        assert_eq!(
            SliceFn::slice(&delta, 1024, 1024)
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u32>(),
            Vec::<u32>::new(),
        );
    }

    #[test]
    fn test_slice_jagged_array_second_chunk_of_two() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect()).unwrap();

        assert_eq!(
            SliceFn::slice(&delta, 1024 + 10, 1024 + 250)
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u32>(),
            ((1024 + 10u32)..(1024 + 250)).collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_slice_jagged_array_empty() {
        let delta = DeltaArray::try_from_vec((0u32..4000).collect()).unwrap();

        assert_eq!(
            SliceFn::slice(&delta, 0, 0)
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u32>(),
            Vec::<u32>::new(),
        );

        assert_eq!(
            SliceFn::slice(&delta, 4096, 4096)
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u32>(),
            Vec::<u32>::new(),
        );

        assert_eq!(
            SliceFn::slice(&delta, 1024, 1024)
                .unwrap()
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u32>(),
            Vec::<u32>::new(),
        );
    }

    #[test]
    fn test_slice_of_slice_of_non_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        let sliced = SliceFn::slice(&delta, 10, 1013).unwrap();
        let sliced_again = slice(sliced, 0, 2).unwrap();

        assert_eq!(
            sliced_again
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u32>(),
            vec![10, 11]
        );
    }

    #[test]
    fn test_slice_of_slice_of_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect()).unwrap();

        let sliced = SliceFn::slice(&delta, 10, 1013).unwrap();
        let sliced_again = slice(sliced, 0, 2).unwrap();

        assert_eq!(
            sliced_again
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u32>(),
            vec![10, 11]
        );
    }

    #[test]
    fn test_slice_of_slice_second_chunk_of_non_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        let sliced = SliceFn::slice(&delta, 1034, 1050).unwrap();
        let sliced_again = slice(sliced, 0, 2).unwrap();

        assert_eq!(
            sliced_again
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u32>(),
            vec![1034, 1035]
        );
    }

    #[test]
    fn test_slice_of_slice_second_chunk_of_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect()).unwrap();

        let sliced = SliceFn::slice(&delta, 1034, 1050).unwrap();
        let sliced_again = slice(sliced, 0, 2).unwrap();

        assert_eq!(
            sliced_again
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u32>(),
            vec![1034, 1035]
        );
    }

    #[test]
    fn test_slice_of_slice_spanning_two_chunks_of_non_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2048).collect()).unwrap();

        let sliced = SliceFn::slice(&delta, 1010, 1050).unwrap();
        let sliced_again = slice(sliced, 5, 20).unwrap();

        assert_eq!(
            sliced_again
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u32>(),
            (1015..1030).collect::<Vec<_>>(),
        );
    }

    #[test]
    fn test_slice_of_slice_spanning_two_chunks_of_jagged() {
        let delta = DeltaArray::try_from_vec((0u32..2000).collect()).unwrap();

        let sliced = SliceFn::slice(&delta, 1010, 1050).unwrap();
        let sliced_again = slice(sliced, 5, 20).unwrap();

        assert_eq!(
            sliced_again
                .into_primitive()
                .unwrap()
                .maybe_null_slice::<u32>(),
            (1015..1030).collect::<Vec<_>>(),
        );
    }
}
