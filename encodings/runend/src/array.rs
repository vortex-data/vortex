use std::fmt::Debug;
use std::sync::{Arc, RwLock};

use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::{
    SearchSortedSide, scalar_at, search_sorted_usize, search_sorted_usize_many,
};
use vortex_array::stats::StatsSet;
use vortex_array::variants::{BoolArrayTrait, PrimitiveArrayTrait};
use vortex_array::vtable::VTableRef;
use vortex_array::{
    Array, ArrayCanonicalImpl, ArrayImpl, ArrayRef, ArrayStatisticsImpl, ArrayValidityImpl,
    ArrayVariantsImpl, Canonical, Encoding, EncodingId, IntoArray, SerdeMetadata, ToCanonical,
    encoding_ids, try_from_array_ref,
};
use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_error::{VortexExpect as _, VortexResult, vortex_bail};
use vortex_mask::Mask;

use crate::compress::{runend_decode_bools, runend_decode_primitive, runend_encode};
use crate::serde::RunEndMetadata;

#[derive(Clone, Debug)]
pub struct RunEndArray {
    ends: ArrayRef,
    values: ArrayRef,
    offset: usize,
    length: usize,
    stats_set: Arc<RwLock<StatsSet>>,
}

try_from_array_ref!(RunEndArray);

pub struct RunEndEncoding;
impl Encoding for RunEndEncoding {
    const ID: EncodingId = EncodingId::new("vortex.runend", encoding_ids::RUN_END);
    type Array = RunEndArray;
    type Metadata = SerdeMetadata<RunEndMetadata>;
}

impl RunEndArray {
    pub fn try_new(ends: ArrayRef, values: ArrayRef) -> VortexResult<Self> {
        let length = if ends.is_empty() {
            0
        } else {
            scalar_at(&ends, ends.len() - 1)?.as_ref().try_into()?
        };
        Self::with_offset_and_length(ends, values, 0, length)
    }

    pub(crate) fn with_offset_and_length(
        ends: ArrayRef,
        values: ArrayRef,
        offset: usize,
        length: usize,
    ) -> VortexResult<Self> {
        if !matches!(values.dtype(), &DType::Bool(_) | &DType::Primitive(_, _)) {
            vortex_bail!(
                "RunEnd array can only have Bool or Primitive values, {} given",
                values.dtype()
            );
        }

        if offset != 0 {
            let first_run_end: usize = scalar_at(&ends, 0)?.as_ref().try_into()?;
            if first_run_end <= offset {
                vortex_bail!("First run end {first_run_end} must be bigger than offset {offset}");
            }
        }

        if !ends.dtype().is_unsigned_int() || ends.dtype().is_nullable() {
            vortex_bail!(MismatchedTypes: "non-nullable unsigned int", ends.dtype());
        }
        if !ends.statistics().compute_is_strict_sorted().unwrap_or(true) {
            vortex_bail!("Ends array must be strictly sorted");
        }

        Ok(Self {
            ends,
            values,
            offset,
            length,
            stats_set: Default::default(),
        })
    }

    /// Convert the given logical index to an index into the `values` array
    pub fn find_physical_index(&self, index: usize) -> VortexResult<usize> {
        search_sorted_usize(self.ends(), index + self.offset(), SearchSortedSide::Right)
            .map(|s| s.to_ends_index(self.ends().len()))
    }

    /// Convert a batch of logical indices into an index for the values. Expects indices to be adjusted by offset unlike
    /// [Self::find_physical_index]
    ///
    /// See: [find_physical_index][Self::find_physical_index].
    pub fn find_physical_indices(&self, indices: &[usize]) -> VortexResult<Buffer<u64>> {
        search_sorted_usize_many(self.ends(), indices, SearchSortedSide::Right).map(|results| {
            results
                .into_iter()
                .map(|result| result.to_ends_index(self.ends().len()) as u64)
                .collect()
        })
    }

    /// Run the array through run-end encoding.
    pub fn encode(array: ArrayRef) -> VortexResult<Self> {
        if let Ok(parray) = PrimitiveArray::try_from(array) {
            let (ends, values) = runend_encode(&parray)?;
            Self::try_new(ends.into_array(), values)
        } else {
            vortex_bail!("REE can only encode primitive arrays")
        }
    }

    /// The offset that the `ends` is relative to.
    ///
    /// This is generally zero for a "new" array, and non-zero after a slicing operation.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// The encoded "ends" of value runs.
    ///
    /// The `i`-th element indicates that there is a run of the same value, beginning
    /// at `ends[i]` (inclusive) and terminating at `ends[i+1]` (exclusive).
    #[inline]
    pub fn ends(&self) -> &ArrayRef {
        &self.ends
    }

    /// The scalar values.
    ///
    /// The `i`-th element is the scalar value for the `i`-th repeated run. The run begins
    /// at `ends[i]` (inclusive) and terminates at `ends[i+1]` (exclusive).
    #[inline]
    pub fn values(&self) -> &ArrayRef {
        &self.values
    }
}

impl ArrayImpl for RunEndArray {
    type Encoding = RunEndEncoding;

    fn _len(&self) -> usize {
        self.length
    }

    fn _dtype(&self) -> &DType {
        self.values.dtype()
    }

    fn _vtable(&self) -> VTableRef {
        VTableRef::new_ref(&RunEndEncoding)
    }
}

impl ArrayVariantsImpl for RunEndArray {
    fn _as_bool_typed(&self) -> Option<&dyn BoolArrayTrait> {
        Some(self)
    }

    fn _as_primitive_typed(&self) -> Option<&dyn PrimitiveArrayTrait> {
        Some(self)
    }
}

impl PrimitiveArrayTrait for RunEndArray {}

impl BoolArrayTrait for RunEndArray {}

impl ArrayValidityImpl for RunEndArray {
    fn _is_valid(&self, index: usize) -> VortexResult<bool> {
        let physical_idx = self
            .find_physical_index(index)
            .vortex_expect("Invalid index");
        self.values().is_valid(physical_idx)
    }

    fn _all_valid(&self) -> VortexResult<bool> {
        self.values().all_valid()
    }

    fn _all_invalid(&self) -> VortexResult<bool> {
        self.values().all_invalid()
    }

    fn _validity_mask(&self) -> VortexResult<Mask> {
        Ok(match self.values().validity_mask()? {
            Mask::AllTrue(_) => Mask::AllTrue(self.len()),
            Mask::AllFalse(_) => Mask::AllFalse(self.len()),
            Mask::Values(values) => {
                let ree_validity = RunEndArray::with_offset_and_length(
                    self.ends().clone(),
                    values.into_array(),
                    self.offset(),
                    self.len(),
                )
                .vortex_expect("invalid array")
                .into_array();
                Mask::from_buffer(ree_validity.to_bool()?.boolean_buffer().clone())
            }
        })
    }
}

impl ArrayCanonicalImpl for RunEndArray {
    fn _to_canonical(&self) -> VortexResult<Canonical> {
        let pends = self.ends().to_primitive()?;
        match self.dtype() {
            DType::Bool(_) => {
                let bools = self.values().to_bool()?;
                runend_decode_bools(pends, bools, self.offset(), self.len()).map(Canonical::Bool)
            }
            DType::Primitive(..) => {
                let pvalues = self.values().to_primitive()?;
                runend_decode_primitive(pends, pvalues, self.offset(), self.len())
                    .map(Canonical::Primitive)
            }
            _ => vortex_bail!("Only Primitive and Bool values are supported"),
        }
    }
}

impl ArrayStatisticsImpl for RunEndArray {
    fn _stats_set(&self) -> &RwLock<StatsSet> {
        &self.stats_set
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::compute::scalar_at;
    use vortex_array::{Array, IntoArray};
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, Nullability, PType};

    use crate::RunEndArray;

    #[test]
    fn test_runend_constructor() {
        let arr = RunEndArray::try_new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
        )
        .unwrap();
        assert_eq!(arr.len(), 10);
        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );

        // 0, 1 => 1
        // 2, 3, 4 => 2
        // 5, 6, 7, 8, 9 => 3
        assert_eq!(scalar_at(&arr, 0).unwrap(), 1.into());
        assert_eq!(scalar_at(&arr, 2).unwrap(), 2.into());
        assert_eq!(scalar_at(&arr, 5).unwrap(), 3.into());
        assert_eq!(scalar_at(&arr, 9).unwrap(), 3.into());
    }
}
