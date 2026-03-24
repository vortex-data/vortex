// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::DeserializeMetadata;
use vortex_array::DynArray;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::patches::extract_offset;
use vortex_array::patches::wrap_with_offset;
use vortex_array::scalar::PValue;
use vortex_array::search_sorted::SearchSorted;
use vortex_array::search_sorted::SearchSortedSide;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;

use crate::compress::runend_decode_primitive;
use crate::compress::runend_decode_varbinview;
use crate::compress::runend_encode;
use crate::decompress_bool::runend_decode_bools;
use crate::kernel::PARENT_KERNELS;
use crate::rules::RULES;

vtable!(RunEnd);

#[derive(Clone, prost::Message)]
pub struct RunEndMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    pub ends_ptype: i32,
    #[prost(uint64, tag = "2")]
    pub num_runs: u64,
    #[prost(uint64, tag = "3")]
    pub offset: u64,
}

impl VTable for RunEnd {
    type Array = RunEndArray;

    type Metadata = ProstMetadata<RunEndMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn vtable(_array: &Self::Array) -> &Self {
        &RunEnd
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &RunEndArray) -> usize {
        array.length
    }

    fn dtype(array: &RunEndArray) -> &DType {
        array.values.dtype()
    }

    fn stats(array: &RunEndArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &RunEndArray, state: &mut H, precision: Precision) {
        array.ends.array_hash(state, precision);
        array.values.array_hash(state, precision);
        array.length.hash(state);
    }

    fn array_eq(array: &RunEndArray, other: &RunEndArray, precision: Precision) -> bool {
        array.ends.array_eq(&other.ends, precision)
            && array.values.array_eq(&other.values, precision)
            && array.length == other.length
    }

    fn nbuffers(_array: &RunEndArray) -> usize {
        0
    }

    fn buffer(_array: &RunEndArray, idx: usize) -> BufferHandle {
        vortex_panic!("RunEndArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &RunEndArray, idx: usize) -> Option<String> {
        vortex_panic!("RunEndArray buffer_name index {idx} out of bounds")
    }

    fn nchildren(_array: &RunEndArray) -> usize {
        2
    }

    fn child(array: &RunEndArray, idx: usize) -> ArrayRef {
        match idx {
            // Return raw ends (without Binary(Sub) wrapper) for serialization.
            // The offset is stored in metadata.
            0 => array.raw_ends_and_offset().0.clone(),
            1 => array.values().clone(),
            _ => vortex_panic!("RunEndArray child index {idx} out of bounds"),
        }
    }

    fn child_name(_array: &RunEndArray, idx: usize) -> String {
        match idx {
            0 => "ends".to_string(),
            1 => "values".to_string(),
            _ => vortex_panic!("RunEndArray child_name index {idx} out of bounds"),
        }
    }

    fn metadata(array: &RunEndArray) -> VortexResult<Self::Metadata> {
        let (raw_ends, offset) = array.raw_ends_and_offset();
        Ok(ProstMetadata(RunEndMetadata {
            ends_ptype: PType::try_from(raw_ends.dtype()).vortex_expect("Must be a valid PType")
                as i32,
            num_runs: raw_ends.len() as u64,
            offset: offset as u64,
        }))
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(metadata.serialize()))
    }

    fn deserialize(
        bytes: &[u8],
        _dtype: &DType,
        _len: usize,
        _buffers: &[BufferHandle],
        _session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        let inner = <ProstMetadata<RunEndMetadata> as DeserializeMetadata>::deserialize(bytes)?;
        Ok(ProstMetadata(inner))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<RunEndArray> {
        let ends_dtype = DType::Primitive(metadata.ends_ptype(), Nullability::NonNullable);
        let runs = usize::try_from(metadata.num_runs).vortex_expect("Must be a valid usize");
        let ends = children.get(0, &ends_dtype, runs)?;
        let offset = usize::try_from(metadata.offset).vortex_expect("Offset must be a valid usize");

        let values = children.get(1, dtype, runs)?;

        RunEndArray::try_new_offset_length(ends, values, offset, len)
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure!(
            children.len() == 2,
            "RunEndArray expects 2 children, got {}",
            children.len()
        );

        let mut children_iter = children.into_iter();
        array.ends = children_iter.next().vortex_expect("ends child");
        array.values = children_iter.next().vortex_expect("values child");

        Ok(())
    }

    fn reduce_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: &Self::Array,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn execute(array: Arc<Self::Array>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        run_end_canonicalize(&array, ctx).map(ExecutionResult::done)
    }
}

#[derive(Clone, Debug)]
pub struct RunEndArray {
    /// The run ends. May be a raw primitive array (offset=0) or a
    /// `Binary(Sub, raw_ends, Constant(offset))` expression when offset != 0.
    ends: ArrayRef,
    values: ArrayRef,
    length: usize,
    stats_set: ArrayStats,
}

pub struct RunEndArrayParts {
    pub ends: ArrayRef,
    pub values: ArrayRef,
}

#[derive(Clone, Debug)]
pub struct RunEnd;

impl RunEnd {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.runend");
}

impl RunEndArray {
    /// Validate the raw (unwrapped) ends and values with a given offset and length.
    fn validate(
        raw_ends: &ArrayRef,
        values: &ArrayRef,
        offset: usize,
        length: usize,
    ) -> VortexResult<()> {
        // DType validation
        vortex_ensure!(
            raw_ends.dtype().is_unsigned_int(),
            "run ends must be unsigned integers, was {}",
            raw_ends.dtype(),
        );
        vortex_ensure!(
            raw_ends.len() == values.len(),
            "run ends len != run values len, {} != {}",
            raw_ends.len(),
            values.len()
        );

        // Handle empty run-ends
        if raw_ends.is_empty() {
            vortex_ensure!(
                offset == 0,
                "non-zero offset provided for empty RunEndArray"
            );
            return Ok(());
        }

        // Avoid building a non-empty array with zero logical length.
        if length == 0 {
            vortex_ensure!(
                raw_ends.is_empty(),
                "run ends must be empty when length is zero"
            );
            return Ok(());
        }

        debug_assert!({
            // Run ends must be strictly sorted for binary search to work correctly.
            let pre_validation = raw_ends.statistics().to_owned();

            let is_sorted = raw_ends
                .statistics()
                .compute_is_strict_sorted()
                .unwrap_or(false);

            // Preserve the original statistics since compute_is_strict_sorted may have mutated them.
            // We don't want to run with different stats in debug mode and outside.
            raw_ends.statistics().inherit(pre_validation.iter());
            is_sorted
        });

        // Skip host-only validation when ends are not host-resident.
        if !raw_ends.is_host() {
            return Ok(());
        }

        // Validate the offset and length are valid for the given ends and values
        if offset != 0 && length != 0 {
            let first_run_end = usize::try_from(&raw_ends.scalar_at(0)?)?;
            if first_run_end <= offset {
                vortex_bail!("First run end {first_run_end} must be bigger than offset {offset}");
            }
        }

        let last_run_end = usize::try_from(&raw_ends.scalar_at(raw_ends.len() - 1)?)?;
        let min_required_end = offset + length;
        if last_run_end < min_required_end {
            vortex_bail!("Last run end {last_run_end} must be >= offset+length {min_required_end}");
        }

        Ok(())
    }
}

impl RunEndArray {
    /// Build a new `RunEndArray` from an array of run `ends` and an array of `values`.
    ///
    /// Panics if any of the validation conditions described in [`RunEndArray::try_new`] is
    /// not satisfied.
    ///
    /// # Examples
    ///
    /// ```
    /// # use vortex_array::arrays::BoolArray;
    /// # use vortex_array::IntoArray;
    /// # use vortex_buffer::buffer;
    /// # use vortex_error::VortexResult;
    /// # use vortex_runend::RunEndArray;
    /// # fn main() -> VortexResult<()> {
    /// let ends = buffer![2u8, 3u8].into_array();
    /// let values = BoolArray::from_iter([false, true]).into_array();
    /// let run_end = RunEndArray::new(ends, values);
    ///
    /// // Array encodes
    /// assert_eq!(run_end.scalar_at(0)?, false.into());
    /// assert_eq!(run_end.scalar_at(1)?, false.into());
    /// assert_eq!(run_end.scalar_at(2)?, true.into());
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(ends: ArrayRef, values: ArrayRef) -> Self {
        Self::try_new(ends, values).vortex_expect("RunEndArray new")
    }

    /// Build a new `RunEndArray` from components.
    ///
    /// # Validation
    ///
    /// The `ends` must be non-nullable unsigned integers.
    pub fn try_new(ends: ArrayRef, values: ArrayRef) -> VortexResult<Self> {
        let length: usize = if ends.is_empty() {
            0
        } else {
            usize::try_from(&ends.scalar_at(ends.len() - 1)?)?
        };

        Self::try_new_offset_length(ends, values, 0, length)
    }

    /// Construct a new sliced `RunEndArray` with the provided offset and length.
    ///
    /// When `offset` is non-zero, the `ends` are wrapped in a lazy `Binary(Sub)` expression
    /// so that the offset information is embedded in the ends array rather than stored separately.
    pub fn try_new_offset_length(
        ends: ArrayRef,
        values: ArrayRef,
        offset: usize,
        length: usize,
    ) -> VortexResult<Self> {
        Self::validate(&ends, &values, offset, length)?;
        let ends = wrap_with_offset(ends, offset)?;

        Ok(Self {
            ends,
            values,
            length,
            stats_set: Default::default(),
        })
    }

    /// Build a new `RunEndArray` without validation.
    ///
    /// The `ends` should already encode any offset via a `Binary(Sub)` expression
    /// (use [`wrap_with_offset`] if needed).
    ///
    /// # Safety
    ///
    /// The caller must ensure that all the validation performed in [`RunEndArray::try_new`] is
    /// satisfied before calling this function.
    pub unsafe fn new_unchecked(ends: ArrayRef, values: ArrayRef, length: usize) -> Self {
        Self {
            ends,
            values,
            length,
            stats_set: Default::default(),
        }
    }

    /// Convert the given logical index to an index into the `values` array.
    ///
    /// Uses the raw (unwrapped) ends and offset extracted from the expression tree.
    pub fn find_physical_index(&self, index: usize) -> VortexResult<usize> {
        let (raw_ends, offset) = self.raw_ends_and_offset();
        Ok(raw_ends
            .as_primitive_typed()
            .search_sorted(&PValue::from(index + offset), SearchSortedSide::Right)?
            .to_ends_index(raw_ends.len()))
    }

    /// Run the array through run-end encoding.
    pub fn encode(array: ArrayRef) -> VortexResult<Self> {
        if let Some(parray) = array.as_opt::<Primitive>() {
            let (ends, values) = runend_encode(parray);
            Ok(Self {
                ends: ends.into_array(),
                values,
                length: array.len(),
                stats_set: Default::default(),
            })
        } else {
            vortex_bail!("REE can only encode primitive arrays")
        }
    }

    /// The offset that the `ends` is relative to.
    ///
    /// This is generally zero for a "new" array, and non-zero after a slicing operation.
    /// The offset is extracted from the Binary(Sub) expression wrapping the ends.
    #[inline]
    pub fn offset(&self) -> usize {
        self.raw_ends_and_offset().1
    }

    /// Extract the raw (unwrapped) ends array and the offset from the expression tree.
    ///
    /// If `ends` is `Binary(Sub, raw_ends, Constant(offset))`, returns `(raw_ends, offset)`.
    /// Otherwise returns `(ends, 0)`.
    pub fn raw_ends_and_offset(&self) -> (&ArrayRef, usize) {
        extract_offset(&self.ends)
    }

    /// The encoded "ends" of value runs.
    ///
    /// If the array has been sliced, this returns the Binary(Sub) expression.
    /// Use [`raw_ends_and_offset`](Self::raw_ends_and_offset) to get the raw ends and offset.
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

    /// Split an `RunEndArray` into parts.
    #[inline]
    pub fn into_parts(self) -> RunEndArrayParts {
        RunEndArrayParts {
            ends: self.ends,
            values: self.values,
        }
    }
}

impl ValidityVTable<RunEnd> for RunEnd {
    fn validity(array: &RunEndArray) -> VortexResult<Validity> {
        Ok(match array.values().validity()? {
            Validity::NonNullable | Validity::AllValid => Validity::AllValid,
            Validity::AllInvalid => Validity::AllInvalid,
            Validity::Array(values_validity) => Validity::Array(unsafe {
                RunEndArray::new_unchecked(array.ends().clone(), values_validity, array.len())
                    .into_array()
            }),
        })
    }
}

/// Fused decompression: extracts the offset from a Binary(Sub) expression on ends
/// and passes it directly to the decode functions, avoiding materialization of the
/// subtracted ends array.
pub(super) fn run_end_canonicalize(
    array: &RunEndArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    // Fused execution: extract offset from Binary(Sub) expression without materializing.
    let (raw_ends, offset) = array.raw_ends_and_offset();
    let pends = raw_ends.clone().execute_as("ends", ctx)?;

    Ok(match array.dtype() {
        DType::Bool(_) => {
            let bools = array.values().clone().execute_as("values", ctx)?;
            runend_decode_bools(pends, bools, offset, array.len())?
        }
        DType::Primitive(..) => {
            let pvalues = array.values().clone().execute_as("values", ctx)?;
            runend_decode_primitive(pends, pvalues, offset, array.len())?.into_array()
        }
        DType::Utf8(_) | DType::Binary(_) => {
            let values = array
                .values()
                .clone()
                .execute_as::<VarBinViewArray>("values", ctx)?;
            runend_decode_varbinview(pends, values, offset, array.len())?.into_array()
        }
        _ => vortex_bail!("Unsupported RunEnd value type: {}", array.dtype()),
    })
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::DictArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_buffer::buffer;

    use crate::RunEndArray;

    #[test]
    fn test_runend_constructor() {
        let arr = RunEndArray::new(
            buffer![2u32, 5, 10].into_array(),
            buffer![1i32, 2, 3].into_array(),
        );
        assert_eq!(arr.len(), 10);
        assert_eq!(
            arr.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );

        // 0, 1 => 1
        // 2, 3, 4 => 2
        // 5, 6, 7, 8, 9 => 3
        let expected = buffer![1, 1, 2, 2, 2, 3, 3, 3, 3, 3].into_array();
        assert_arrays_eq!(arr.into_array(), expected);
    }

    #[test]
    fn test_runend_utf8() {
        let values = VarBinViewArray::from_iter_str(["a", "b", "c"]).into_array();
        let arr = RunEndArray::new(buffer![2u32, 5, 10].into_array(), values);
        assert_eq!(arr.len(), 10);
        assert_eq!(arr.dtype(), &DType::Utf8(Nullability::NonNullable));

        let expected =
            VarBinViewArray::from_iter_str(["a", "a", "b", "b", "b", "c", "c", "c", "c", "c"])
                .into_array();
        assert_arrays_eq!(arr.into_array(), expected);
    }

    #[test]
    fn test_runend_dict() {
        let dict_values = VarBinViewArray::from_iter_str(["x", "y", "z"]).into_array();
        let dict_codes = buffer![0u32, 1, 2].into_array();
        let dict = DictArray::try_new(dict_codes, dict_values).unwrap();

        let arr =
            RunEndArray::try_new(buffer![2u32, 5, 10].into_array(), dict.into_array()).unwrap();
        assert_eq!(arr.len(), 10);

        let expected =
            VarBinViewArray::from_iter_str(["x", "x", "y", "y", "y", "z", "z", "z", "z", "z"])
                .into_array();
        assert_arrays_eq!(arr.into_array(), expected);
    }
}
