// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;

use kernel::PARENT_KERNELS;
use prost::Message as _;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::DynArray;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionStep;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ToCanonical;
use vortex_array::arrays::ConstantArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::builtins::ArrayBuiltins;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesMetadata;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::ScalarValue;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
use vortex_array::vtable::patches_child;
use vortex_array::vtable::patches_child_name;
use vortex_array::vtable::patches_nchildren;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect as _;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_panic;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::canonical::execute_sparse;
use crate::rules::RULES;

mod canonical;
mod compute;
mod kernel;
mod ops;
mod rules;
mod slice;

vtable!(Sparse);

#[derive(Debug)]
pub struct SparseMetadata {
    patches: PatchesMetadata,
    fill_value: Scalar,
}

#[derive(Clone, prost::Message)]
#[repr(C)]
pub struct ProstPatchesMetadata {
    #[prost(message, required, tag = "1")]
    patches: PatchesMetadata,
}

impl VTable for Sparse {
    type Array = SparseArray;

    type Metadata = SparseMetadata;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn vtable(_array: &Self::Array) -> &Self {
        &Sparse
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &SparseArray) -> usize {
        array.patches.array_len()
    }

    fn dtype(array: &SparseArray) -> &DType {
        array.fill_scalar().dtype()
    }

    fn stats(array: &SparseArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &SparseArray, state: &mut H, precision: Precision) {
        array.patches.array_hash(state, precision);
        array.fill_value.hash(state);
    }

    fn array_eq(array: &SparseArray, other: &SparseArray, precision: Precision) -> bool {
        array.patches.array_eq(&other.patches, precision) && array.fill_value == other.fill_value
    }

    fn nbuffers(_array: &SparseArray) -> usize {
        1
    }

    fn buffer(array: &SparseArray, idx: usize) -> BufferHandle {
        match idx {
            0 => {
                let fill_value_buffer =
                    ScalarValue::to_proto_bytes::<ByteBufferMut>(array.fill_value.value()).freeze();
                BufferHandle::new_host(fill_value_buffer)
            }
            _ => vortex_panic!("SparseArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: &SparseArray, idx: usize) -> Option<String> {
        match idx {
            0 => Some("fill_value".to_string()),
            _ => vortex_panic!("SparseArray buffer_name index {idx} out of bounds"),
        }
    }

    fn nchildren(array: &SparseArray) -> usize {
        patches_nchildren(array.patches())
    }

    fn child(array: &SparseArray, idx: usize) -> ArrayRef {
        patches_child(array.patches(), idx)
    }

    fn child_name(_array: &SparseArray, idx: usize) -> String {
        patches_child_name(idx).to_string()
    }

    fn metadata(array: &SparseArray) -> VortexResult<Self::Metadata> {
        let patches = array.patches().to_metadata(array.len(), array.dtype())?;

        Ok(SparseMetadata {
            patches,
            fill_value: array.fill_value.clone(),
        })
    }

    fn serialize(metadata: Self::Metadata) -> VortexResult<Option<Vec<u8>>> {
        let prost_patches = ProstPatchesMetadata {
            patches: metadata.patches,
        };

        // Note that we DO NOT serialize the fill value since that is stored in the buffers.
        Ok(Some(prost_patches.encode_to_vec()))
    }

    fn deserialize(
        bytes: &[u8],
        dtype: &DType,
        _len: usize,
        buffers: &[BufferHandle],
        session: &VortexSession,
    ) -> VortexResult<Self::Metadata> {
        let prost_patches = ProstPatchesMetadata::decode(bytes)?;

        // Once we have the patches metadata, we need to get the fill value from the buffers.

        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let scalar_bytes: &[u8] = &buffers[0].clone().try_to_host_sync()?;

        let scalar_value = ScalarValue::from_proto_bytes(scalar_bytes, dtype, session)?;
        let fill_value = Scalar::try_new(dtype.clone(), scalar_value)?;

        Ok(SparseMetadata {
            patches: prost_patches.patches,
            fill_value,
        })
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<SparseArray> {
        vortex_ensure_eq!(
            children.len(),
            2,
            "SparseArray expects 2 children for sparse encoding, found {}",
            children.len()
        );
        vortex_ensure_eq!(
            metadata.patches.offset()?,
            0,
            "Patches must start at offset 0"
        );

        let patch_indices = children.get(
            0,
            &metadata.patches.indices_dtype()?,
            metadata.patches.len()?,
        )?;
        let patch_values = children.get(1, dtype, metadata.patches.len()?)?;

        SparseArray::try_new(
            patch_indices,
            patch_values,
            len,
            metadata.fill_value.clone(),
        )
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        vortex_ensure_eq!(
            children.len(),
            2,
            "SparseArray expects 2 children, got {}",
            children.len()
        );

        let mut children_iter = children.into_iter();
        let patch_indices = children_iter.next().vortex_expect("patch_indices child");
        let patch_values = children_iter.next().vortex_expect("patch_values child");

        array.patches = Patches::new(
            array.patches.array_len(),
            array.patches.offset(),
            patch_indices,
            patch_values,
            array.patches.chunk_offsets().clone(),
        )?;

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

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
        execute_sparse(array, ctx).map(ExecutionStep::Done)
    }
}

#[derive(Clone, Debug)]
pub struct SparseArray {
    patches: Patches,
    fill_value: Scalar,
    stats_set: ArrayStats,
}

#[derive(Debug)]
pub struct Sparse;

impl Sparse {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.sparse");
}

impl SparseArray {
    pub fn try_new(
        indices: ArrayRef,
        values: ArrayRef,
        len: usize,
        fill_value: Scalar,
    ) -> VortexResult<Self> {
        vortex_ensure!(
            indices.len() == values.len(),
            "Mismatched indices {} and values {} length",
            indices.len(),
            values.len()
        );

        if indices.is_host() {
            debug_assert_eq!(
                indices.statistics().compute_is_strict_sorted(),
                Some(true),
                "SparseArray: indices must be strict-sorted"
            );

            // Verify the indices are all in the valid range
            if !indices.is_empty() {
                let last_index = usize::try_from(&indices.scalar_at(indices.len() - 1)?)?;

                vortex_ensure!(
                    last_index < len,
                    "Array length was {len} but the last index is {last_index}"
                );
            }
        }

        Ok(Self {
            // TODO(0ax1): handle chunk offsets
            patches: Patches::new(len, 0, indices, values, None)?,
            fill_value,
            stats_set: Default::default(),
        })
    }

    /// Build a new SparseArray from an existing set of patches.
    pub fn try_new_from_patches(patches: Patches, fill_value: Scalar) -> VortexResult<Self> {
        vortex_ensure!(
            fill_value.dtype() == patches.values().dtype(),
            "fill value, {:?}, should be instance of values dtype, {} but was {}.",
            fill_value,
            patches.values().dtype(),
            fill_value.dtype(),
        );

        Ok(Self {
            patches,
            fill_value,
            stats_set: Default::default(),
        })
    }

    pub(crate) unsafe fn new_unchecked(patches: Patches, fill_value: Scalar) -> Self {
        Self {
            patches,
            fill_value,
            stats_set: Default::default(),
        }
    }

    #[inline]
    pub fn patches(&self) -> &Patches {
        &self.patches
    }

    #[inline]
    pub fn resolved_patches(&self) -> VortexResult<Patches> {
        let patches = self.patches();
        let indices_offset = Scalar::from(patches.offset()).cast(patches.indices().dtype())?;
        let indices = patches.indices().to_array().binary(
            ConstantArray::new(indices_offset, patches.indices().len()).into_array(),
            Operator::Sub,
        )?;

        Patches::new(
            patches.array_len(),
            0,
            indices,
            patches.values().clone(),
            // TODO(0ax1): handle chunk offsets
            None,
        )
    }

    #[inline]
    pub fn fill_scalar(&self) -> &Scalar {
        &self.fill_value
    }

    /// Encode given array as a SparseArray.
    ///
    /// Optionally provided fill value will be respected if the array is less than 90% null.
    pub fn encode(array: &ArrayRef, fill_value: Option<Scalar>) -> VortexResult<ArrayRef> {
        if let Some(fill_value) = fill_value.as_ref()
            && array.dtype() != fill_value.dtype()
        {
            vortex_bail!(
                "Array and fill value types must match. got {} and {}",
                array.dtype(),
                fill_value.dtype()
            )
        }
        let mask = array.validity_mask()?;

        if mask.all_false() {
            // Array is constant NULL
            return Ok(
                ConstantArray::new(Scalar::null(array.dtype().clone()), array.len()).into_array(),
            );
        } else if mask.false_count() as f64 > (0.9 * mask.len() as f64) {
            // Array is dominated by NULL but has non-NULL values
            // TODO(joe): use exe ctx?
            let non_null_values = array.filter(mask.clone())?.to_canonical()?.into_array();
            let non_null_indices = match mask.indices() {
                AllOr::All => {
                    // We already know that the mask is 90%+ false
                    unreachable!("Mask is mostly null")
                }
                AllOr::None => {
                    // we know there are some non-NULL values
                    unreachable!("Mask is mostly null but not all null")
                }
                AllOr::Some(values) => {
                    let buffer: Buffer<u32> = values
                        .iter()
                        .map(|&v| v.try_into().vortex_expect("indices must fit in u32"))
                        .collect();

                    buffer.into_array()
                }
            };

            return Ok(SparseArray::try_new(
                non_null_indices,
                non_null_values,
                array.len(),
                Scalar::null(array.dtype().clone()),
            )?
            .into_array());
        }

        let fill = if let Some(fill) = fill_value {
            fill
        } else {
            // TODO(robert): Support other dtypes, only thing missing is getting most common value out of the array
            let (top_pvalue, _) = array
                .to_primitive()
                .top_value()?
                .vortex_expect("Non empty or all null array");

            Scalar::primitive_value(top_pvalue, top_pvalue.ptype(), array.dtype().nullability())
        };

        let fill_array = ConstantArray::new(fill.clone(), array.len()).into_array();
        let non_top_mask = Mask::from_buffer(
            array
                .to_array()
                .binary(fill_array.clone(), Operator::NotEq)?
                .fill_null(Scalar::bool(true, Nullability::NonNullable))?
                .to_bool()
                .to_bit_buffer(),
        );

        let non_top_values = array
            .filter(non_top_mask.clone())?
            .to_canonical()?
            .into_array();

        let indices: Buffer<u64> = match non_top_mask {
            Mask::AllTrue(count) => {
                // all true -> complete slice
                (0u64..count as u64).collect()
            }
            Mask::AllFalse(_) => {
                // All values are equal to the top value
                return Ok(fill_array);
            }
            Mask::Values(values) => values.indices().iter().map(|v| *v as u64).collect(),
        };

        SparseArray::try_new(indices.into_array(), non_top_values, array.len(), fill)
            .map(|a| a.into_array())
    }
}

impl ValidityVTable<Sparse> for Sparse {
    fn validity(array: &SparseArray) -> VortexResult<Validity> {
        let patches = unsafe {
            Patches::new_unchecked(
                array.patches.array_len(),
                array.patches.offset(),
                array.patches.indices().clone(),
                array
                    .patches
                    .values()
                    .validity()?
                    .to_array(array.patches.values().len()),
                array.patches.chunk_offsets().clone(),
                array.patches.offset_within_chunk(),
            )
        };

        Ok(Validity::Array(
            unsafe { SparseArray::new_unchecked(patches, array.fill_value.is_valid().into()) }
                .into_array(),
        ))
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_array::DynArray;
    use vortex_array::IntoArray;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::builtins::ArrayBuiltins;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;

    use super::*;

    fn nullable_fill() -> Scalar {
        Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable))
    }

    fn non_nullable_fill() -> Scalar {
        Scalar::from(42i32)
    }

    fn sparse_array(fill_value: Scalar) -> ArrayRef {
        // merged array: [null, null, 100, null, null, 200, null, null, 300, null]
        let mut values = buffer![100i32, 200, 300].into_array();
        values = values.cast(fill_value.dtype().clone()).unwrap();

        SparseArray::try_new(buffer![2u64, 5, 8].into_array(), values, 10, fill_value)
            .unwrap()
            .into_array()
    }

    #[test]
    pub fn test_scalar_at() {
        let array = sparse_array(nullable_fill());

        assert_eq!(array.scalar_at(0).unwrap(), nullable_fill());
        assert_eq!(array.scalar_at(2).unwrap(), Scalar::from(Some(100_i32)));
        assert_eq!(array.scalar_at(5).unwrap(), Scalar::from(Some(200_i32)));
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_scalar_at_oob() {
        let array = sparse_array(nullable_fill());
        array.scalar_at(10).unwrap();
    }

    #[test]
    pub fn test_scalar_at_again() {
        let arr = SparseArray::try_new(
            ConstantArray::new(10u32, 1).into_array(),
            ConstantArray::new(Scalar::primitive(1234u32, Nullability::Nullable), 1).into_array(),
            100,
            Scalar::null(DType::Primitive(PType::U32, Nullability::Nullable)),
        )
        .unwrap();

        assert_eq!(
            arr.scalar_at(10)
                .unwrap()
                .as_primitive()
                .typed_value::<u32>(),
            Some(1234)
        );
        assert!(arr.scalar_at(0).unwrap().is_null());
        assert!(arr.scalar_at(99).unwrap().is_null());
    }

    #[test]
    pub fn scalar_at_sliced() {
        let sliced = sparse_array(nullable_fill()).slice(2..7).unwrap();
        assert_eq!(usize::try_from(&sliced.scalar_at(0).unwrap()).unwrap(), 100);
    }

    #[test]
    pub fn validity_mask_sliced_null_fill() {
        let sliced = sparse_array(nullable_fill()).slice(2..7).unwrap();
        assert_eq!(
            sliced.validity_mask().unwrap(),
            Mask::from_iter(vec![true, false, false, true, false])
        );
    }

    #[test]
    pub fn validity_mask_sliced_nonnull_fill() {
        let sliced = SparseArray::try_new(
            buffer![2u64, 5, 8].into_array(),
            ConstantArray::new(
                Scalar::null(DType::Primitive(PType::F32, Nullability::Nullable)),
                3,
            )
            .into_array(),
            10,
            Scalar::primitive(1.0f32, Nullability::Nullable),
        )
        .unwrap()
        .slice(2..7)
        .unwrap();

        assert_eq!(
            sliced.validity_mask().unwrap(),
            Mask::from_iter(vec![false, true, true, false, true])
        );
    }

    #[test]
    pub fn scalar_at_sliced_twice() {
        let sliced_once = sparse_array(nullable_fill()).slice(1..8).unwrap();
        assert_eq!(
            usize::try_from(&sliced_once.scalar_at(1).unwrap()).unwrap(),
            100
        );

        let sliced_twice = sliced_once.slice(1..6).unwrap();
        assert_eq!(
            usize::try_from(&sliced_twice.scalar_at(3).unwrap()).unwrap(),
            200
        );
    }

    #[test]
    pub fn sparse_validity_mask() {
        let array = sparse_array(nullable_fill());
        assert_eq!(
            array
                .validity_mask()
                .unwrap()
                .to_bit_buffer()
                .iter()
                .collect_vec(),
            [
                false, false, true, false, false, true, false, false, true, false
            ]
        );
    }

    #[test]
    fn sparse_validity_mask_non_null_fill() {
        let array = sparse_array(non_nullable_fill());
        assert!(array.validity_mask().unwrap().all_true());
    }

    #[test]
    #[should_panic]
    fn test_invalid_length() {
        let values = buffer![15_u32, 135, 13531, 42].into_array();
        let indices = buffer![10_u64, 11, 50, 100].into_array();

        SparseArray::try_new(indices, values, 100, 0_u32.into()).unwrap();
    }

    #[test]
    fn test_valid_length() {
        let values = buffer![15_u32, 135, 13531, 42].into_array();
        let indices = buffer![10_u64, 11, 50, 100].into_array();

        SparseArray::try_new(indices, values, 101, 0_u32.into()).unwrap();
    }

    #[test]
    fn encode_with_nulls() {
        let original = PrimitiveArray::new(
            buffer![0i32, 1, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4],
            Validity::from_iter(vec![
                true, true, false, true, false, true, false, true, true, false, true, false,
            ]),
        );
        let sparse = SparseArray::encode(&original.clone().into_array(), None)
            .vortex_expect("SparseArray::encode should succeed for test data");
        assert_eq!(
            sparse.validity_mask().unwrap(),
            Mask::from_iter(vec![
                true, true, false, true, false, true, false, true, true, false, true, false,
            ])
        );
        assert_arrays_eq!(sparse.to_primitive(), original);
    }

    #[test]
    fn validity_mask_includes_null_values_when_fill_is_null() {
        let indices = buffer![0u8, 2, 4, 6, 8].into_array();
        let values = PrimitiveArray::from_option_iter([Some(0i16), Some(1), None, None, Some(4)])
            .into_array();
        let array =
            SparseArray::try_new(indices, values, 10, Scalar::null_native::<i16>()).unwrap();
        let actual = array.validity_mask().unwrap();
        let expected = Mask::from_iter([
            true, false, true, false, false, false, false, false, true, false,
        ]);

        assert_eq!(actual, expected);
    }
}
