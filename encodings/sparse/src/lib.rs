// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use kernel::PARENT_KERNELS;
use prost::Message as _;
use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::LEGACY_SESSION;
use vortex_array::Precision;
#[expect(deprecated)]
use vortex_array::ToCanonical;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::bool::BoolArrayExt;
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
use vortex_array::validity::Validity;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityVTable;
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
use vortex_session::registry::CachedId;

use crate::canonical::execute_sparse;
use crate::rules::RULES;

mod canonical;
mod compute;
mod kernel;
mod ops;
mod rules;
mod slice;

/// A [`Sparse`]-encoded Vortex array.
pub type SparseArray = Array<Sparse>;

#[derive(Clone, prost::Message)]
#[repr(C)]
pub struct SparseMetadata {
    #[prost(message, required, tag = "1")]
    patches: PatchesMetadata,
}

impl ArrayHash for SparseData {
    fn array_hash<H: Hasher>(&self, state: &mut H, precision: Precision) {
        self.patches.array_hash(state, precision);
        self.fill_value.hash(state);
    }
}

impl ArrayEq for SparseData {
    fn array_eq(&self, other: &Self, precision: Precision) -> bool {
        self.patches.array_eq(&other.patches, precision) && self.fill_value == other.fill_value
    }
}

impl VTable for Sparse {
    type ArrayData = SparseData;

    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.sparse");
        *ID
    }

    fn validate(
        &self,
        data: &Self::ArrayData,
        dtype: &DType,
        len: usize,
        _slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        SparseData::validate(data.patches(), data.fill_scalar(), dtype, len)
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        1
    }

    fn buffer(array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        match idx {
            0 => {
                let fill_value_buffer =
                    ScalarValue::to_proto_bytes::<ByteBufferMut>(array.fill_value.value()).freeze();
                BufferHandle::new_host(fill_value_buffer)
            }
            _ => vortex_panic!("SparseArray buffer index {idx} out of bounds"),
        }
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        match idx {
            0 => Some("fill_value".to_string()),
            _ => vortex_panic!("SparseArray buffer_name index {idx} out of bounds"),
        }
    }

    fn serialize(
        array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        let patches = array.patches().to_metadata(array.len(), array.dtype())?;
        let metadata = SparseMetadata { patches };

        // Note that we DO NOT serialize the fill value since that is stored in the buffers.
        Ok(Some(metadata.encode_to_vec()))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],
        buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        let metadata = SparseMetadata::decode(metadata)?;

        // Once we have the patches metadata, we need to get the fill value from the buffers.

        if buffers.len() != 1 {
            vortex_bail!("Expected 1 buffer, got {}", buffers.len());
        }
        let scalar_bytes: &[u8] = &buffers[0].clone().try_to_host_sync()?;

        let scalar_value = ScalarValue::from_proto_bytes(scalar_bytes, dtype, session)?;
        let fill_value = Scalar::try_new(dtype.clone(), scalar_value)?;

        vortex_ensure_eq!(
            children.len(),
            2,
            "SparseArray expects 2 children for sparse encoding, found {}",
            children.len()
        );

        let patch_indices = children.get(
            0,
            &metadata.patches.indices_dtype()?,
            metadata.patches.len()?,
        )?;
        let patch_values = children.get(1, dtype, metadata.patches.len()?)?;

        let patches = Patches::new(
            len,
            metadata.patches.offset()?,
            patch_indices,
            patch_values,
            None,
        )?;
        let slots = SparseData::make_slots(&patches);
        let data = SparseData::try_new_from_patches(patches, fill_value)?;
        Ok(ArrayParts::new(self.clone(), dtype.clone(), len, data).with_slots(slots))
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array, parent, child_idx)
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        execute_sparse(&array, ctx).map(ExecutionResult::done)
    }
}

pub(crate) const NUM_SLOTS: usize = 3;
pub(crate) const SLOT_NAMES: [&str; NUM_SLOTS] =
    ["patch_indices", "patch_values", "patch_chunk_offsets"];

#[derive(Clone, Debug)]
pub struct SparseData {
    patches: Patches,
    fill_value: Scalar,
}

impl Display for SparseData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "fill_value: {}", self.fill_value)
    }
}

#[derive(Clone, Debug)]
pub struct Sparse;

impl Sparse {
    /// Construct a new [`SparseArray`] from indices, values, length, and fill value.
    pub fn try_new(
        indices: ArrayRef,
        values: ArrayRef,
        len: usize,
        fill_value: Scalar,
    ) -> VortexResult<SparseArray> {
        let dtype = fill_value.dtype().clone();
        let patches = Patches::new(len, 0, indices, values, None)?;
        let slots = SparseData::make_slots(&patches);
        let data = SparseData::try_new_from_patches(patches, fill_value)?;
        Ok(unsafe {
            Array::from_parts_unchecked(ArrayParts::new(Sparse, dtype, len, data).with_slots(slots))
        })
    }

    pub fn try_new_from_patches(patches: Patches, fill_value: Scalar) -> VortexResult<SparseArray> {
        let dtype = fill_value.dtype().clone();
        let len = patches.array_len();
        let slots = SparseData::make_slots(&patches);
        let data = SparseData::try_new_from_patches(patches, fill_value)?;
        Ok(unsafe {
            Array::from_parts_unchecked(ArrayParts::new(Sparse, dtype, len, data).with_slots(slots))
        })
    }

    pub(crate) unsafe fn new_unchecked(patches: Patches, fill_value: Scalar) -> SparseArray {
        let dtype = fill_value.dtype().clone();
        let len = patches.array_len();
        let slots = SparseData::make_slots(&patches);
        let data = unsafe { SparseData::new_unchecked(patches, fill_value) };
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(Sparse, dtype, len, data).with_slots(slots))
        }
    }

    /// Encode the given array as a [`SparseArray`].
    pub fn encode(array: &ArrayRef, fill_value: Option<Scalar>) -> VortexResult<ArrayRef> {
        SparseData::encode(array, fill_value)
    }
}

impl SparseData {
    fn normalize_patches_dtype(patches: Patches, fill_value: &Scalar) -> VortexResult<Patches> {
        let fill_dtype = fill_value.dtype();
        let values_dtype = patches.values().dtype();

        vortex_ensure!(
            values_dtype.eq_ignore_nullability(fill_dtype),
            "fill value, {:?}, should be instance of values dtype, {} but was {}.",
            fill_value,
            values_dtype,
            fill_dtype,
        );

        if values_dtype == fill_dtype {
            Ok(patches)
        } else {
            patches.cast_values(fill_dtype)
        }
    }

    pub fn validate(
        patches: &Patches,
        fill_value: &Scalar,
        dtype: &DType,
        len: usize,
    ) -> VortexResult<()> {
        vortex_ensure!(
            fill_value.dtype() == dtype,
            "fill value dtype {} does not match array dtype {}",
            fill_value.dtype(),
            dtype,
        );
        vortex_ensure!(
            patches.array_len() == len,
            "patches length {} does not match array length {}",
            patches.array_len(),
            len
        );
        vortex_ensure!(
            patches.values().dtype() == dtype,
            "patch values dtype {} does not match array dtype {}",
            patches.values().dtype(),
            dtype,
        );
        Ok(())
    }

    fn make_slots(patches: &Patches) -> Vec<Option<ArrayRef>> {
        vec![
            Some(patches.indices().clone()),
            Some(patches.values().clone()),
            patches.chunk_offsets().clone(),
        ]
    }

    /// Build a new SparseArray from an existing set of patches.
    pub fn try_new_from_patches(patches: Patches, fill_value: Scalar) -> VortexResult<Self> {
        let patches = Self::normalize_patches_dtype(patches, &fill_value)?;
        Ok(Self {
            patches,
            fill_value,
        })
    }

    pub(crate) unsafe fn new_unchecked(patches: Patches, fill_value: Scalar) -> Self {
        Self {
            patches,
            fill_value,
        }
    }

    /// Returns the length of the array.
    #[inline]
    pub fn len(&self) -> usize {
        self.patches.array_len()
    }

    /// Returns whether the array is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.patches.array_len() == 0
    }

    /// Returns the logical data type of the array.
    #[inline]
    pub fn dtype(&self) -> &DType {
        self.fill_scalar().dtype()
    }

    #[inline]
    pub fn patches(&self) -> &Patches {
        &self.patches
    }

    #[inline]
    pub fn resolved_patches(&self) -> VortexResult<Patches> {
        let patches = self.patches();
        let indices_offset = Scalar::from(patches.offset()).cast(patches.indices().dtype())?;
        let indices = patches.indices().binary(
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
            && !array.dtype().eq_ignore_nullability(fill_value.dtype())
        {
            vortex_bail!(
                "Array and fill value types must have the same base type. got {} and {}",
                array.dtype(),
                fill_value.dtype()
            )
        }
        let mask = array
            .validity()?
            .to_mask(array.len(), &mut LEGACY_SESSION.create_execution_ctx())?;

        if mask.all_false() {
            // Array is constant NULL
            return Ok(
                ConstantArray::new(Scalar::null(array.dtype().clone()), array.len()).into_array(),
            );
        } else if mask.false_count() as f64 > (0.9 * mask.len() as f64) {
            // Array is dominated by NULL but has non-NULL values
            // TODO(joe): use exe ctx?
            let non_null_values = array
                .filter(mask.clone())?
                .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())?
                .into_array();
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

            return Sparse::try_new(
                non_null_indices,
                non_null_values,
                array.len(),
                Scalar::null(array.dtype().clone()),
            )
            .map(IntoArray::into_array);
        }

        let fill = if let Some(fill) = fill_value {
            fill.cast(array.dtype())?
        } else {
            // TODO(robert): Support other dtypes, only thing missing is getting most common value out of the array
            #[expect(deprecated)]
            let primitive = array.to_primitive();
            let (top_pvalue, _) = primitive
                .top_value()?
                .vortex_expect("Non empty or all null array");

            Scalar::primitive_value(top_pvalue, top_pvalue.ptype(), array.dtype().nullability())
        };

        let fill_array = ConstantArray::new(fill.clone(), array.len()).into_array();
        #[expect(deprecated)]
        let non_top_bool = array
            .binary(fill_array.clone(), Operator::NotEq)?
            .fill_null(Scalar::bool(true, Nullability::NonNullable))?
            .to_bool();
        let non_top_mask = Mask::from_buffer(non_top_bool.to_bit_buffer());

        let non_top_values = array
            .filter(non_top_mask.clone())?
            .execute::<Canonical>(&mut LEGACY_SESSION.create_execution_ctx())?
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

        Sparse::try_new(indices.into_array(), non_top_values, array.len(), fill)
            .map(IntoArray::into_array)
    }
}

impl ValidityVTable<Sparse> for Sparse {
    fn validity(array: ArrayView<'_, Sparse>) -> VortexResult<Validity> {
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
            unsafe { Sparse::new_unchecked(patches, array.fill_value.is_valid().into()) }
                .into_array(),
        ))
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;
    use vortex_array::IntoArray;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
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
    use crate::Sparse;

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

        Sparse::try_new(buffer![2u64, 5, 8].into_array(), values, 10, fill_value)
            .unwrap()
            .into_array()
    }

    #[test]
    pub fn test_scalar_at() {
        let array = sparse_array(nullable_fill());

        assert_eq!(
            array
                .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            nullable_fill()
        );
        assert_eq!(
            array
                .execute_scalar(2, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::from(Some(100_i32))
        );
        assert_eq!(
            array
                .execute_scalar(5, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Scalar::from(Some(200_i32))
        );
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn test_scalar_at_oob() {
        let array = sparse_array(nullable_fill());
        array
            .execute_scalar(10, &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
    }

    #[test]
    pub fn test_scalar_at_again() {
        let arr = Sparse::try_new(
            ConstantArray::new(10u32, 1).into_array(),
            ConstantArray::new(Scalar::primitive(1234u32, Nullability::Nullable), 1).into_array(),
            100,
            Scalar::null(DType::Primitive(PType::U32, Nullability::Nullable)),
        )
        .unwrap();

        assert_eq!(
            arr.execute_scalar(10, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
                .as_primitive()
                .typed_value::<u32>(),
            Some(1234)
        );
        assert!(
            arr.execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
                .is_null()
        );
        assert!(
            arr.execute_scalar(99, &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
                .is_null()
        );
    }

    #[test]
    pub fn scalar_at_sliced() {
        let sliced = sparse_array(nullable_fill()).slice(2..7).unwrap();
        assert_eq!(
            usize::try_from(
                &sliced
                    .execute_scalar(0, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap()
            )
            .unwrap(),
            100
        );
    }

    #[test]
    pub fn validity_mask_sliced_null_fill() {
        let sliced = sparse_array(nullable_fill()).slice(2..7).unwrap();
        assert_eq!(
            sliced
                .validity()
                .unwrap()
                .to_mask(sliced.len(), &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Mask::from_iter(vec![true, false, false, true, false])
        );
    }

    #[test]
    pub fn validity_mask_sliced_nonnull_fill() {
        let sliced = Sparse::try_new(
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
            sliced
                .validity()
                .unwrap()
                .to_mask(sliced.len(), &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Mask::from_iter(vec![false, true, true, false, true])
        );
    }

    #[test]
    pub fn scalar_at_sliced_twice() {
        let sliced_once = sparse_array(nullable_fill()).slice(1..8).unwrap();
        assert_eq!(
            usize::try_from(
                &sliced_once
                    .execute_scalar(1, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap()
            )
            .unwrap(),
            100
        );

        let sliced_twice = sliced_once.slice(1..6).unwrap();
        assert_eq!(
            usize::try_from(
                &sliced_twice
                    .execute_scalar(3, &mut LEGACY_SESSION.create_execution_ctx())
                    .unwrap()
            )
            .unwrap(),
            200
        );
    }

    #[test]
    pub fn sparse_validity_mask() {
        let array = sparse_array(nullable_fill());
        assert_eq!(
            array
                .validity()
                .unwrap()
                .to_mask(array.len(), &mut LEGACY_SESSION.create_execution_ctx())
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
        assert!(
            array
                .validity()
                .unwrap()
                .to_mask(array.len(), &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap()
                .all_true()
        );
    }

    #[test]
    #[should_panic]
    fn test_invalid_length() {
        let values = buffer![15_u32, 135, 13531, 42].into_array();
        let indices = buffer![10_u64, 11, 50, 100].into_array();

        Sparse::try_new(indices, values, 100, 0_u32.into()).unwrap();
    }

    #[test]
    fn test_valid_length() {
        let values = buffer![15_u32, 135, 13531, 42].into_array();
        let indices = buffer![10_u64, 11, 50, 100].into_array();

        Sparse::try_new(indices, values, 101, 0_u32.into()).unwrap();
    }

    #[test]
    fn encode_with_nulls() {
        let original = PrimitiveArray::new(
            buffer![0i32, 1, 2, 3, 3, 3, 3, 3, 3, 3, 4, 4],
            Validity::from_iter(vec![
                true, true, false, true, false, true, false, true, true, false, true, false,
            ]),
        );
        let sparse = Sparse::encode(&original.clone().into_array(), None)
            .vortex_expect("Sparse::encode should succeed for test data");
        assert_eq!(
            sparse
                .validity()
                .unwrap()
                .to_mask(sparse.len(), &mut LEGACY_SESSION.create_execution_ctx())
                .unwrap(),
            Mask::from_iter(vec![
                true, true, false, true, false, true, false, true, true, false, true, false,
            ])
        );
        #[expect(deprecated)]
        let sparse_primitive = sparse.to_primitive();
        assert_arrays_eq!(sparse_primitive, original);
    }

    #[test]
    fn validity_mask_includes_null_values_when_fill_is_null() {
        let indices = buffer![0u8, 2, 4, 6, 8].into_array();
        let values = PrimitiveArray::from_option_iter([Some(0i16), Some(1), None, None, Some(4)])
            .into_array();
        let array = Sparse::try_new(indices, values, 10, Scalar::null_native::<i16>()).unwrap();
        let actual = array
            .validity()
            .unwrap()
            .to_mask(array.len(), &mut LEGACY_SESSION.create_execution_ctx())
            .unwrap();
        let expected = Mask::from_iter([
            true, false, true, false, false, false, false, false, true, false,
        ]);

        assert_eq!(actual, expected);
    }
}
