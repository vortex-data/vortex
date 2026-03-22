// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;

use itertools::Itertools;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayRef;
use vortex_array::DeserializeMetadata;
use vortex_array::DynArray;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionStep;
use vortex_array::IntoArray;
use vortex_array::Precision;
use vortex_array::ProstMetadata;
use vortex_array::SerializeMetadata;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::patches::Patches;
use vortex_array::patches::PatchesMetadata;
use vortex_array::serde::ArrayChildren;
use vortex_array::stats::ArrayStats;
use vortex_array::stats::StatsSetRef;
use vortex_array::validity::Validity;
use vortex_array::vtable;
use vortex_array::vtable::ArrayId;
use vortex_array::vtable::VTable;
use vortex_array::vtable::ValidityChild;
use vortex_array::vtable::ValidityVTableFromChild;
use vortex_array::vtable::patches_child;
use vortex_array::vtable::patches_child_name;
use vortex_array::vtable::patches_nchildren;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_mask::Mask;
use vortex_session::VortexSession;

use crate::alp_rd::kernel::PARENT_KERNELS;
use crate::alp_rd::rules::RULES;
use crate::alp_rd_decode;

vtable!(ALPRD);

#[derive(Clone, prost::Message)]
pub struct ALPRDMetadata {
    #[prost(uint32, tag = "1")]
    right_bit_width: u32,
    #[prost(uint32, tag = "2")]
    dict_len: u32,
    #[prost(uint32, repeated, tag = "3")]
    dict: Vec<u32>,
    #[prost(enumeration = "PType", tag = "4")]
    left_parts_ptype: i32,
    #[prost(message, tag = "5")]
    patches: Option<PatchesMetadata>,
}

impl VTable for ALPRD {
    type Array = ALPRDArray;

    type Metadata = ProstMetadata<ALPRDMetadata>;
    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn vtable(_array: &Self::Array) -> &Self {
        &ALPRD
    }

    fn id(&self) -> ArrayId {
        Self::ID
    }

    fn len(array: &ALPRDArray) -> usize {
        array.left_parts.len()
    }

    fn dtype(array: &ALPRDArray) -> &DType {
        &array.dtype
    }

    fn stats(array: &ALPRDArray) -> StatsSetRef<'_> {
        array.stats_set.to_ref(array.as_ref())
    }

    fn array_hash<H: std::hash::Hasher>(array: &ALPRDArray, state: &mut H, precision: Precision) {
        array.dtype.hash(state);
        array.left_parts.array_hash(state, precision);
        array.left_parts_dictionary.array_hash(state, precision);
        array.right_parts.array_hash(state, precision);
        array.right_bit_width.hash(state);
        array.left_parts_patches.array_hash(state, precision);
    }

    fn array_eq(array: &ALPRDArray, other: &ALPRDArray, precision: Precision) -> bool {
        array.dtype == other.dtype
            && array.left_parts.array_eq(&other.left_parts, precision)
            && array
                .left_parts_dictionary
                .array_eq(&other.left_parts_dictionary, precision)
            && array.right_parts.array_eq(&other.right_parts, precision)
            && array.right_bit_width == other.right_bit_width
            && array
                .left_parts_patches
                .array_eq(&other.left_parts_patches, precision)
    }

    fn nbuffers(_array: &ALPRDArray) -> usize {
        0
    }

    fn buffer(_array: &ALPRDArray, idx: usize) -> BufferHandle {
        vortex_panic!("ALPRDArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: &ALPRDArray, _idx: usize) -> Option<String> {
        None
    }

    fn nchildren(array: &ALPRDArray) -> usize {
        2 + array.left_parts_patches().map_or(0, patches_nchildren)
    }

    fn child(array: &ALPRDArray, idx: usize) -> ArrayRef {
        match idx {
            0 => array.left_parts().clone(),
            1 => array.right_parts().clone(),
            _ => {
                let patches = array
                    .left_parts_patches()
                    .unwrap_or_else(|| vortex_panic!("ALPRDArray child index {idx} out of bounds"));
                patches_child(patches, idx - 2)
            }
        }
    }

    fn child_name(array: &ALPRDArray, idx: usize) -> String {
        match idx {
            0 => "left_parts".to_string(),
            1 => "right_parts".to_string(),
            _ => {
                if array.left_parts_patches().is_none() {
                    vortex_panic!("ALPRDArray child_name index {idx} out of bounds");
                }
                patches_child_name(idx - 2).to_string()
            }
        }
    }

    fn metadata(array: &ALPRDArray) -> VortexResult<Self::Metadata> {
        let dict = array
            .left_parts_dictionary()
            .iter()
            .map(|&i| i as u32)
            .collect::<Vec<_>>();

        Ok(ProstMetadata(ALPRDMetadata {
            right_bit_width: array.right_bit_width() as u32,
            dict_len: array.left_parts_dictionary().len() as u32,
            dict,
            left_parts_ptype: array.left_parts.dtype().as_ptype() as i32,
            patches: array
                .left_parts_patches()
                .map(|p| p.to_metadata(array.len(), array.left_parts().dtype()))
                .transpose()?,
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
        Ok(ProstMetadata(
            <ProstMetadata<ALPRDMetadata> as DeserializeMetadata>::deserialize(bytes)?,
        ))
    }

    fn build(
        dtype: &DType,
        len: usize,
        metadata: &Self::Metadata,
        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
    ) -> VortexResult<ALPRDArray> {
        if children.len() < 2 {
            vortex_bail!(
                "Expected at least 2 children for ALPRD encoding, found {}",
                children.len()
            );
        }

        let left_parts_dtype = DType::Primitive(metadata.0.left_parts_ptype(), dtype.nullability());
        let left_parts = children.get(0, &left_parts_dtype, len)?;
        let left_parts_dictionary: Buffer<u16> = metadata.0.dict.as_slice()
            [0..metadata.0.dict_len as usize]
            .iter()
            .map(|&i| {
                u16::try_from(i)
                    .map_err(|_| vortex_err!("left_parts_dictionary code {i} does not fit in u16"))
            })
            .try_collect()?;

        let right_parts_dtype = match &dtype {
            DType::Primitive(PType::F32, _) => {
                DType::Primitive(PType::U32, Nullability::NonNullable)
            }
            DType::Primitive(PType::F64, _) => {
                DType::Primitive(PType::U64, Nullability::NonNullable)
            }
            _ => vortex_bail!("Expected f32 or f64 dtype, got {:?}", dtype),
        };
        let right_parts = children.get(1, &right_parts_dtype, len)?;

        let left_parts_patches = metadata
            .0
            .patches
            .map(|p| {
                let indices = children.get(2, &p.indices_dtype()?, p.len()?)?;
                let values = children.get(3, &left_parts_dtype, p.len()?)?;

                Patches::new(
                    len,
                    p.offset()?,
                    indices,
                    values,
                    // TODO(0ax1): handle chunk offsets
                    None,
                )
            })
            .transpose()?;

        ALPRDArray::try_new(
            dtype.clone(),
            left_parts,
            left_parts_dictionary,
            right_parts,
            u8::try_from(metadata.0.right_bit_width).map_err(|_| {
                vortex_err!(
                    "right_bit_width {} out of u8 range",
                    metadata.0.right_bit_width
                )
            })?,
            left_parts_patches,
        )
    }

    fn with_children(array: &mut Self::Array, children: Vec<ArrayRef>) -> VortexResult<()> {
        // Children: left_parts, right_parts, patches (if present): indices, values
        let patches_info = array
            .left_parts_patches
            .as_ref()
            .map(|p| (p.array_len(), p.offset()));

        let expected_children = if patches_info.is_some() { 4 } else { 2 };

        vortex_ensure!(
            children.len() == expected_children,
            "ALPRDArray expects {} children, got {}",
            expected_children,
            children.len()
        );

        let mut children_iter = children.into_iter();
        array.left_parts = children_iter
            .next()
            .ok_or_else(|| vortex_err!("Expected left_parts child"))?;
        array.right_parts = children_iter
            .next()
            .ok_or_else(|| vortex_err!("Expected right_parts child"))?;

        if let Some((array_len, offset)) = patches_info {
            let indices = children_iter
                .next()
                .ok_or_else(|| vortex_err!("Expected patch indices child"))?;
            let values = children_iter
                .next()
                .ok_or_else(|| vortex_err!("Expected patch values child"))?;

            array.left_parts_patches = Some(Patches::new(
                array_len, offset, indices, values,
                None, // chunk_offsets not currently supported for ALPRD
            )?);
        }

        Ok(())
    }

    fn execute(array: &Self::Array, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionStep> {
        let left_parts = array.left_parts().clone().execute::<PrimitiveArray>(ctx)?;
        let right_parts = array.right_parts().clone().execute::<PrimitiveArray>(ctx)?;

        // Decode the left_parts using our builtin dictionary.
        let left_parts_dict = array.left_parts_dictionary();

        let validity = array
            .left_parts()
            .validity()?
            .to_array(array.len())
            .execute::<Mask>(ctx)?;

        let decoded_array = if array.is_f32() {
            PrimitiveArray::new(
                alp_rd_decode::<f32>(
                    left_parts.into_buffer::<u16>(),
                    left_parts_dict,
                    array.right_bit_width,
                    right_parts.into_buffer_mut::<u32>(),
                    array.left_parts_patches(),
                    ctx,
                )?,
                Validity::from_mask(validity, array.dtype().nullability()),
            )
        } else {
            PrimitiveArray::new(
                alp_rd_decode::<f64>(
                    left_parts.into_buffer::<u16>(),
                    left_parts_dict,
                    array.right_bit_width,
                    right_parts.into_buffer_mut::<u64>(),
                    array.left_parts_patches(),
                    ctx,
                )?,
                Validity::from_mask(validity, array.dtype().nullability()),
            )
        };

        Ok(ExecutionStep::Done(decoded_array.into_array()))
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
}

#[derive(Clone, Debug)]
pub struct ALPRDArray {
    dtype: DType,
    left_parts: ArrayRef,
    left_parts_patches: Option<Patches>,
    left_parts_dictionary: Buffer<u16>,
    right_parts: ArrayRef,
    right_bit_width: u8,
    stats_set: ArrayStats,
}

#[derive(Debug)]
pub struct ALPRD;

impl ALPRD {
    pub const ID: ArrayId = ArrayId::new_ref("vortex.alprd");
}

impl ALPRDArray {
    /// Build a new `ALPRDArray` from components.
    pub fn try_new(
        dtype: DType,
        left_parts: ArrayRef,
        left_parts_dictionary: Buffer<u16>,
        right_parts: ArrayRef,
        right_bit_width: u8,
        left_parts_patches: Option<Patches>,
    ) -> VortexResult<Self> {
        if !dtype.is_float() {
            vortex_bail!("ALPRDArray given invalid DType ({dtype})");
        }

        let len = left_parts.len();
        if right_parts.len() != len {
            vortex_bail!(
                "left_parts (len {}) and right_parts (len {}) must be of same length",
                len,
                right_parts.len()
            );
        }

        if !left_parts.dtype().is_unsigned_int() {
            vortex_bail!("left_parts dtype must be uint");
        }
        // we delegate array validity to the left_parts child
        if dtype.is_nullable() != left_parts.dtype().is_nullable() {
            vortex_bail!(
                "ALPRDArray dtype nullability ({}) must match left_parts dtype nullability ({})",
                dtype,
                left_parts.dtype()
            );
        }

        // we enforce right_parts to be non-nullable uint
        if !right_parts.dtype().is_unsigned_int() || right_parts.dtype().is_nullable() {
            vortex_bail!(MismatchedTypes: "non-nullable uint", right_parts.dtype());
        }

        let left_parts_patches = left_parts_patches
            .map(|patches| {
                if !patches.values().all_valid()? {
                    vortex_bail!("patches must be all valid: {}", patches.values());
                }
                // TODO(ngates): assert the DType, don't cast it.
                // TODO(joe): assert the DType, don't cast it in the next PR.
                let mut patches = patches.cast_values(left_parts.dtype())?;
                // Force execution of the lazy cast so patch values are materialized
                // before serialization.
                *patches.values_mut() = patches.values().to_canonical()?.into_array();
                Ok(patches)
            })
            .transpose()?;

        Ok(Self {
            dtype,
            left_parts,
            left_parts_dictionary,
            right_parts,
            right_bit_width,
            left_parts_patches,
            stats_set: Default::default(),
        })
    }

    /// Build a new `ALPRDArray` from components. This does not perform any validation, and instead
    /// it constructs it from parts.
    pub(crate) unsafe fn new_unchecked(
        dtype: DType,
        left_parts: ArrayRef,
        left_parts_dictionary: Buffer<u16>,
        right_parts: ArrayRef,
        right_bit_width: u8,
        left_parts_patches: Option<Patches>,
    ) -> Self {
        Self {
            dtype,
            left_parts,
            left_parts_patches,
            left_parts_dictionary,
            right_parts,
            right_bit_width,
            stats_set: Default::default(),
        }
    }

    /// Returns true if logical type of the array values is f32.
    ///
    /// Returns false if the logical type of the array values is f64.
    #[inline]
    pub fn is_f32(&self) -> bool {
        matches!(&self.dtype, DType::Primitive(PType::F32, _))
    }

    /// The leftmost (most significant) bits of the floating point values stored in the array.
    ///
    /// These are bit-packed and dictionary encoded, and cannot directly be interpreted without
    /// the metadata of this array.
    pub fn left_parts(&self) -> &ArrayRef {
        &self.left_parts
    }

    /// The rightmost (least significant) bits of the floating point values stored in the array.
    pub fn right_parts(&self) -> &ArrayRef {
        &self.right_parts
    }

    #[inline]
    pub fn right_bit_width(&self) -> u8 {
        self.right_bit_width
    }

    /// Patches of left-most bits.
    pub fn left_parts_patches(&self) -> Option<&Patches> {
        self.left_parts_patches.as_ref()
    }

    /// The dictionary that maps the codes in `left_parts` into bit patterns.
    #[inline]
    pub fn left_parts_dictionary(&self) -> &Buffer<u16> {
        &self.left_parts_dictionary
    }

    pub fn replace_left_parts_patches(&mut self, patches: Option<Patches>) {
        self.left_parts_patches = patches;
    }
}

impl ValidityChild<ALPRD> for ALPRD {
    fn validity_child(array: &ALPRDArray) -> &ArrayRef {
        array.left_parts()
    }
}

#[cfg(test)]
mod test {
    use rstest::rstest;
    use vortex_array::ProstMetadata;
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::PType;
    use vortex_array::patches::PatchesMetadata;
    use vortex_array::test_harness::check_metadata;

    use super::ALPRDMetadata;
    use crate::ALPRDFloat;
    use crate::alp_rd;

    #[rstest]
    #[case(vec![0.1f32.next_up(); 1024], 1.123_848_f32)]
    #[case(vec![0.1f64.next_up(); 1024], 1.123_848_591_110_992_f64)]
    fn test_array_encode_with_nulls_and_patches<T: ALPRDFloat>(
        #[case] reals: Vec<T>,
        #[case] seed: T,
    ) {
        assert_eq!(reals.len(), 1024, "test expects 1024-length fixture");
        // Null out some of the values.
        let mut reals: Vec<Option<T>> = reals.into_iter().map(Some).collect();
        reals[1] = None;
        reals[5] = None;
        reals[900] = None;

        // Create a new array from this.
        let real_array = PrimitiveArray::from_option_iter(reals.iter().cloned());

        // Pick a seed that we know will trigger lots of patches.
        let encoder: alp_rd::RDEncoder = alp_rd::RDEncoder::new(&[seed.powi(-2)]);

        let rd_array = encoder.encode(&real_array);

        let decoded = rd_array.to_primitive();

        assert_arrays_eq!(decoded, PrimitiveArray::from_option_iter(reals));
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn test_alprd_metadata() {
        check_metadata(
            "alprd.metadata",
            ProstMetadata(ALPRDMetadata {
                right_bit_width: u32::MAX,
                patches: Some(PatchesMetadata::new(
                    usize::MAX,
                    usize::MAX,
                    PType::U64,
                    None,
                    None,
                    None,
                )),
                dict: Vec::new(),
                left_parts_ptype: PType::U64 as i32,
                dict_len: 8,
            }),
        );
    }
}
