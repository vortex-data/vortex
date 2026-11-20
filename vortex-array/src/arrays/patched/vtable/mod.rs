// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayEq;
use crate::ArrayHash;
mod kernels;
mod operations;
mod plugin;
mod slice;

use std::hash::Hash;
use std::hash::Hasher;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::Canonical;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::array::with_empty_buffers;
use crate::arrays::Constant;
use crate::arrays::Primitive;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::arrays::patched::PatchedData;
use crate::arrays::patched::PatchedSlots;
use crate::arrays::patched::PatchedSlotsView;
use crate::arrays::patched::execute::PatchedCanonicalParts;
use crate::arrays::patched::execute::PatchedInner;
use crate::arrays::patched::execute::execute_canonical;
use crate::arrays::patched::execute::scatter;
use crate::arrays::patched::execute::to_canonical;
use crate::buffer::BufferHandle;
use crate::builders::ArrayBuilder;
use crate::builders::PrimitiveBuilder;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::match_each_native_ptype;
use crate::matcher::Matcher;
use crate::require_child;
use crate::validity::Validity;

/// A [`Patched`]-encoded Vortex array.
pub type PatchedArray = Array<Patched>;

pub(crate) fn initialize(session: &VortexSession) {
    kernels::initialize(session);
}

#[derive(Clone, Debug)]
pub struct Patched;

/// Matches canonical and constant arrays: the forms of the inner array that a patched array
/// executes from.
struct CanonicalOrConstant;

impl Matcher for CanonicalOrConstant {
    type Match<'a> = &'a ArrayRef;

    fn try_match(array: &ArrayRef) -> Option<Self::Match<'_>> {
        (array.is::<Constant>() || AnyCanonical::matches(array)).then_some(array)
    }
}

impl ValidityVTable<Patched> for Patched {
    fn validity(array: ArrayView<'_, Patched>) -> VortexResult<Validity> {
        let inner = array.inner().validity()?;
        let values = array.patch_values().validity()?;
        if inner.definitely_no_nulls() && values.definitely_no_nulls() {
            return Ok(Validity::AllValid);
        }
        if inner.definitely_all_null() && values.definitely_all_null() {
            return Ok(Validity::AllInvalid);
        }

        // The validity is itself a lazily patched array of the children's validity bits.
        let slots = PatchedSlots {
            inner: inner.to_array(array.len()),
            lane_offsets: array.lane_offsets().clone(),
            patch_indices: array.patch_indices().clone(),
            patch_values: values.to_array(array.patch_values().len()),
        }
        .into_slots();
        // SAFETY: the validity arrays have the lengths of the children they describe, and share
        // their layout.
        let validity = unsafe {
            Patched::new_unchecked(
                DType::Bool(Nullability::NonNullable),
                array.len(),
                slots,
                array.n_lanes(),
                array.offset(),
            )
        };
        Ok(Validity::Array(validity.into_array()))
    }
}

#[derive(Clone, prost::Message)]
pub struct PatchedMetadata {
    /// The total number of patches, and the length of the indices and values child arrays.
    #[prost(uint32, tag = "1")]
    pub(crate) n_patches: u32,

    /// The number of lanes used for patch indexing. Must be a power of two between 1 and 1024.
    #[prost(uint32, tag = "2")]
    pub(crate) n_lanes: u32,

    /// An offset into the first chunk's patches that should be considered in-view.
    ///
    /// Always between 0 and 1023.
    #[prost(uint32, tag = "3")]
    pub(crate) offset: u32,
}

impl ArrayHash for PatchedData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.offset.hash(state);
        self.n_lanes.hash(state);
    }
}

impl ArrayEq for PatchedData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.offset == other.offset && self.n_lanes == other.n_lanes
    }
}

impl VTable for Patched {
    type TypedArrayData = PatchedData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.patched");
        *ID
    }

    fn validate(
        &self,
        data: &PatchedData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        data.validate(dtype, len, &PatchedSlotsView::from_slots(slots))
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("invalid buffer index for PatchedArray: {idx}");
    }

    fn buffer_name(_array: ArrayView<'_, Self>, idx: usize) -> Option<String> {
        vortex_panic!("invalid buffer index for PatchedArray: {idx}");
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn append_to_builder(
        array: ArrayView<'_, Self>,
        builder: &mut dyn ArrayBuilder,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let dtype = array.array().dtype();

        // Primitive values are appended from the inner array and then overwritten in place, as
        // long as the patches cannot change the validity of the appended values.
        let fast_path = dtype.is_primitive()
            && array.inner().validity()?.definitely_no_nulls()
            && array.patch_values().validity()?.definitely_no_nulls();
        if !fast_path {
            let canonical = array
                .array()
                .clone()
                .execute::<Canonical>(ctx)?
                .into_array();
            return canonical.append_to_builder(builder, ctx);
        }

        let len = array.len();
        array.inner().append_to_builder(builder, ctx)?;

        let layout = array.execute_layout(ctx)?;
        let layout = layout.layout()?;
        let values = array
            .patch_values()
            .clone()
            .execute::<crate::arrays::PrimitiveArray>(ctx)?;

        match_each_native_ptype!(dtype.as_ptype(), |V| {
            let typed_builder = builder
                .as_any_mut()
                .downcast_mut::<PrimitiveBuilder<V>>()
                .vortex_expect("correctly typed builder");

            // Overwrite the last `len` elements of the builder, which were just appended from
            // the inner array.
            let output = typed_builder.values_mut();
            let trailer = output.len() - len;
            scatter(&mut output[trailer..], values.as_slice::<V>(), &layout);
        });

        Ok(())
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        PatchedSlots::NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        // A constant inner array is patched directly, without being materialized first. That
        // includes the constant results of functions pushed down into a constant inner array.
        let array =
            require_child!(array, array.inner(), PatchedSlots::INNER => CanonicalOrConstant);
        let array =
            require_child!(array, array.lane_offsets(), PatchedSlots::LANE_OFFSETS => Primitive);
        let array =
            require_child!(array, array.patch_indices(), PatchedSlots::PATCH_INDICES => Primitive);
        let array =
            require_child!(array, array.patch_values(), PatchedSlots::PATCH_VALUES => AnyCanonical);

        let layout = array.execute_layout(ctx)?;
        let layout = layout.layout()?;
        let dtype = array.dtype().clone();
        let slots = match array.try_into_parts() {
            Ok(parts) => PatchedSlots::from_slots(parts.slots),
            Err(array) => array.slots_view().to_owned(),
        };

        let result = execute_canonical(
            PatchedCanonicalParts {
                dtype,
                layout,
                lane_offsets: slots.lane_offsets,
                patch_indices: slots.patch_indices,
                inner: match slots.inner.as_opt::<Constant>() {
                    Some(constant) => PatchedInner::Constant(constant.scalar().clone()),
                    None => PatchedInner::Canonical(to_canonical(slots.inner, ctx)?),
                },
                values: to_canonical(slots.patch_values, ctx)?,
            },
            ctx,
        )?;
        Ok(ExecutionResult::done(result))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_buffer::ByteBufferMut;
    use vortex_buffer::buffer;
    use vortex_buffer::buffer_mut;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use crate::Array;
    use crate::ArrayContext;
    use crate::ArrayParts;
    use crate::ArraySlots;
    use crate::IntoArray;
    use crate::VortexSessionExecute;
    use crate::arrays::Patched;
    use crate::arrays::PatchedArray;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::patched::PatchedArrayExt;
    use crate::arrays::patched::PatchedArraySlotsExt;
    use crate::arrays::patched::PatchedData;
    use crate::arrays::patched::PatchedSlots;
    use crate::arrays::patched::PatchedSlotsView;
    use crate::assert_arrays_eq;
    use crate::builders::builder_with_capacity_in;
    use crate::optimizer::ArrayOptimizer;
    use crate::patches::Patches;
    use crate::serde::SerializeOptions;
    use crate::serde::SerializedArray;
    use crate::session::ArraySessionExt;
    use crate::validity::Validity;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[test]
    fn test_execute() {
        let mut ctx = SESSION.create_execution_ctx();
        let values = buffer![0u16; 1024].into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![1u16; 3].into_array(),
            None,
            &mut ctx,
        )
        .unwrap();

        let mut ctx = SESSION.create_execution_ctx();

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array();

        let mut expected = buffer_mut![0u16; 1024];
        expected[1] = 1;
        expected[2] = 1;
        expected[3] = 1;

        assert_arrays_eq!(array, expected.into_array(), &mut ctx);
    }

    #[test]
    fn test_execute_sliced() {
        let mut ctx = SESSION.create_execution_ctx();
        let values = buffer![0u16; 1024].into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![1u16; 3].into_array(),
            None,
            &mut ctx,
        )
        .unwrap();

        let mut ctx = SESSION.create_execution_ctx();

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array()
            .slice(3..1024)
            .unwrap();

        let mut expected = buffer_mut![0u16; 1021];
        expected[0] = 1;

        assert_arrays_eq!(array, expected.into_array(), &mut ctx);
    }

    #[test]
    fn test_append_to_builder_non_nullable() {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::new(buffer![0u16; 1024], Validity::NonNullable).into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![10u16, 20, 30].into_array(),
            None,
            &mut ctx,
        )
        .unwrap();

        let mut ctx = SESSION.create_execution_ctx();

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array();

        let mut builder = builder_with_capacity_in(
            array.dtype(),
            array.len(),
            vortex_buffer::BufferAllocatorRef::static_ref(),
        );
        array.append_to_builder(builder.as_mut(), &mut ctx).unwrap();

        let result = builder.finish();

        let mut expected = buffer_mut![0u16; 1024];
        expected[1] = 10;
        expected[2] = 20;
        expected[3] = 30;
        let expected = expected.into_array();

        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn test_append_to_builder_sliced() {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::new(buffer![0u16; 1024], Validity::NonNullable).into_array();
        let patches = Patches::new(
            1024,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![10u16, 20, 30].into_array(),
            None,
            &mut ctx,
        )
        .unwrap();

        let mut ctx = SESSION.create_execution_ctx();

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array()
            .slice(3..1024)
            .unwrap();

        let mut builder = builder_with_capacity_in(
            array.dtype(),
            array.len(),
            vortex_buffer::BufferAllocatorRef::static_ref(),
        );
        array.append_to_builder(builder.as_mut(), &mut ctx).unwrap();

        let result = builder.finish();

        let mut expected = buffer_mut![0u16; 1021];
        expected[0] = 30;
        let expected = expected.into_array();

        assert_arrays_eq!(result, expected, &mut ctx);
    }

    #[test]
    fn test_append_to_builder_with_validity() {
        let mut ctx = SESSION.create_execution_ctx();
        // Create inner array with nulls at indices 0 and 5.
        let validity = Validity::from_iter((0..10).map(|i| i != 0 && i != 5));
        let values = PrimitiveArray::new(buffer![0u16; 10], validity).into_array();

        // Apply patches at indices 1, 2, 3.
        let patches = Patches::new(
            10,
            0,
            buffer![1u32, 2, 3].into_array(),
            buffer![10u16, 20, 30].into_array(),
            None,
            &mut ctx,
        )
        .unwrap();

        let mut ctx = SESSION.create_execution_ctx();

        let array = Patched::from_array_and_patches(values, &patches, &mut ctx)
            .unwrap()
            .into_array();

        let mut builder = builder_with_capacity_in(
            array.dtype(),
            array.len(),
            vortex_buffer::BufferAllocatorRef::static_ref(),
        );
        array.append_to_builder(builder.as_mut(), &mut ctx).unwrap();

        let result = builder.finish();

        // Expected: null at 0, patched 10/20/30 at 1/2/3, zero at 4, null at 5, zeros at 6-9.
        let expected = PrimitiveArray::from_option_iter([
            None,
            Some(10u16),
            Some(20),
            Some(30),
            Some(0),
            None,
            Some(0),
            Some(0),
            Some(0),
            Some(0),
        ])
        .into_array();

        assert_arrays_eq!(result, expected, &mut ctx);
    }

    fn make_patched_array(
        inner: impl IntoIterator<Item = u16>,
        patch_indices: &[u32],
        patch_values: &[u16],
    ) -> VortexResult<PatchedArray> {
        let mut ctx = SESSION.create_execution_ctx();
        let values: Vec<u16> = inner.into_iter().collect();
        let len = values.len();
        let array = PrimitiveArray::from_iter(values).into_array();

        let indices = PrimitiveArray::from_iter(patch_indices.iter().copied()).into_array();
        let patch_vals = PrimitiveArray::from_iter(patch_values.iter().copied()).into_array();

        let patches = Patches::new(len, 0, indices, patch_vals, None, &mut ctx)?;

        let mut ctx = SESSION.create_execution_ctx();

        Patched::from_array_and_patches(array, &patches, &mut ctx)
    }

    #[rstest]
    #[case::basic(
        make_patched_array(vec![0u16; 1024], &[1, 2, 3], &[10, 20, 30]).unwrap().into_array(),
        (0..1024).map(|i| match i { 1 => 10u16, 2 => 20, 3 => 30, _ => 0 }).collect::<Vec<_>>()
    )]
    #[case::multi_chunk(
        make_patched_array(vec![0u16; 4096], &[100, 1500, 2500, 3500], &[11, 22, 33, 44]).unwrap().into_array(),
        (0..4096).map(|i| match i { 100 => 11u16, 1500 => 22, 2500 => 33, 3500 => 44, _ => 0 }).collect::<Vec<_>>()
    )]
    #[case::sliced({
        let arr = make_patched_array(vec![0u16; 1024], &[1, 2, 3], &[10, 20, 30]).unwrap();
        arr.into_array().slice(2..1024).unwrap().optimize_ctx(&SESSION).unwrap()
    }, (2..1024).map(|i| match i { 2 => 20u16, 3 => 30, _ => 0 }).collect::<Vec<_>>())]
    fn test_serde_roundtrip(#[case] array: crate::ArrayRef, #[case] expected: Vec<u16>) {
        let dtype = array.dtype().clone();
        let len = array.len();

        let session = VortexSession::empty();
        session.arrays().register(Patched);

        let ctx = ArrayContext::empty().with_allowed_ids(
            session
                .arrays()
                .registry()
                .read(|map| map.keys().copied().collect()),
        );
        let serialized = array
            .serialize(&ctx, &session, &SerializeOptions::default())
            .unwrap();

        // Concat into a single buffer.
        let mut concat = ByteBufferMut::empty();
        for buf in serialized {
            concat.extend_from_slice(buf.as_ref());
        }
        let concat = concat.freeze();

        let parts = SerializedArray::try_from(concat).unwrap();
        let decoded = parts
            .decode(&dtype, len, &ReadContext::new(ctx.to_ids()), &session)
            .unwrap();

        assert!(decoded.is::<Patched>());
        let expected = PrimitiveArray::from_iter(expected);
        let mut exec_ctx = session.create_execution_ctx();
        assert_arrays_eq!(array, &expected, &mut exec_ctx);
        assert_arrays_eq!(decoded, expected, &mut exec_ctx);
    }

    #[test]
    fn test_with_slots_basic() -> VortexResult<()> {
        let array = make_patched_array(vec![0u16; 1024], &[1, 2, 3], &[10, 20, 30])?;

        // Get original children via accessor methods
        let slots = PatchedSlots::from_slots(
            array
                .as_array()
                .slots()
                .iter()
                .cloned()
                .collect::<ArraySlots>(),
        );
        let view = PatchedSlotsView::from_slots(array.as_array().slots());
        assert_eq!(view.inner.len(), array.inner().len());

        // Create new PatchedArray with same children using with_slots
        let array_ref = array.into_array();
        // SAFETY: the replacement slots are the original children, preserving logical values and
        // parent statistics.
        let new_array = unsafe { array_ref.clone().with_slots(slots.into_slots()) }?;

        assert!(new_array.is::<Patched>());
        assert_eq!(array_ref.len(), new_array.len());
        assert_eq!(array_ref.dtype(), new_array.dtype());

        let mut ctx = SESSION.create_execution_ctx();

        let expected = PrimitiveArray::from_iter((0..1024).map(|i| match i {
            1 => 10u16,
            2 => 20,
            3 => 30,
            _ => 0,
        }));
        assert_arrays_eq!(new_array, expected, &mut ctx);

        Ok(())
    }

    #[test]
    fn test_rebuild_modified_inner_from_parts() -> VortexResult<()> {
        let array = make_patched_array(vec![0u16; 10], &[1, 2, 3], &[10, 20, 30])?;

        // Create a different inner array (all 5s instead of 0s)
        let new_inner = PrimitiveArray::from_iter(vec![5u16; 10]).into_array();
        let slots = PatchedSlots {
            inner: new_inner,
            lane_offsets: array.lane_offsets().clone(),
            patch_indices: array.patch_indices().clone(),
            patch_values: array.patch_values().clone(),
        };

        let data = PatchedData {
            n_lanes: array.n_lanes(),
            offset: array.offset(),
        };
        let new_array = Array::try_from_parts(ArrayParts::new(
            Patched,
            array.dtype().clone(),
            array.len(),
            data,
            slots.into_slots(),
        ))?
        .into_array();

        // Execute and verify the inner values changed (except at patch positions)
        let mut ctx = SESSION.create_execution_ctx();

        // Expected: all 5s except indices 1, 2, 3 which are patched to 10, 20, 30
        let expected = PrimitiveArray::from_iter([5u16, 10, 20, 30, 5, 5, 5, 5, 5, 5]);
        assert_arrays_eq!(new_array, expected, &mut ctx);

        Ok(())
    }
}
