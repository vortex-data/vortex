// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Assertions for block fusion, read from the execution trace. See [`crate::block`].

use std::marker::PhantomData;

use smallvec::smallvec;
use vortex_buffer::Buffer;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use super::trace::trace_op;
use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::IntoArray;
use crate::VTable;
use crate::VortexSessionExecute;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::EmptyArrayData;
use crate::array::ValidityChild;
use crate::array::ValidityVTableFromChild;
use crate::array::vtable::NotSupported;
use crate::array::vtable::with_empty_buffers;
use crate::arrays::Primitive;
use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::PrimitiveArrayExt;
use crate::assert_arrays_eq;
use crate::block::Block;
use crate::block::BlockOp;
use crate::block::Written;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::half::f16;
use crate::match_each_native_ptype;
use crate::require_spine;

/// Execute `array` to canonical with block fusion off, then on, assert both give the same values
/// and dtype, and return the fusion steps of the fused run's trace: see [`fusion_lines`].
///
/// The fused run takes `array` itself, so a leaf it alone owns can be decoded in place.
pub fn fusion_trace(array: ArrayRef, session: &VortexSession) -> VortexResult<String> {
    let expected = array
        .clone()
        .execute::<Canonical>(&mut session.create_execution_ctx().with_block_fusion(false))?
        .into_array();
    let mut ctx = session.create_execution_ctx().with_block_fusion(true);
    let traced = trace_op(|| array.execute::<Canonical>(&mut ctx))?;
    let output = traced.output.into_array();
    vortex_ensure_eq!(output.dtype(), expected.dtype());
    assert_arrays_eq!(output, expected, &mut ctx);
    Ok(fusion_lines(&traced.trace.to_string()))
}

/// The fusion steps of a rendered trace: each slot request (`ExecuteSpine` for a spine,
/// `ExecuteSlot` otherwise), `fuse start`, `fuse extend` and `drive`, and how each op ran over the
/// first block. Each `drive` line is one fused chain, and an op over its leaf's own buffer shows
/// `leaf -> leaf`.
pub fn fusion_lines(trace: &str) -> String {
    trace
        .lines()
        .map(str::trim)
        .filter(|line| {
            ["ExecuteSpine", "ExecuteSlot", "fuse ", "drive ", "block_op"]
                .iter()
                .any(|prefix| line.starts_with(prefix))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The number of chains the executor drove in `fusion_lines`.
pub fn fused_chains(fusion_lines: &str) -> usize {
    fusion_lines
        .lines()
        .filter(|line| line.starts_with("drive "))
        .count()
}

/// Wrap a primitive `child` in a fusable test parent that transforms every value by one step and
/// preserves validity, so source-only encoding crates can exercise `VTable::block_source` end to
/// end without depending on another encoding crate.
pub fn fuse_parent(child: ArrayRef) -> ArrayRef {
    let dtype = child.dtype().clone();
    let len = child.len();
    Array::<TestSpine>::try_from_parts(ArrayParts::new(
        TestSpine,
        dtype,
        len,
        EmptyArrayData,
        smallvec![Some(child)],
    ))
    .unwrap_or_else(|err| vortex_panic!("fuse_parent: {err}"))
    .into_array()
}

#[derive(Clone, Debug)]
struct TestSpine;

impl ValidityChild<TestSpine> for TestSpine {
    fn validity_child(array: ArrayView<'_, TestSpine>) -> ArrayRef {
        array.slots()[0]
            .clone()
            .unwrap_or_else(|| vortex_panic!("TestSpine child"))
    }
}

impl VTable for TestSpine {
    type TypedArrayData = EmptyArrayData;
    type OperationsVTable = NotSupported;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.test.spine");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(dtype.is_primitive(), "TestSpine requires primitive dtype");
        vortex_ensure_eq!(slots.len(), 1);
        if let Some(child) = &slots[0] {
            vortex_ensure_eq!(child.dtype(), dtype);
            vortex_ensure_eq!(child.len(), len);
        }
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("TestSpine buffer {idx}")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn with_buffers(
        &self,
        array: ArrayView<'_, Self>,
        buffers: &[BufferHandle],
    ) -> VortexResult<ArrayParts<Self>> {
        with_empty_buffers(self, array, buffers)
    }

    fn slot_name(_array: ArrayView<'_, Self>, _idx: usize) -> String {
        "child".to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let child = array.slots()[0]
            .clone()
            .unwrap_or_else(|| vortex_panic!("TestSpine child"));
        let array = require_spine!(array, child, 0 => Primitive);
        let child = array.slots()[0]
            .clone()
            .unwrap_or_else(|| vortex_panic!("TestSpine child"))
            .downcast::<Primitive>();
        let validity = child.validity()?;
        let ptype = child.ptype();
        let bytes = match_each_native_ptype!(ptype, |T| {
            Buffer::<T>::from_iter(child.as_slice::<T>().iter().copied().map(SpineStep::step))
                .into_byte_buffer()
        });
        Ok(ExecutionResult::done(
            PrimitiveArray::from_byte_buffer(bytes, ptype, validity).into_array(),
        ))
    }

    fn block_op(
        array: ArrayView<'_, Self>,
        _slot_idx: usize,
        _child_dtype: &DType,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Box<dyn BlockOp>>> {
        Ok(Some(match_each_native_ptype!(
            array.dtype().as_ptype(),
            |T| { Box::new(SpineOp::<T>(PhantomData)) as Box<dyn BlockOp> }
        )))
    }
}

struct SpineOp<T>(PhantomData<T>);

impl<T: SpineStep> BlockOp for SpineOp<T> {
    fn process<'a>(&mut self, block: Block<'a>) -> VortexResult<Written<'a>> {
        Ok(block.map(T::step))
    }
}

trait SpineStep: NativePType {
    fn step(self) -> Self;
}

macro_rules! impl_int_spine_step {
    ($($T:ty),+ $(,)?) => {
        $(
            impl SpineStep for $T {
                #[inline]
                fn step(self) -> Self {
                    self.wrapping_add(1)
                }
            }
        )+
    };
}

impl_int_spine_step!(u8, u16, u32, u64, i8, i16, i32, i64);

macro_rules! impl_float_spine_step {
    ($($T:ty),+ $(,)?) => {
        $(
            impl SpineStep for $T {
                #[inline]
                fn step(self) -> Self {
                    if self == <$T>::from(0u8) {
                        <$T>::from(1u8)
                    } else {
                        -self
                    }
                }
            }
        )+
    };
}

impl_float_spine_step!(f16, f32, f64);
