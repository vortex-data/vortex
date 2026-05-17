// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! `ProgressionArray`: `start, start+step, ...` in O(1) memory.
//!
//! Used by [`crate::smj`] as the right index column for SMJ Cartesian
//! products. Minimal `VTable` (scalar_at, validity, execute-as-Primitive);
//! all other ops fall through to canonicalisation.

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use vortex_array::Array;
use vortex_array::ArrayEq;
use vortex_array::ArrayHash;
use vortex_array::ArrayId;
use vortex_array::ArrayParts;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ExecutionResult;
use vortex_array::IntoArray;
use vortex_array::OperationsVTable;
use vortex_array::Precision;
use vortex_array::VTable;
use vortex_array::ValidityVTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::scalar::Scalar;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_buffer::BufferMut;

pub type ProgressionArray = Array<Progression>;

#[derive(Clone, Debug)]
pub struct ProgressionData {
    pub(super) start: u64,
    pub(super) step: u64,
}

impl Display for ProgressionData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "start={}, step={}", self.start, self.step)
    }
}

impl ArrayHash for ProgressionData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _: Precision) {
        self.start.hash(state);
        self.step.hash(state);
    }
}

impl ArrayEq for ProgressionData {
    fn array_eq(&self, other: &Self, _: Precision) -> bool {
        self.start == other.start && self.step == other.step
    }
}

#[derive(Clone, Debug)]
pub struct Progression;

impl Progression {
    /// Construct `start, start+step, ..., start+(len-1)*step`.
    ///
    /// The constructor lives on the marker type rather than
    /// `Array<Progression>` because Rust's orphan rule forbids
    /// defining inherent methods on `Array<V>` from outside
    /// `vortex-array`.
    pub fn new(start: u64, step: u64, len: usize) -> ProgressionArray {
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(
                Progression,
                DType::Primitive(PType::U64, Nullability::NonNullable),
                len,
                ProgressionData { start, step },
            ))
        }
    }
}

impl VTable for Progression {
    type TypedArrayData = ProgressionData;
    type OperationsVTable = Self;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.prototype.progression");
        *ID
    }

    fn validate(
        &self,
        _data: &ProgressionData,
        dtype: &DType,
        _len: usize,
        _slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            *dtype == DType::Primitive(PType::U64, Nullability::NonNullable),
            "ProgressionArray dtype must be U64 NonNullable"
        );
        Ok(())
    }

    fn nbuffers(_: ArrayView<'_, Self>) -> usize {
        0
    }
    fn buffer(_: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("ProgressionArray buffer {idx} out of bounds")
    }
    fn buffer_name(_: ArrayView<'_, Self>, _: usize) -> Option<String> {
        None
    }
    fn slot_name(_: ArrayView<'_, Self>, idx: usize) -> String {
        vortex_panic!("ProgressionArray slot {idx} out of bounds")
    }

    fn serialize(_: ArrayView<'_, Self>, _: &VortexSession) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }
    fn deserialize(
        &self,
        _: &DType,
        _: usize,
        _: &[u8],
        _: &[BufferHandle],
        _: &dyn ArrayChildren,
        _: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("ProgressionArray is research-only and not deserialisable");
    }

    fn execute(array: Array<Self>, _: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        let len = array.len();
        let (start, step) = (array.start, array.step);
        let mut buf = BufferMut::<u64>::with_capacity(len);
        let mut v = start;
        for _ in 0..len {
            buf.push(v);
            v = v.wrapping_add(step);
        }
        Ok(ExecutionResult::done(
            PrimitiveArray::new(buf.freeze(), Validity::NonNullable).into_array(),
        ))
    }
}

impl OperationsVTable<Progression> for Progression {
    fn scalar_at(
        array: ArrayView<'_, Progression>,
        index: usize,
        _: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let v = array.start.wrapping_add((index as u64).wrapping_mul(array.step));
        Ok(Scalar::from(v))
    }
}

impl ValidityVTable<Progression> for Progression {
    fn validity(_: ArrayView<'_, Progression>) -> VortexResult<Validity> {
        Ok(Validity::NonNullable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use vortex_array::LEGACY_SESSION;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Primitive;

    #[test]
    fn scalar_at_returns_start_plus_step_times_index() -> VortexResult<()> {
        let arr = Progression::new(7, 5, 4);
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        for i in 0..4 {
            let s = Progression::scalar_at(arr.as_view(), i, &mut ctx)?;
            assert_eq!(u64::try_from(&s)?, 7 + 5 * (i as u64));
        }
        Ok(())
    }

    #[test]
    fn canonicalises_to_primitive() -> VortexResult<()> {
        let arr = Progression::new(100, 2, 6).into_array();
        let mut ctx = LEGACY_SESSION.create_execution_ctx();
        let out: ArrayRef = arr.execute::<ArrayRef>(&mut ctx)?;
        let prim = out.as_typed::<Primitive>().expect("primitive");
        assert_eq!(prim.as_slice::<u64>(), &[100u64, 102, 104, 106, 108, 110]);
        Ok(())
    }

    /// O(1) memory: small and large progressions have the same in-memory footprint.
    #[test]
    fn memory_is_o1() {
        let small = Progression::new(0, 1, 8).into_array();
        let large = Progression::new(0, 1, 1_000_000).into_array();
        assert_eq!(small.nbytes(), large.nbytes());
    }
}
