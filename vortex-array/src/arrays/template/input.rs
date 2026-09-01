// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayEq;
use crate::ArrayHash;
use crate::ArrayParts;
use crate::ArrayRef;
use crate::EqMode;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayView;
use crate::array::TypedArrayRef;
use crate::array::VTable;
use crate::array::ValidityVTable;
use crate::array::with_empty_buffers;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::serde::ArrayChildren;
use crate::validity::Validity;
use crate::vtable::NotSupported;

/// An identity for one lexical template expansion.
///
/// Template scopes are process-local implementation details.  Lazy template arrays deliberately
/// have no persistent serialization representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TemplateScope(u64);

impl TemplateScope {
    pub(crate) fn fresh() -> Self {
        static NEXT_SCOPE: AtomicU64 = AtomicU64::new(0);
        Self(NEXT_SCOPE.fetch_add(1, Ordering::Relaxed))
    }
}

impl Display for TemplateScope {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Clone, Debug)]
pub struct TemplateInputData {
    scope: TemplateScope,
    slot: usize,
}

impl Display for TemplateInputData {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "scope: {}, slot: {}", self.scope, self.slot)
    }
}

impl ArrayHash for TemplateInputData {
    fn array_hash<H: Hasher>(&self, state: &mut H, _accuracy: EqMode) {
        self.scope.hash(state);
        self.slot.hash(state);
    }
}

impl ArrayEq for TemplateInputData {
    fn array_eq(&self, other: &Self, _accuracy: EqMode) -> bool {
        self.scope == other.scope && self.slot == other.slot
    }
}

/// A zero-length symbolic array input in a scoped template body.
pub type TemplateInputArray = Array<TemplateInput>;

#[derive(Clone, Debug)]
pub struct TemplateInput;

pub trait TemplateInputArrayExt: TypedArrayRef<TemplateInput> {
    fn scope(&self) -> TemplateScope {
        self.scope
    }

    fn slot(&self) -> usize {
        self.slot
    }
}
impl<T: TypedArrayRef<TemplateInput>> TemplateInputArrayExt for T {}

impl Array<TemplateInput> {
    /// Create one symbolic input.  Template inputs intentionally have no physical values.
    pub fn new(scope: TemplateScope, slot: usize, dtype: DType) -> Self {
        unsafe {
            Array::from_parts_unchecked(ArrayParts::new(
                TemplateInput,
                dtype,
                0,
                TemplateInputData { scope, slot },
            ))
        }
    }
}

impl VTable for TemplateInput {
    type TypedArrayData = TemplateInputData;
    type OperationsVTable = NotSupported;
    type ValidityVTable = Self;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.template-input");
        *ID
    }

    fn validate(
        &self,
        _data: &Self::TypedArrayData,
        _dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        vortex_ensure!(
            len == 0,
            "TemplateInputArray must have length zero, got {len}"
        );
        vortex_ensure!(
            slots.is_empty(),
            "TemplateInputArray must not have child slots"
        );
        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, _idx: usize) -> BufferHandle {
        vortex_panic!("TemplateInputArray has no buffers")
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

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        vortex_panic!("TemplateInputArray slot index {idx} out of bounds")
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(None)
    }

    fn deserialize(
        &self,
        _dtype: &DType,
        _len: usize,
        _metadata: &[u8],
        _buffers: &[BufferHandle],
        _children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        vortex_bail!("TemplateInputArray is not serializable")
    }

    fn execute(_array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        vortex_bail!(
            "TemplateInputArray cannot be executed directly; instantiate its template first"
        )
    }
}

impl ValidityVTable<TemplateInput> for TemplateInput {
    fn validity(_array: ArrayView<'_, TemplateInput>) -> VortexResult<Validity> {
        // Every template input is empty.  AllValid works for nullable empty dtypes without
        // pretending that it has material values.
        Ok(Validity::AllValid)
    }
}
