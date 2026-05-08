// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use kernel::PARENT_KERNELS;
use smallvec::smallvec;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;
use vortex_session::VortexSession;
use vortex_session::registry::CachedId;

use crate::ArrayRef;
use crate::EmptyArrayData;
use crate::ExecutionCtx;
use crate::ExecutionResult;
use crate::array::Array;
use crate::array::ArrayId;
use crate::array::ArrayParts;
use crate::array::ArrayView;
use crate::array::VTable;
use crate::array::ValidityVTableFromChild;
use crate::arrays::extension::array::SLOT_NAMES;
use crate::arrays::extension::array::STORAGE_SLOT;
use crate::arrays::extension::compute::rules::PARENT_RULES;
use crate::arrays::extension::compute::rules::RULES;
use crate::buffer::BufferHandle;
use crate::dtype::DType;
use crate::serde::ArrayChildren;

mod kernel;
mod operations;
mod validity;

/// An extension array that wraps another array with additional type information.
///
/// **⚠️ Unstable API**: This is an experimental feature that may change significantly
/// in future versions. The extension type system is still evolving.
///
/// Unlike Apache Arrow's extension arrays, Vortex extension arrays provide a more flexible
/// mechanism for adding semantic meaning to existing array types without requiring
/// changes to the core type system.
///
/// ## Design Philosophy
///
/// Extension arrays serve as a type-safe wrapper that:
/// - Preserves the underlying storage format and operations
/// - Adds semantic type information via `ExtDType`
/// - Enables custom serialization and deserialization logic
/// - Allows domain-specific interpretations of generic data
///
/// ## Storage and Type Relationship
///
/// The extension array maintains a strict contract:
/// - **Storage array**: Contains the actual data in a standard Vortex encoding
/// - **Extension type**: Defines how to interpret the storage data semantically
/// - **Type safety**: The storage array's dtype must match the extension type's storage dtype
///
/// ## Use Cases
///
/// Extension arrays are ideal for:
/// - **Custom numeric types**: Units of measurement, currencies
/// - **Temporal types**: Custom date/time formats, time zones, calendars
/// - **Domain-specific types**: UUIDs, IP addresses, geographic coordinates
/// - **Encoded types**: Base64 strings, compressed data, encrypted values
///
/// ## Validity and Operations
///
/// Extension arrays delegate validity and most operations to their storage array:
/// - Validity is inherited from the underlying storage
/// - Slicing preserves the extension type
/// - Scalar access wraps storage scalars with extension metadata
#[derive(Clone, Debug)]
pub struct Extension;

/// A [`Extension`]-encoded Vortex array.
pub type ExtensionArray = Array<Extension>;

impl VTable for Extension {
    type TypedArrayData = EmptyArrayData;

    type OperationsVTable = Self;
    type ValidityVTable = ValidityVTableFromChild;

    fn id(&self) -> ArrayId {
        static ID: CachedId = CachedId::new("vortex.ext");
        *ID
    }

    fn validate(
        &self,
        _data: &EmptyArrayData,
        dtype: &DType,
        len: usize,
        slots: &[Option<ArrayRef>],
    ) -> VortexResult<()> {
        let storage = slots[STORAGE_SLOT]
            .as_ref()
            .vortex_expect("ExtensionArray storage slot");
        vortex_ensure_eq!(
            storage.len(),
            len,
            "ExtensionArray length {} does not match outer length {len}",
            storage.len(),
        );

        let ext_dtype = dtype
            .as_extension_opt()
            .ok_or_else(|| vortex_err!("not an extension dtype"))?;

        let actual_dtype = DType::Extension(ext_dtype.clone());
        vortex_ensure_eq!(
            &actual_dtype,
            dtype,
            "ExtensionArray dtype {actual_dtype} does not match outer dtype {dtype}",
        );

        Ok(())
    }

    fn nbuffers(_array: ArrayView<'_, Self>) -> usize {
        0
    }

    fn buffer(_array: ArrayView<'_, Self>, idx: usize) -> BufferHandle {
        vortex_panic!("ExtensionArray buffer index {idx} out of bounds")
    }

    fn buffer_name(_array: ArrayView<'_, Self>, _idx: usize) -> Option<String> {
        None
    }

    fn serialize(
        _array: ArrayView<'_, Self>,
        _session: &VortexSession,
    ) -> VortexResult<Option<Vec<u8>>> {
        Ok(Some(vec![]))
    }

    fn deserialize(
        &self,
        dtype: &DType,
        len: usize,
        metadata: &[u8],

        _buffers: &[BufferHandle],
        children: &dyn ArrayChildren,
        _session: &VortexSession,
    ) -> VortexResult<ArrayParts<Self>> {
        if !metadata.is_empty() {
            vortex_bail!(
                "ExtensionArray expects empty metadata, got {} bytes",
                metadata.len()
            );
        }
        let DType::Extension(ext_dtype) = dtype else {
            vortex_bail!("Not an extension DType");
        };
        if children.len() != 1 {
            vortex_bail!("Expected 1 child, got {}", children.len());
        }
        let storage = children.get(0, ext_dtype.storage_dtype(), len)?;
        Ok(
            ArrayParts::new(self.clone(), dtype.clone(), len, EmptyArrayData)
                .with_slots(smallvec![Some(storage)]),
        )
    }

    fn slot_name(_array: ArrayView<'_, Self>, idx: usize) -> String {
        SLOT_NAMES[idx].to_string()
    }

    fn execute(array: Array<Self>, _ctx: &mut ExecutionCtx) -> VortexResult<ExecutionResult> {
        Ok(ExecutionResult::done(array))
    }

    fn execute_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_KERNELS.execute(array, parent, child_idx, ctx)
    }

    fn reduce(array: ArrayView<'_, Self>) -> VortexResult<Option<ArrayRef>> {
        RULES.evaluate(array)
    }

    fn reduce_parent(
        array: ArrayView<'_, Self>,
        parent: &ArrayRef,
        child_idx: usize,
    ) -> VortexResult<Option<ArrayRef>> {
        PARENT_RULES.evaluate(array, parent, child_idx)
    }
}
