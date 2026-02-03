// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_buffer::Buffer;
use vortex_dtype::NativePType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_scalar::Scalar;
use vortex_session::VortexSession;

use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::arrays::DecimalArray;
use crate::arrays::DecimalVTable;
use crate::arrays::ExtensionArray;
use crate::arrays::ExtensionVTable;
use crate::arrays::FixedSizeListArray;
use crate::arrays::FixedSizeListVTable;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewVTable;
use crate::arrays::NullArray;
use crate::arrays::NullVTable;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::arrays::StructArray;
use crate::arrays::StructVTable;
use crate::arrays::VarBinViewArray;
use crate::arrays::VarBinViewVTable;

/// Marker trait for types that can be executed.
///
/// If the `ArrayRef` cannot inhabit `Self` this will panic.
pub trait Executable: Sized {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self>;
}

impl dyn Array + '_ {
    /// Execute this array to produce an instance of `E`.
    pub fn execute<E: Executable>(self: Arc<Self>, ctx: &mut ExecutionCtx) -> VortexResult<E> {
        E::execute(self, ctx)
    }
}

/// Execution context for batch CPU compute.
pub struct ExecutionCtx {
    session: VortexSession,
}

impl ExecutionCtx {
    /// Create a new execution context with the given session.
    pub fn new(session: VortexSession) -> Self {
        Self { session }
    }

    /// Get the session associated with this execution context.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }
}

/// An enum capturing either a columnar array or a constant scalar value.
pub enum Columnar {
    /// A columnar array.
    Array(Canonical),
    /// A constant scalar value.
    Scalar(Scalar),
}

impl Executable for Columnar {
    /// This is the main execution loop for Vortex in-memory arrays.
    ///
    /// This will iteratively reduce and execute arrays until we reach either a constant array or
    /// an array in canonical form.
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        loop {
            // 0. Check for termination conditions
            if let Some(constant) = array.as_opt::<ConstantVTable>() {
                return Ok(Columnar::Scalar(constant.into_parts()));
            }
            if array.is_canonical() {
                todo!("Extract canonical array without re-executing");
            }

            // 1. reduce / reduce_parent (metadata-only rewrites)

            // 2. execute_parent (child-driven optimized execution)

            // 3. step ONE child (the first non-canonical, non-constant one)
            //  → restart from (1) if progress was made

            // 4. if no progress anywhere → call canonicalize, done
        }
    }
}

/// Recursively execute the array to canonical form.
/// This will replace the recursive usage of `to_canonical()`.
/// An `ExecutionCtx` is will be used to limit access to buffers.
impl Executable for Canonical {
    fn execute(mut array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        // Try and dispatch to a child that can optimize execution.
        // TODO(ngates): maybe put a limit on reduce_parent iterations
        'outer: loop {
            for (child_idx, child) in array.children().iter().enumerate() {
                if let Some(result) = child
                    .vtable()
                    .execute_parent(child, &array, child_idx, ctx)?
                {
                    tracing::debug!(
                        "Executed array {} via child {} optimization.",
                        array.encoding_id(),
                        child.encoding_id()
                    );
                    array = result;
                    continue 'outer;
                }
            }
            break;
        }

        // Otherwise fall back to the default execution.
        array.vtable().canonicalize(&array, ctx)
    }
}

/// Execute the array and return a [`CanonicalOutput`].
///
/// This may short-circuit for constant arrays, returning [`CanonicalOutput::Constant`]
/// instead of fully materializing the array.
impl Executable for CanonicalOutput {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        // Attempt to short-circuit constant arrays.
        if let Some(constant) = array.as_opt::<ConstantVTable>() {
            return Ok(CanonicalOutput::Constant(ConstantArray::new(
                constant.scalar().clone(),
                constant.len(),
            )));
        }

        tracing::debug!("Executing array {}:\n{}", array, array.display_tree());
        Ok(CanonicalOutput::Array(array.execute(ctx)?))
    }
}

/// Execute a primitive array into a buffer of native values.
///
/// This will error if the array is not all-valid.
impl<T: NativePType> Executable for Buffer<T> {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let array = array.execute::<PrimitiveArray>(ctx)?;
        vortex_ensure!(
            array.all_valid()?,
            "Cannot execute to native buffer: array is not all-valid."
        );
        Ok(array.into_buffer())
    }
}

/// Execute the array to canonical form and unwrap as a [`PrimitiveArray`].
///
/// This will panic if the array's dtype is not primitive.
impl Executable for PrimitiveArray {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.try_into::<PrimitiveVTable>() {
            Ok(primitive) => Ok(primitive),
            Err(array) => Ok(array.execute::<Canonical>(ctx)?.into_primitive()),
        }
    }
}

/// Execute the array to canonical form and unwrap as a [`BoolArray`].
///
/// This will panic if the array's dtype is not bool.
impl Executable for BoolArray {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.try_into::<BoolVTable>() {
            Ok(bool_array) => Ok(bool_array),
            Err(array) => Ok(array.execute::<Canonical>(ctx)?.into_bool()),
        }
    }
}

/// Execute the array to canonical form and unwrap as a [`NullArray`].
///
/// This will panic if the array's dtype is not null.
impl Executable for NullArray {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.try_into::<NullVTable>() {
            Ok(null_array) => Ok(null_array),
            Err(array) => Ok(array.execute::<Canonical>(ctx)?.into_null()),
        }
    }
}

/// Execute the array to canonical form and unwrap as a [`VarBinViewArray`].
///
/// This will panic if the array's dtype is not utf8 or binary.
impl Executable for VarBinViewArray {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.try_into::<VarBinViewVTable>() {
            Ok(varbinview) => Ok(varbinview),
            Err(array) => Ok(array.execute::<Canonical>(ctx)?.into_varbinview()),
        }
    }
}

/// Execute the array to canonical form and unwrap as an [`ExtensionArray`].
///
/// This will panic if the array's dtype is not an extension type.
impl Executable for ExtensionArray {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.try_into::<ExtensionVTable>() {
            Ok(ext_array) => Ok(ext_array),
            Err(array) => Ok(array.execute::<Canonical>(ctx)?.into_extension()),
        }
    }
}

/// Execute the array to canonical form and unwrap as a [`DecimalArray`].
///
/// This will panic if the array's dtype is not decimal.
impl Executable for DecimalArray {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.try_into::<DecimalVTable>() {
            Ok(decimal) => Ok(decimal),
            Err(array) => Ok(array.execute::<Canonical>(ctx)?.into_decimal()),
        }
    }
}

/// Execute the array to canonical form and unwrap as a [`ListViewArray`].
///
/// This will panic if the array's dtype is not list.
impl Executable for ListViewArray {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.try_into::<ListViewVTable>() {
            Ok(list) => Ok(list),
            Err(array) => Ok(array.execute::<Canonical>(ctx)?.into_listview()),
        }
    }
}

/// Execute the array to canonical form and unwrap as a [`FixedSizeListArray`].
///
/// This will panic if the array's dtype is not fixed size list.
impl Executable for FixedSizeListArray {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.try_into::<FixedSizeListVTable>() {
            Ok(fsl) => Ok(fsl),
            Err(array) => Ok(array.execute::<Canonical>(ctx)?.into_fixed_size_list()),
        }
    }
}

/// Execute the array to canonical form and unwrap as a [`StructArray`].
///
/// This will panic if the array's dtype is not struct.
impl Executable for StructArray {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        match array.try_into::<StructVTable>() {
            Ok(struct_array) => Ok(struct_array),
            Err(array) => Ok(array.execute::<Canonical>(ctx)?.into_struct()),
        }
    }
}

/// Extension trait for creating an execution context from a session.
pub trait VortexSessionExecute {
    /// Create a new execution context from this session.
    fn create_execution_ctx(&self) -> ExecutionCtx;
}

impl VortexSessionExecute for VortexSession {
    fn create_execution_ctx(&self) -> ExecutionCtx {
        ExecutionCtx::new(self.clone())
    }
}
