// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;
use std::fmt::Display;
use std::ops::Deref;
use std::ops::DerefMut;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use vortex_buffer::Buffer;
use vortex_dtype::DType;
use vortex_dtype::NativePType;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_scalar::Scalar;
use vortex_session::VortexSession;

use crate::AnyCanonical;
use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::IntoArray;
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
use crate::arrays::constant_canonicalize;

/// Marker trait for types that can be executed.
///
/// If the `ArrayRef` cannot inhabit `Self` this will panic.
pub trait Executable: Sized {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self>;
}

fn short_type_name<T>() -> &'static str {
    let full = std::any::type_name::<T>();
    full.rsplit("::").next().unwrap_or(full)
}

impl dyn Array + '_ {
    /// Execute this array to produce an instance of `E`.
    pub fn execute<E: Executable>(self: Arc<Self>, ctx: &mut ExecutionCtx) -> VortexResult<E> {
        ctx.log_entry(
            &self,
            format_args!("execute<{}> {}", short_type_name::<E>(), self),
        );
        let mut scope = ctx.child_scope();
        E::execute(self, &mut scope)
    }

    /// Execute this array, labeling the step with a child name for tracing.
    ///
    /// Use this in `canonicalize` implementations to annotate which child role
    /// (e.g. "ends", "values", "codes") is being executed.
    pub fn execute_as<E: Executable>(
        self: Arc<Self>,
        name: &str,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<E> {
        ctx.log_entry(
            &self,
            format_args!("{}: execute<{}> {}", name, short_type_name::<E>(), self),
        );
        let mut scope = ctx.child_scope();
        E::execute(self, &mut scope)
    }
}

/// Execution context for batch CPU compute.
///
/// Accumulates a trace of execution steps. Individual steps are logged at TRACE level for
/// real-time following, and the full trace is dumped at DEBUG level when the context is dropped.
pub struct ExecutionCtx {
    id: usize,
    session: VortexSession,
    depth: usize,
    ops: Vec<String>,
}

impl ExecutionCtx {
    /// Create a new execution context with the given session.
    pub fn new(session: VortexSession) -> Self {
        static EXEC_CTX_ID: AtomicUsize = AtomicUsize::new(0);
        let id = EXEC_CTX_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Self {
            id,
            session,
            depth: 0,
            ops: Vec::new(),
        }
    }

    /// Get the session associated with this execution context.
    pub fn session(&self) -> &VortexSession {
        &self.session
    }

    /// Log an execution step at the current depth.
    ///
    /// Steps are accumulated and dumped as a single trace on Drop at DEBUG level.
    /// Individual steps are also logged at TRACE level for real-time following.
    pub fn log(&mut self, msg: fmt::Arguments<'_>) {
        if tracing::enabled!(tracing::Level::DEBUG) {
            let indent = "  ".repeat(self.depth);
            let formatted = format!("{indent}{msg}");
            tracing::trace!("exec[{}]: {formatted}", self.id);
            self.ops.push(formatted);
        }
    }

    /// Log an execution entry point. On the first call into this context, the full
    /// `display_tree` of the array is included so the starting state is visible.
    fn log_entry(&mut self, array: &dyn Array, msg: fmt::Arguments<'_>) {
        if tracing::enabled!(tracing::Level::DEBUG) {
            if self.ops.is_empty() {
                self.log(format_args!("{msg}\n{}", array.display_tree()));
            } else {
                self.log(msg);
            }
        }
    }

    /// Create a child scope at an incremented depth. The depth is automatically
    /// decremented when the returned [`ExecutionScope`] is dropped.
    ///
    /// The scope derefs to `&mut ExecutionCtx` so it can be used wherever
    /// `&mut ExecutionCtx` is expected.
    pub fn child_scope(&mut self) -> ExecutionScope<'_> {
        self.depth += 1;
        ExecutionScope(self)
    }
}

/// RAII guard that decrements the [`ExecutionCtx`] depth when dropped.
///
/// Created via [`ExecutionCtx::child_scope`]. Derefs to `ExecutionCtx` so it can
/// be passed transparently to any function that takes `&mut ExecutionCtx`.
pub struct ExecutionScope<'a>(&'a mut ExecutionCtx);

impl Deref for ExecutionScope<'_> {
    type Target = ExecutionCtx;

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl DerefMut for ExecutionScope<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}

impl Drop for ExecutionScope<'_> {
    fn drop(&mut self) {
        self.0.depth -= 1;
    }
}

impl Display for ExecutionCtx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "exec[{}]", self.id)
    }
}

impl Drop for ExecutionCtx {
    fn drop(&mut self) {
        if !self.ops.is_empty() {
            tracing::debug!("exec[{}] trace:\n{}", self.id, self.ops.join("\n"));
        }
    }
}

/// The result of expression execution.
pub enum Columnar {
    Array(Canonical),
    Scalar(ConstantArray),
}

impl Columnar {
    pub fn constant<S: Into<Scalar>>(scalar: S, len: usize) -> Self {
        Columnar::Scalar(ConstantArray::new(scalar.into(), len))
    }

    /// Returns the length of this execution result.
    pub fn len(&self) -> usize {
        match self {
            Columnar::Array(canonical) => canonical.len(),
            Columnar::Scalar(constant) => constant.len(),
        }
    }

    /// Returns true if this execution result has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the data type of this execution result.
    pub fn dtype(&self) -> &DType {
        match self {
            Columnar::Array(canonical) => canonical.dtype(),
            Columnar::Scalar(constant) => constant.dtype(),
        }
    }
}

impl IntoArray for Columnar {
    fn into_array(self) -> ArrayRef {
        match self {
            Columnar::Array(canonical) => canonical.into_array(),
            Columnar::Scalar(constant) => constant.into_array(),
        }
    }
}

impl Executable for Columnar {
    /// This is the main execution loop for Vortex in-memory arrays.
    ///
    /// This will iteratively reduce and execute arrays until we reach either a constant array or
    /// an array in canonical form.
    fn execute(mut array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        'exec: loop {
            // 0. Check for termination conditions
            if let Some(constant) = array.as_opt::<ConstantVTable>() {
                ctx.log(format_args!("-> constant({})", constant.scalar()));
                return Ok(Columnar::Scalar(constant.clone()));
            }
            if let Some(canonical) = array.as_opt::<AnyCanonical>() {
                ctx.log(format_args!("-> canonical {}", array));
                return Ok(Columnar::Array(canonical.into()));
            }

            // 1. reduce / reduce_parent (metadata-only rewrites)
            // TODO(ngates): let's assume that arrays are reduced on construction for now.
            if let Some(reduced) = array.vtable().reduce(&array)? {
                ctx.log(format_args!("reduce: rewrote {} -> {}", array, reduced));
                array = reduced;
                continue 'exec;
            }
            for (child_idx, child) in array.children().iter().enumerate() {
                if let Some(reduced_parent) =
                    child.vtable().reduce_parent(child, &array, child_idx)?
                {
                    ctx.log(format_args!(
                        "reduce_parent: child[{}]({}) rewrote {} -> {}",
                        child_idx,
                        child.encoding_id(),
                        array,
                        reduced_parent
                    ));
                    array = reduced_parent;
                    continue 'exec;
                }
            }

            // 2. execute_parent (child-driven optimized execution)
            for (child_idx, child) in array.children().iter().enumerate() {
                if let Some(executed_parent) = child
                    .vtable()
                    .execute_parent(child, &array, child_idx, ctx)?
                {
                    ctx.log(format_args!(
                        "execute_parent: child[{}]({}) rewrote {} -> {}",
                        child_idx,
                        child.encoding_id(),
                        array,
                        executed_parent
                    ));
                    array = executed_parent;
                    continue 'exec;
                }
            }

            // 3. if no progress anywhere → call canonicalize, done
            ctx.log(format_args!("canonicalize {}", array));
            let result = {
                let mut scope = ctx.child_scope();
                array
                    .vtable()
                    .canonicalize(&array, &mut scope)
                    .map(Columnar::Array)
            };
            if let Ok(ref columnar) = result {
                match columnar {
                    Columnar::Array(c) => ctx.log(format_args!("-> {}", c.as_ref())),
                    Columnar::Scalar(s) => ctx.log(format_args!("-> constant({})", s.scalar())),
                }
            }
            return result;
        }
    }
}

/// Recursively execute the array to canonical form.
/// This will replace the recursive usage of `to_canonical()`.
/// An `ExecutionCtx` is will be used to limit access to buffers.
impl Executable for Canonical {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        if let Some(canonical) = array.as_opt::<AnyCanonical>() {
            return Ok(canonical.into());
        }

        // Avoid going via array.execute<Columnar>() to keep logs easy to read
        Ok(match Columnar::execute(array, ctx)? {
            Columnar::Array(c) => c,
            Columnar::Scalar(s) => constant_canonicalize(&s)?,
        })
    }
}

/// Execute a primitive array into a buffer of native values.
///
/// This will error if the array is not all-valid.
impl<T: NativePType> Executable for Buffer<T> {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let array = PrimitiveArray::execute(array, ctx)?;
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
            Err(array) => Ok(Canonical::execute(array, ctx)?.into_primitive()),
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
            Err(array) => Ok(Canonical::execute(array, ctx)?.into_bool()),
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
            Err(array) => Ok(Canonical::execute(array, ctx)?.into_null()),
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
            Err(array) => Ok(Canonical::execute(array, ctx)?.into_varbinview()),
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
            Err(array) => Ok(Canonical::execute(array, ctx)?.into_extension()),
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
            Err(array) => Ok(Canonical::execute(array, ctx)?.into_decimal()),
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
            Err(array) => Ok(Canonical::execute(array, ctx)?.into_listview()),
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
            Err(array) => Ok(Canonical::execute(array, ctx)?.into_fixed_size_list()),
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
            Err(array) => Ok(Canonical::execute(array, ctx)?.into_struct()),
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
