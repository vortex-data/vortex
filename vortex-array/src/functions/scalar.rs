// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;

use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_utils::debug_with::DebugWith;
use vortex_vector::Datum;
use vortex_vector::Scalar;

use crate::functions::Arity;
use crate::functions::Monotonicity;
use crate::functions::NullHandling;
use crate::functions::ScalarFnVTable;
use crate::functions::execution::ExecutionCtx;

/// An instance of a scalar function bound to some invocation options.
pub struct ScalarFn {
    vtable: ScalarFnVTable,
    options: Box<dyn Any + Send + Sync>,
}

impl ScalarFn {
    /// Create a new scalar function instance.
    ///
    /// # Safety
    ///
    /// The options must be of the correct type for the given vtable.
    pub(crate) unsafe fn new_unchecked(
        vtable: ScalarFnVTable,
        options: Box<dyn Any + Send + Sync>,
    ) -> Self {
        Self { vtable, options }
    }

    /// Get the options for this scalar function.
    pub fn options(&self) -> ScalarFnOptions<'_> {
        ScalarFnOptions {
            vtable: &self.vtable,
            options: self.options.as_ref(),
        }
    }

    /// Return the signature information for this scalar function.
    pub fn signature(&self) -> ScalarFnSignature<'_> {
        ScalarFnSignature {
            vtable: &self.vtable,
            options: self.options.as_ref(),
        }
    }

    pub fn return_dtype(&self, arg_types: &[DType]) -> VortexResult<DType> {
        self.vtable
            .as_dyn()
            .return_dtype(self.options.as_ref(), arg_types)
    }

    pub fn execute(&self, ctx: &ExecutionCtx) -> VortexResult<Datum> {
        self.vtable.as_dyn().execute(self.options.as_ref(), ctx)
    }
}

impl Clone for ScalarFn {
    fn clone(&self) -> Self {
        Self {
            vtable: self.vtable.clone(),
            options: self.vtable.as_dyn().options_clone(self.options.as_ref()),
        }
    }
}

impl Debug for ScalarFn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScalarFn")
            .field("id", &self.vtable.id())
            .field(
                "options",
                &DebugWith(|fmt| {
                    self.vtable
                        .as_dyn()
                        .options_debug(self.options.as_ref(), fmt)
                }),
            )
            .finish()
    }
}

impl Display for ScalarFn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}(", self.vtable.id())?;
        self.vtable
            .as_dyn()
            .options_display(self.options.as_ref(), f)?;
        write!(f, ")")
    }
}

impl PartialEq for ScalarFn {
    fn eq(&self, other: &Self) -> bool {
        self.vtable.id() == other.vtable.id()
            && self
                .vtable
                .as_dyn()
                .options_eq(self.options.as_ref(), other.options.as_ref())
    }
}
impl Eq for ScalarFn {}

impl Hash for ScalarFn {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.vtable.id().hash(state);
        self.vtable
            .as_dyn()
            .options_hash(self.options.as_ref(), state);
    }
}

/// Opaque reference to scalar function signature information.
pub struct ScalarFnSignature<'a> {
    pub(super) vtable: &'a ScalarFnVTable,
    pub(super) options: &'a dyn Any,
}

impl ScalarFnSignature<'_> {
    pub fn arity(&self) -> Arity {
        self.vtable.as_dyn().arity(self.options)
    }

    pub fn identity_element(&self) -> Option<Scalar> {
        self.vtable.as_dyn().identity_element(self.options)
    }

    pub fn absorbing_element(&self) -> Option<Scalar> {
        self.vtable.as_dyn().absorbing_element(self.options)
    }

    pub fn is_commutative(&self) -> bool {
        self.vtable.as_dyn().is_commutative(self.options)
    }

    pub fn is_idempotent(&self) -> bool {
        self.vtable.as_dyn().is_idempotent(self.options)
    }

    pub fn is_involution(&self) -> bool {
        self.vtable.as_dyn().is_involution(self.options)
    }

    pub fn monotonicity(&self, arg_idx: usize) -> Monotonicity {
        self.vtable.as_dyn().monotonicity(self.options, arg_idx)
    }

    pub fn null_handling(&self) -> NullHandling {
        self.vtable.as_dyn().null_handling(self.options)
    }

    pub fn arg_name(&self, arg_idx: usize) -> Option<String> {
        self.vtable.as_dyn().arg_name(self.options, arg_idx)
    }
}

/// Opaque reference to scalar function options.
pub struct ScalarFnOptions<'a> {
    pub(super) vtable: &'a ScalarFnVTable,
    pub(super) options: &'a dyn Any,
}

impl ScalarFnOptions<'_> {
    /// Get the options as a `dyn Any`.
    pub fn as_any(&self) -> &dyn Any {
        self.options
    }
}

impl Display for ScalarFnOptions<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.vtable.as_dyn().options_display(self.options, f)
    }
}

impl Debug for ScalarFnOptions<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.vtable.as_dyn().options_debug(self.options, f)
    }
}

impl ScalarFnOptions<'_> {
    /// Serializes the options into a byte vector.
    pub fn serialize(&self) -> VortexResult<Option<Vec<u8>>> {
        self.vtable.as_dyn().options_serialize(self.options)
    }
}
