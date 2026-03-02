// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env::VarError;
use std::sync::LazyLock;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_panic;

use crate::AnyCanonical;
use crate::Array;
use crate::ArrayRef;
use crate::Canonical;
use crate::CanonicalView;
use crate::Executable;
use crate::ExecutionCtx;
use crate::ExecutionStep;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::dtype::DType;
use crate::matcher::Matcher;
use crate::optimizer::ArrayOptimizer;
use crate::scalar::Scalar;

/// Represents a columnnar array of data, either in canonical form or as a constant array.
///
/// Since the [`Canonical`] enum has one variant per logical data type, it is inefficient for
/// representing constant arrays. The [`Columnar`] enum allows holding an array of data in either
/// canonical or constant form enabling efficient handling of constants during execution.
pub enum Columnar {
    /// A columnar array in canonical form.
    Canonical(Canonical),
    /// A columnar array in constant form.
    Constant(ConstantArray),
}

impl Columnar {
    /// Creates a new columnar array from a scalar.
    pub fn constant<S: Into<Scalar>>(scalar: S, len: usize) -> Self {
        Columnar::Constant(ConstantArray::new(scalar.into(), len))
    }

    /// Returns the length of this columnar array.
    pub fn len(&self) -> usize {
        match self {
            Columnar::Canonical(canonical) => canonical.len(),
            Columnar::Constant(constant) => constant.len(),
        }
    }

    /// Returns true if this columnar array has length zero.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the data type of this columnar array.
    pub fn dtype(&self) -> &DType {
        match self {
            Columnar::Canonical(canonical) => canonical.dtype(),
            Columnar::Constant(constant) => constant.dtype(),
        }
    }
}

impl IntoArray for Columnar {
    fn into_array(self) -> ArrayRef {
        match self {
            Columnar::Canonical(canonical) => canonical.into_array(),
            Columnar::Constant(constant) => constant.into_array(),
        }
    }
}

/// Executing into a [`Columnar`] is implemented using an iterative scheduler with an explicit
/// work stack.
///
/// The scheduler repeatedly:
/// 1. Checks if the current array is columnar (constant or canonical) — if so, pops the stack.
/// 2. Runs reduce/reduce_parent rules to fixpoint.
/// 3. Tries execute_parent on each child.
/// 4. Calls `execute` which returns an [`ExecutionStep`].
///
/// For safety, we will error when the number of execution iterations reaches 128. We may want this
/// to be configurable in the future in case of highly complex array trees, but in practice we
/// don't expect to ever reach this limit.
impl Executable for Columnar {
    fn execute(root: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        static MAX_ITERATIONS: LazyLock<usize> =
            LazyLock::new(|| match std::env::var("VORTEX_MAX_ITERATIONS") {
                Ok(val) => val.parse::<usize>().unwrap_or_else(|e| {
                    vortex_panic!("VORTEX_MAX_ITERATIONS is not a valid usize: {e}")
                }),
                Err(VarError::NotPresent) => 128,
                Err(VarError::NotUnicode(_)) => {
                    vortex_panic!("VORTEX_MAX_ITERATIONS is not a valid unicode string")
                }
            });

        let mut current = root.optimize()?;
        let mut stack: Vec<(ArrayRef, usize)> = Vec::new();

        for _ in 0..*MAX_ITERATIONS {
            // Check for columnar termination (constant or canonical)
            if let Some(columnar) = try_as_columnar(&current) {
                match stack.pop() {
                    None => {
                        // Stack empty — we're done
                        ctx.log(format_args!("-> columnar {}", current));
                        return Ok(columnar);
                    }
                    Some((parent, child_idx)) => {
                        // Replace the child in the parent and continue
                        current = parent.with_child(child_idx, current)?;
                        current = current.optimize()?;
                        continue;
                    }
                }
            }

            // Try execute_parent (child-driven optimized execution)
            if let Some(rewritten) = try_execute_parent(&current, ctx)? {
                ctx.log(format_args!(
                    "execute_parent rewrote {} -> {}",
                    current, rewritten
                ));
                current = rewritten.optimize()?;
                continue;
            }

            // Execute the array itself
            match current.vtable().execute(&current, ctx)? {
                ExecutionStep::ExecuteChild(i) => {
                    let child = current
                        .nth_child(i)
                        .vortex_expect("ExecuteChild index in bounds");
                    ctx.log(format_args!(
                        "ExecuteChild({i}): pushing {}, focusing on {}",
                        current, child
                    ));
                    stack.push((current, i));
                    current = child.optimize()?;
                }
                ExecutionStep::ColumnarizeChild(i) => {
                    let child = current
                        .nth_child(i)
                        .vortex_expect("ColumnarizeChild index in bounds");
                    ctx.log(format_args!(
                        "ColumnarizeChild({i}): pushing {}, focusing on {}",
                        current, child
                    ));
                    stack.push((current, i));
                    // No cross-step optimization for ColumnarizeChild
                    current = child;
                }
                ExecutionStep::Done(result) => {
                    ctx.log(format_args!("Done: {} -> {}", current, result));
                    current = result;
                }
            }
        }

        vortex_bail!("Exceeded maximum execution iterations while executing to Columnar")
    }
}

/// Try to interpret an array as columnar (constant or canonical).
fn try_as_columnar(array: &ArrayRef) -> Option<Columnar> {
    if let Some(constant) = array.as_opt::<ConstantVTable>() {
        Some(Columnar::Constant(constant.clone()))
    } else {
        array
            .as_opt::<AnyCanonical>()
            .map(|c| Columnar::Canonical(c.into()))
    }
}

/// Try execute_parent on each child of the array.
fn try_execute_parent(array: &ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Option<ArrayRef>> {
    for child_idx in 0..array.nchildren() {
        let child = array
            .nth_child(child_idx)
            .vortex_expect("checked nchildren");
        if let Some(result) = child
            .vtable()
            .execute_parent(&child, array, child_idx, ctx)?
        {
            result.statistics().inherit_from(array.statistics());
            return Ok(Some(result));
        }
    }
    Ok(None)
}

pub enum ColumnarView<'a> {
    Canonical(CanonicalView<'a>),
    Constant(&'a ConstantArray),
}

impl<'a> AsRef<dyn Array> for ColumnarView<'a> {
    fn as_ref(&self) -> &dyn Array {
        match self {
            ColumnarView::Canonical(canonical) => canonical.as_ref(),
            ColumnarView::Constant(constant) => constant.as_ref(),
        }
    }
}

pub struct AnyColumnar;
impl Matcher for AnyColumnar {
    type Match<'a> = ColumnarView<'a>;

    fn try_match<'a>(array: &'a dyn Array) -> Option<Self::Match<'a>> {
        if let Some(constant) = array.as_opt::<ConstantVTable>() {
            Some(ColumnarView::Constant(constant))
        } else {
            array.as_opt::<AnyCanonical>().map(ColumnarView::Canonical)
        }
    }
}
