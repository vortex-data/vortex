// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::Canonical;
use crate::CanonicalView;
use crate::DynArray;
use crate::Executable;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::dtype::DType;
use crate::executor::MAX_ITERATIONS;
use crate::matcher::Matcher;
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

/// Executing into a [`Columnar`] is implemented by repeatedly executing the array until we
/// converge on either a constant or canonical.
///
/// For safety, we will error when the number of execution iterations reaches 128. We may want this
/// to be configurable in the future in case of highly complex array trees, but in practice we
/// don't expect to ever reach this limit.
impl Executable for Columnar {
    fn execute(mut array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        for _ in 0..*MAX_ITERATIONS {
            // Check for termination conditions
            if let Some(constant) = array.as_opt::<ConstantVTable>() {
                ctx.log(format_args!("-> constant({})", constant.scalar()));
                return Ok(Columnar::Constant(constant.clone()));
            }
            if let Some(canonical) = array.as_opt::<AnyCanonical>() {
                ctx.log(format_args!("-> canonical {}", array));
                return Ok(Columnar::Canonical(canonical.into()));
            }

            // Otherwise execute the array one step
            array = array.execute(ctx)?;
        }

        // If we reach here, we exceeded the maximum number of iterations, so error.
        vortex_bail!("Exceeded maximum execution iterations while executing to Columnar")
    }
}

pub enum ColumnarView<'a> {
    Canonical(CanonicalView<'a>),
    Constant(&'a ConstantArray),
}

impl<'a> AsRef<dyn DynArray> for ColumnarView<'a> {
    fn as_ref(&self) -> &dyn DynArray {
        match self {
            ColumnarView::Canonical(canonical) => canonical.as_ref(),
            ColumnarView::Constant(constant) => constant.as_ref(),
        }
    }
}

pub struct AnyColumnar;
impl Matcher for AnyColumnar {
    type Match<'a> = ColumnarView<'a>;

    fn try_match<'a>(array: &'a dyn DynArray) -> Option<Self::Match<'a>> {
        if let Some(constant) = array.as_opt::<ConstantVTable>() {
            Some(ColumnarView::Constant(constant))
        } else {
            array.as_opt::<AnyCanonical>().map(ColumnarView::Canonical)
        }
    }
}
