// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::AnyCanonical;
use crate::ArrayRef;
use crate::Canonical;
use crate::CanonicalView;
use crate::Executable;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::ArrayView;
use crate::arrays::Constant;
use crate::arrays::ConstantArray;
use crate::dtype::DType;
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

/// Execute into [`Columnar`] by running `execute_until` with the [`AnyColumnar`] matcher.
impl Executable for Columnar {
    fn execute(array: ArrayRef, ctx: &mut ExecutionCtx) -> VortexResult<Self> {
        let result = array.execute_until::<AnyColumnar>(ctx)?;
        if let Some(constant) = result.as_opt::<Constant>() {
            Ok(Columnar::Constant(constant.into_owned()))
        } else {
            Ok(Columnar::Canonical(
                result
                    .as_opt::<AnyCanonical>()
                    .map(Canonical::from)
                    .vortex_expect("execute_until::<AnyColumnar> must return a columnar array"),
            ))
        }
    }
}

pub enum ColumnarView<'a> {
    Canonical(CanonicalView<'a>),
    Constant(ArrayView<'a, Constant>),
}

pub struct AnyColumnar;
impl Matcher for AnyColumnar {
    type Match<'a> = ColumnarView<'a>;

    fn try_match<'a>(array: &'a ArrayRef) -> Option<Self::Match<'a>> {
        if let Some(constant) = array.as_opt::<Constant>() {
            Some(ColumnarView::Constant(constant))
        } else {
            array.as_opt::<AnyCanonical>().map(ColumnarView::Canonical)
        }
    }
}
