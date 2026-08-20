// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Shared execution for native geometry scalar functions.
//!
//! [`dispatch_unary`] and [`dispatch_binary`] handle constant/column operands and strict null
//! propagation without prescribing how a kernel represents geometries or builds its output.
//! Native columnar kernels such as `ST_MakeLine` use these dispatchers directly.
//!
//! [`execute_unary_geo_types`] and [`execute_binary_geo_types`] are convenience adapters for
//! row-oriented algorithms from the `geo` ecosystem. They decode valid inputs into
//! `geo_types::Geometry`; the final output is still a Vortex [`ArrayRef`], such as an `f64` or
//! boolean array.

mod binary;
mod geo_types;
mod unary;

pub(crate) use binary::dispatch_binary;
pub(crate) use binary::execute_binary_geo_types;
pub(crate) use unary::dispatch_unary;
pub(crate) use unary::execute_unary_geo_types;
use vortex_array::ArrayRef;
use vortex_array::dtype::Nullability;
use vortex_array::scalar::Scalar;
use vortex_mask::Mask;

/// A non-null operand presented to a geometry kernel.
pub(crate) enum Operand {
    /// One scalar value repeated for every row.
    Constant(Scalar),
    /// A column with one value per row.
    Column(ArrayRef),
}

/// Shared batch state presented to a null-propagating geometry kernel with `N` operands.
///
/// Binary kernels use the default materialized [`Mask`]. Unary columnar kernels can instead
/// retain a lazy [`vortex_array::validity::Validity`] until they need row-wise access.
pub(crate) struct Execution<const N: usize, V = Mask> {
    /// Constant/column shape of each operand.
    pub(crate) operands: [Operand; N],
    /// Validity state required by the kernel.
    pub(crate) valid: V,
    /// Number of output rows.
    pub(crate) len: usize,
    /// Output nullability from the scalar function's return dtype.
    pub(crate) nullability: Nullability,
}
