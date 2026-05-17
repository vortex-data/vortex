// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Per-encoding fast-path implementations of [`RowSizeKernel`] and [`RowEncodeKernel`] for
//! encodings defined in `vortex-array`.
//!
//! Each impl in this module lives here (rather than under the corresponding encoding's
//! `compute` module in `vortex-array`) so the orphan rule is satisfied: the trait is
//! defined in `vortex-row` and the impl is also in `vortex-row`, while the array type
//! (`Constant`, `Dict`) remains in `vortex-array`.
//!
//! [`RowSizeKernel`]: crate::size::RowSizeKernel
//! [`RowEncodeKernel`]: crate::encode::RowEncodeKernel

mod constant;
mod dict;
