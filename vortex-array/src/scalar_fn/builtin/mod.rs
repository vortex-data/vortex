// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The scalar functions that vortex-array itself is built on.
//!
//! These are the functions that the core crate's own arrays, validity and expression machinery
//! construct: literals and struct projection, selection and packing for expressions, `is_null`,
//! `is_not_null`, `not` and the boolean operators for validity reduction, and `cast`, `mask`,
//! `fill_null` and the comparison and arithmetic operators, which canonical arrays apply lazily to
//! their own children (offsets, indices, codes and validity). [`ArrayBuiltins`] exposes them as
//! methods.
//!
//! Every other scalar function lives in the `vortex-functions` crate, which registers them, and
//! their kernels for vortex-array's encodings, with a session.
//!
//! [`ArrayBuiltins`]: crate::builtins::ArrayBuiltins

pub mod binary;
pub mod cast;
mod field;
pub mod fill_null;
pub mod get_item;
pub mod is_not_null;
pub mod is_null;
pub mod literal;
pub mod mask;
pub mod not;
pub mod operators;
pub mod pack;
pub mod select;
