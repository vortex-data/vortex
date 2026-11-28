// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Aggregate functions over geometry columns.

mod aabb;

pub use aabb::AabbPartial;
pub use aabb::GeometryAabb;
pub(crate) use aabb::aabb_of_array;
