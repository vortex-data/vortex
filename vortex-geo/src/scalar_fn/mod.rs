// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Geometry scalar functions over the [`Point`](crate::extension::Point) type. Currently
//! [`GeoDistance`], the planar distance between two point columns.

pub mod distance;

pub use distance::GeoDistance;
