// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::vector::vector::Vector;
use vortex_array::stats::StatsSet;
use vortex_dtype::DType;

/// What is an array in a world with vectors? Is an Array just an in-memory layout? At that point,
/// should we just make Buffers an enum of in-memory and lazy, where all vectorized compute has the
/// ability to request additional buffers?
pub struct Array {}
