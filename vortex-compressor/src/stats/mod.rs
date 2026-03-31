// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression statistics types and caching.

mod cache;
mod float;
mod integer;
mod options;
mod string;

pub use cache::ArrayAndStats;
pub use float::DistinctInfo as FloatDistinctInfo;
pub use float::ErasedStats as FloatErasedStats;
pub use float::FloatStats;
pub use float::TypedStats as FloatTypedStats;
pub use integer::DistinctInfo as IntegerDistinctInfo;
pub use integer::ErasedStats as IntegerErasedStats;
pub use integer::IntegerStats;
pub use integer::TypedStats as IntegerTypedStats;
pub use options::GenerateStatsOptions;
pub use string::StringStats;
