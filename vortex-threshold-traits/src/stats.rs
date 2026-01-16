#![allow(clippy::disallowed_types)]
//! Stats grid for multi-dimensional search spaces.
//!
//! A `StatsGrid` defines the search space over data statistics like length,
//! density, sparsity, etc. Each dimension is named and has an associated
//! [`Scale`] that generates the values to search.
//!
//! # Example
//!
//! ```
//! use vortex_threshold_traits::{StatsGrid, Scale};
//!
//! let grid = StatsGrid::new()
//!     .dimension("len", Scale::log2(6, 12))           // 64 to 4096
//!     .dimension("density", Scale::steps(0.0, 1.0, 5)); // 0%, 25%, 50%, 75%, 100%
//!
//! // Iterate over all combinations
//! for point in grid.iter() {
//!     println!("len={}, density={}", point["len"], point["density"]);
//! }
//! ```

use std::collections::HashMap;

use serde::Deserialize;
use serde::Serialize;

use crate::Scale;

/// A named dimension in the stats grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dimension {
    /// Name of this dimension (e.g., "len", "density")
    pub name: String,
    /// Scale defining the values for this dimension
    pub scale: Scale,
}

/// A point in the stats grid - a specific combination of dimension values.
#[derive(Debug, Clone, Default)]
pub struct StatsPoint {
    values: HashMap<String, f64>,
}

impl StatsPoint {
    /// Creates a new empty stats point.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a dimension value.
    #[must_use]
    pub fn with(mut self, name: impl Into<String>, value: f64) -> Self {
        self.values.insert(name.into(), value);
        self
    }

    /// Gets a dimension value by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<f64> {
        self.values.get(name).copied()
    }

    /// Gets a dimension value as usize.
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn get_usize(&self, name: &str) -> Option<usize> {
        self.values.get(name).map(|&v| v as usize)
    }

    /// Returns an iterator over all (name, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, f64)> {
        self.values.iter().map(|(k, &v)| (k.as_str(), v))
    }

    /// Returns the number of dimensions.
    #[must_use]
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns true if this point has no dimensions.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

impl std::ops::Index<&str> for StatsPoint {
    type Output = f64;

    #[allow(clippy::panic)]
    fn index(&self, name: &str) -> &Self::Output {
        self.values
            .get(name)
            .unwrap_or_else(|| panic!("dimension '{}' not found in StatsPoint", name))
    }
}

/// Multi-dimensional grid over stats values.
///
/// Use this to define the search space for threshold finding.
/// Each dimension represents a data statistic (like length or density)
/// and has a scale defining which values to test.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StatsGrid {
    dimensions: Vec<Dimension>,
}

impl StatsGrid {
    /// Creates a new empty stats grid.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a dimension to the grid.
    ///
    /// # Example
    ///
    /// ```
    /// use vortex_threshold_traits::{StatsGrid, Scale};
    ///
    /// let grid = StatsGrid::new()
    ///     .dimension("len", Scale::log2(6, 10))
    ///     .dimension("density", Scale::steps(0.0, 1.0, 3));
    /// ```
    #[must_use]
    pub fn dimension(mut self, name: impl Into<String>, scale: Scale) -> Self {
        self.dimensions.push(Dimension {
            name: name.into(),
            scale,
        });
        self
    }

    /// Returns the dimensions in this grid.
    #[must_use]
    pub fn dimensions(&self) -> &[Dimension] {
        &self.dimensions
    }

    /// Returns the total number of points in the grid (product of all dimension sizes).
    #[must_use]
    pub fn total_points(&self) -> usize {
        if self.dimensions.is_empty() {
            return 1; // Single point with no dimensions
        }
        self.dimensions.iter().map(|d| d.scale.len()).product()
    }

    /// Returns an iterator over all points in the grid.
    ///
    /// Iterates in row-major order (last dimension varies fastest).
    pub fn iter(&self) -> StatsGridIter<'_> {
        StatsGridIter::new(self)
    }
}

/// Iterator over all points in a stats grid.
pub struct StatsGridIter<'a> {
    grid: &'a StatsGrid,
    /// Current indices into each dimension
    indices: Vec<usize>,
    /// Cached values for each dimension
    dim_values: Vec<Vec<f64>>,
    /// Whether iteration is complete
    done: bool,
}

impl<'a> StatsGridIter<'a> {
    fn new(grid: &'a StatsGrid) -> Self {
        let dim_values: Vec<Vec<f64>> = grid
            .dimensions
            .iter()
            .map(|d| d.scale.values().collect())
            .collect();

        let done = dim_values.iter().any(|v| v.is_empty());
        let indices = vec![0; grid.dimensions.len()];

        Self {
            grid,
            indices,
            dim_values,
            done,
        }
    }

    fn current_point(&self) -> StatsPoint {
        let mut point = StatsPoint::new();
        for (i, dim) in self.grid.dimensions.iter().enumerate() {
            point
                .values
                .insert(dim.name.clone(), self.dim_values[i][self.indices[i]]);
        }
        point
    }

    fn advance(&mut self) {
        if self.grid.dimensions.is_empty() {
            self.done = true;
            return;
        }

        // Increment last dimension, carrying over as needed
        let mut i = self.indices.len() - 1;
        loop {
            self.indices[i] += 1;
            if self.indices[i] < self.dim_values[i].len() {
                return; // No carry needed
            }
            // Carry: reset this dimension and increment the previous one
            self.indices[i] = 0;
            if i == 0 {
                self.done = true;
                return;
            }
            i -= 1;
        }
    }
}

impl<'a> Iterator for StatsGridIter<'a> {
    type Item = StatsPoint;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }

        // Handle empty grid (no dimensions) - return one empty point
        if self.grid.dimensions.is_empty() {
            self.done = true;
            return Some(StatsPoint::new());
        }

        let point = self.current_point();
        self.advance();
        Some(point)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.done {
            (0, Some(0))
        } else {
            let remaining = self.grid.total_points();
            (remaining, Some(remaining))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_point() {
        let point = StatsPoint::new().with("len", 1024.0).with("density", 0.5);

        assert_eq!(point["len"], 1024.0);
        assert_eq!(point["density"], 0.5);
        assert_eq!(point.get_usize("len"), Some(1024));
        assert_eq!(point.len(), 2);
    }

    #[test]
    fn test_empty_grid() {
        let grid = StatsGrid::new();
        let points: Vec<_> = grid.iter().collect();
        assert_eq!(points.len(), 1); // One empty point
        assert!(points[0].is_empty());
    }

    #[test]
    fn test_single_dimension() {
        let grid = StatsGrid::new().dimension("len", Scale::explicit(vec![64.0, 128.0, 256.0]));

        let points: Vec<_> = grid.iter().collect();
        assert_eq!(points.len(), 3);
        assert_eq!(points[0]["len"], 64.0);
        assert_eq!(points[1]["len"], 128.0);
        assert_eq!(points[2]["len"], 256.0);
    }

    #[test]
    fn test_two_dimensions() {
        let grid = StatsGrid::new()
            .dimension("len", Scale::explicit(vec![64.0, 128.0]))
            .dimension("density", Scale::explicit(vec![0.0, 0.5, 1.0]));

        assert_eq!(grid.total_points(), 6);

        let points: Vec<_> = grid.iter().collect();
        assert_eq!(points.len(), 6);

        // Row-major order: density varies fastest
        assert_eq!((points[0]["len"], points[0]["density"]), (64.0, 0.0));
        assert_eq!((points[1]["len"], points[1]["density"]), (64.0, 0.5));
        assert_eq!((points[2]["len"], points[2]["density"]), (64.0, 1.0));
        assert_eq!((points[3]["len"], points[3]["density"]), (128.0, 0.0));
        assert_eq!((points[4]["len"], points[4]["density"]), (128.0, 0.5));
        assert_eq!((points[5]["len"], points[5]["density"]), (128.0, 1.0));
    }

    #[test]
    fn test_three_dimensions() {
        let grid = StatsGrid::new()
            .dimension("a", Scale::explicit(vec![1.0, 2.0]))
            .dimension("b", Scale::explicit(vec![10.0, 20.0]))
            .dimension("c", Scale::explicit(vec![100.0, 200.0]));

        assert_eq!(grid.total_points(), 8);

        let points: Vec<_> = grid.iter().collect();
        assert_eq!(points.len(), 8);

        // First point
        assert_eq!(points[0]["a"], 1.0);
        assert_eq!(points[0]["b"], 10.0);
        assert_eq!(points[0]["c"], 100.0);

        // Last point
        assert_eq!(points[7]["a"], 2.0);
        assert_eq!(points[7]["b"], 20.0);
        assert_eq!(points[7]["c"], 200.0);
    }

    #[test]
    fn test_with_log2_scale() {
        let grid = StatsGrid::new()
            .dimension("len", Scale::log2(6, 8))
            .dimension("density", Scale::steps(0.0, 1.0, 3));

        assert_eq!(grid.total_points(), 9); // 3 lengths × 3 densities

        let points: Vec<_> = grid.iter().collect();
        assert_eq!(points.len(), 9);

        // Check first and last
        assert_eq!(points[0]["len"], 64.0);
        assert_eq!(points[0]["density"], 0.0);
        assert_eq!(points[8]["len"], 256.0);
        assert_eq!(points[8]["density"], 1.0);
    }
}
