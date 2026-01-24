//! Axis tick mark computation for charts.

use super::scales::scale_x;
use super::scales::scale_y;
use crate::db::CommitInfo;

/// A tick mark with its pixel position and label text.
pub struct Tick {
    /// Pixel coordinate (X for x-ticks, Y for y-ticks).
    pub position: f64,
    /// Human-readable label.
    pub label: String,
}

/// Computes Y-axis tick marks with evenly spaced values.
///
/// # Arguments
/// * `y_min` - Minimum Y value
/// * `y_max` - Maximum Y value
/// * `count` - Number of tick marks (not including endpoints adds 1)
pub fn compute_y_ticks(y_min: f64, y_max: f64, count: usize) -> Vec<Tick> {
    (0..=count)
        .map(|i| {
            let val = y_min + (y_max - y_min) * (i as f64 / count as f64);
            Tick {
                position: scale_y(val, y_min, y_max),
                label: format!("{:.0}", val),
            }
        })
        .collect()
}

/// Computes X-axis tick marks at evenly spaced commit intervals.
///
/// Returns tick marks with date labels (MM/DD format).
///
/// # Arguments
/// * `commits` - The commit data (for timestamps)
/// * `count` - Number of tick marks
pub fn compute_x_ticks(commits: &[CommitInfo], count: usize) -> Vec<Tick> {
    let commit_count = commits.len();
    let x_max = commit_count as f64;

    (0..=count)
        .filter_map(|i| {
            let idx = (i * commit_count.saturating_sub(1)) / count;
            commits.get(idx).map(|c| Tick {
                position: scale_x(idx, x_max),
                label: c.timestamp.format("%m/%d").to_string(),
            })
        })
        .collect()
}
