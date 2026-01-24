//! SVG path generation for chart series.

use super::scales::scale_x;
use super::scales::scale_y;
use crate::db::Series;

/// A rendered series path ready for SVG output.
pub struct SeriesPath {
    /// CSS color value (e.g., "#19a508").
    pub color: String,
    /// SVG path data string (e.g., "M 60 200 L 70 195 L 80 198").
    pub path_d: String,
}

/// Generates SVG path data for all series.
///
/// # Arguments
/// * `series` - The series data to render
/// * `x_max` - Total number of commits (for X scaling)
/// * `y_min` - Minimum Y value (for Y scaling)
/// * `y_max` - Maximum Y value (for Y scaling)
pub fn generate_series_paths(
    series: &[Series],
    x_max: f64,
    y_min: f64,
    y_max: f64,
) -> Vec<SeriesPath> {
    series
        .iter()
        .map(|s| {
            let points: Vec<(f64, f64)> = s
                .points
                .iter()
                .map(|p| {
                    (
                        scale_x(p.commit_idx, x_max),
                        scale_y(p.value_ms(), y_min, y_max),
                    )
                })
                .collect();

            let path_d = points
                .iter()
                .enumerate()
                .map(|(i, (x, y))| {
                    if i == 0 {
                        format!("M {} {}", x, y)
                    } else {
                        format!("L {} {}", x, y)
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");

            SeriesPath {
                color: s.color.clone(),
                path_d,
            }
        })
        .collect()
}
