//! Coordinate scaling functions for chart rendering.
//!
//! These functions transform data coordinates to SVG pixel coordinates.

/// Chart dimension constants.
pub const WIDTH: f64 = 800.0;
pub const HEIGHT: f64 = 400.0;
pub const PADDING: f64 = 60.0;

/// Returns the usable chart width (total width minus padding on both sides).
pub fn chart_width() -> f64 {
    WIDTH - 2.0 * PADDING
}

/// Returns the usable chart height (total height minus padding on both sides).
pub fn chart_height() -> f64 {
    HEIGHT - 2.0 * PADDING
}

/// Transforms a commit index to an X pixel coordinate.
///
/// # Arguments
/// * `idx` - The commit index (0-based)
/// * `x_max` - The total number of commits
pub fn scale_x(idx: usize, x_max: f64) -> f64 {
    PADDING + (idx as f64 / x_max) * chart_width()
}

/// Transforms a data value to a Y pixel coordinate.
///
/// Note: SVG Y coordinates increase downward, so this inverts the value
/// to place higher values at the top of the chart.
///
/// # Arguments
/// * `val` - The data value (e.g., milliseconds)
/// * `y_min` - The minimum Y value in the data range
/// * `y_max` - The maximum Y value in the data range
pub fn scale_y(val: f64, y_min: f64, y_max: f64) -> f64 {
    HEIGHT - PADDING - ((val - y_min) / (y_max - y_min)) * chart_height()
}
