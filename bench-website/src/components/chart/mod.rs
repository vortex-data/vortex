//! SVG chart component for rendering benchmark data.
//!
//! This module provides a server-side rendered SVG chart component.
//! For Phase 1, charts are static. Phase 2 will add interactivity
//! via the Plotters Canvas backend.
//!
//! # Submodules
//!
//! - [`scales`]: Coordinate transformation functions
//! - [`paths`]: SVG path data generation
//! - [`ticks`]: Axis tick mark computation

pub mod paths;
pub mod scales;
pub mod ticks;

use leptos::prelude::*;

use self::paths::generate_series_paths;
use self::scales::HEIGHT;
use self::scales::PADDING;
use self::scales::WIDTH;
use self::ticks::compute_x_ticks;
use self::ticks::compute_y_ticks;
use crate::db::ChartData;

/// Renders an SVG line chart for benchmark data.
///
/// Displays multiple series as colored lines with a legend.
/// Axes show time values (Y) and commit dates (X).
#[component]
pub fn Chart(data: ChartData) -> impl IntoView {
    let (y_min, y_max) = data.y_range();
    let x_max = data.commits.len() as f64;

    // Generate chart data using helper modules
    let paths = generate_series_paths(&data.series, x_max, y_min, y_max);
    let y_ticks = compute_y_ticks(y_min, y_max, 5);
    let x_ticks = compute_x_ticks(&data.commits, 5);

    let title = data.title.clone();
    let unit = data.unit.clone();

    view! {
        <div class="chart-container bg-white rounded-lg shadow-md p-4 mb-6">
            <h3 class="text-lg font-semibold text-gray-800 mb-4">{title}</h3>
            <svg
                width=WIDTH.to_string()
                height=HEIGHT.to_string()
                class="chart-svg"
                viewBox=format!("0 0 {} {}", WIDTH, HEIGHT)
            >
                // Background
                <rect
                    x="0" y="0"
                    width=WIDTH.to_string()
                    height=HEIGHT.to_string()
                    fill="#f9fafb"
                />

                // Grid lines
                {y_ticks.iter().map(|tick| {
                    view! {
                        <line
                            x1=PADDING.to_string()
                            y1=tick.position.to_string()
                            x2=(WIDTH - PADDING).to_string()
                            y2=tick.position.to_string()
                            stroke="#e5e7eb"
                            stroke-width="1"
                        />
                    }
                }).collect_view()}

                // Y-axis
                <line
                    x1=PADDING.to_string()
                    y1=PADDING.to_string()
                    x2=PADDING.to_string()
                    y2=(HEIGHT - PADDING).to_string()
                    stroke="#9ca3af"
                    stroke-width="1"
                />

                // X-axis
                <line
                    x1=PADDING.to_string()
                    y1=(HEIGHT - PADDING).to_string()
                    x2=(WIDTH - PADDING).to_string()
                    y2=(HEIGHT - PADDING).to_string()
                    stroke="#9ca3af"
                    stroke-width="1"
                />

                // Y-axis labels
                {y_ticks.iter().map(|tick| {
                    view! {
                        <text
                            x=(PADDING - 8.0).to_string()
                            y=tick.position.to_string()
                            text-anchor="end"
                            dominant-baseline="middle"
                            style="font-size: 10px; fill: #6b7280"
                        >
                            {tick.label.clone()}
                        </text>
                    }
                }).collect_view()}

                // Y-axis title
                <text
                    x="15"
                    y=(HEIGHT / 2.0).to_string()
                    transform=format!("rotate(-90, 15, {})", HEIGHT / 2.0)
                    text-anchor="middle"
                    style="font-size: 12px; fill: #4b5563"
                >
                    {format!("Time ({})", unit)}
                </text>

                // X-axis tick labels
                {x_ticks.iter().map(|tick| {
                    view! {
                        <text
                            x=tick.position.to_string()
                            y=(HEIGHT - PADDING + 15.0).to_string()
                            text-anchor="middle"
                            style="font-size: 10px; fill: #6b7280"
                        >
                            {tick.label.clone()}
                        </text>
                    }
                }).collect_view()}

                // Series lines
                {paths.iter().map(|p| {
                    view! {
                        <path
                            d=p.path_d.clone()
                            stroke=p.color.clone()
                            stroke-width="2"
                            fill="none"
                        />
                    }
                }).collect_view()}
            </svg>

            // Legend
            <div class="flex gap-6 mt-4 justify-center">
                {data.series.iter().map(|s| {
                    let color = s.color.clone();
                    let name = s.display_name.clone();
                    view! {
                        <div class="flex items-center gap-2">
                            <div
                                class="w-4 h-3 rounded-sm"
                                style=format!("background-color: {}", color)
                            ></div>
                            <span class="text-sm text-gray-700">{name}</span>
                        </div>
                    }
                }).collect_view()}
            </div>
        </div>
    }
}
