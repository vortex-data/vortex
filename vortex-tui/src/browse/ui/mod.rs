// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! UI rendering components for the TUI browser.

mod layouts;
mod query;
mod segments;

use layouts::render_layouts;
pub use query::QueryFocus;
pub use query::QueryState;
pub use query::SortDirection;
use query::render_query;
use ratatui::prelude::*;
use ratatui::widgets::Block;
use ratatui::widgets::BorderType;
use ratatui::widgets::Borders;
use ratatui::widgets::Tabs;
pub use segments::SegmentGridState;

use super::app::AppState;
use super::app::KeyMode;
use super::app::Tab;
use crate::browse::ui::segments::segments_ui;

/// Render the complete TUI application to the given frame.
///
/// This is the main entry point for rendering. It draws:
/// - The outer border with title and help text
/// - The tab bar showing available views
/// - The content area for the currently selected tab
pub fn render_app(app: &mut AppState<'_>, frame: &mut Frame<'_>) {
    // Render the outer tab view, then render the inner frame view.
    let bottom_text = if app.key_mode == KeyMode::Search {
        Line::from(format!(
            "Searching (press esc to exit): {}",
            app.search_filter.as_str()
        ))
        .yellow()
        .on_black()
        .left_aligned()
    } else {
        Line::from("press q to quit |  ← to go up a level | ENTER to select | / to search")
            .centered()
    };
    let shell = Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().magenta())
        .title_top("Vortex Browser")
        .title_bottom(bottom_text)
        .title_alignment(Alignment::Center);

    // The rest of the app is rendered inside the shell.
    let inner_area = shell.inner(frame.area());

    frame.render_widget(shell, frame.area());

    // Split the inner area into a Tab view area and the rest of the screen.
    let [tab_view, app_view] = Layout::vertical([
        // Tab bar area - 1 line
        Constraint::Length(1),
        // Rest of the interior space for app view
        Constraint::Min(0),
    ])
    .areas(inner_area);

    // Display a tab indicator.
    let selected_tab = match app.current_tab {
        Tab::Layout => 0,
        Tab::Segments => 1,
        Tab::Query => 2,
    };

    let tabs = Tabs::new(["File Layout", "Segments", "Query"])
        .style(Style::default().bold().white())
        .highlight_style(Style::default().bold().black().on_white())
        .select(Some(selected_tab));

    frame.render_widget(tabs, tab_view);

    // Render the view for the current tab.
    match app.current_tab {
        Tab::Layout => {
            render_layouts(app, app_view, frame.buffer_mut());
        }
        Tab::Segments => segments_ui(app, app_view, frame.buffer_mut()),
        Tab::Query => render_query(app, app_view, frame.buffer_mut()),
    }
}
