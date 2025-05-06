mod layouts;
mod segments;

use layouts::render_layouts;
use ratatui::prelude::*;
use ratatui::widgets::{Block, BorderType, Borders, Tabs};
pub use segments::SegmentGridState;

use super::app::{AppState, KeyMode, Tab};
use crate::browse::ui::segments::segments_ui;

pub fn render_app(app: &mut AppState, frame: &mut Frame) {
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
    };

    let tabs = Tabs::new([
        "File Layout",
        "Segments",
        // TODO(aduffy): add SQL query interface
        // "Query",
    ])
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
    }
}
