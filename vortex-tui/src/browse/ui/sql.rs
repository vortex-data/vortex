use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::widgets::{Paragraph, Widget};

use crate::browse::app::AppState;
use crate::browse::make_container;

pub fn render_sql(app_state: &mut AppState, area: Rect, buf: &mut Buffer) {
    let data = Paragraph::new("Data goes here");
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints(vec![Constraint::Percentage(20), Constraint::Percentage(80)])
        .split(area);

    let input_container = make_container("Input");
    let data_container = make_container("Data");

    let input_area = input_container.inner(layout[0]);
    let data_area = input_container.inner(layout[1]);

    input_container.render(layout[0], buf);
    data_container.render(layout[1], buf);

    app_state.sql_state.input.render(input_area, buf);
    data.render(data_area, buf);
}
