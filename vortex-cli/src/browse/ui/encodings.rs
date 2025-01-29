use ratatui::prelude::Widget;
use ratatui::widgets::Paragraph;

use crate::browse::app::AppState;

pub fn encodings_ui(_app_state: &AppState) -> impl Widget {
    Paragraph::new("TODO: Encodings View").centered()
}
