use std::path::Path;

use app::{create_file_app, AppState, Tab};
use crossterm::event;
use crossterm::event::{Event, KeyCode, KeyEventKind};
use ratatui::widgets::ListState;
use ratatui::DefaultTerminal;
use ui::render_app;
use vortex::error::VortexResult;

use crate::TOKIO_RUNTIME;

mod app;
mod ui;

// Use the VortexResult and potentially launch a Backtrace.
fn run(mut terminal: DefaultTerminal, mut app: AppState) -> VortexResult<()> {
    loop {
        terminal.draw(|frame| render_app(&mut app, frame))?;

        if let Event::Key(key) = event::read()? {
            if key.kind == KeyEventKind::Press {
                match key.code {
                    KeyCode::Char('q') => break Ok(()),
                    KeyCode::Tab => {
                        // toggle between tabs
                        app.current_tab = match app.current_tab {
                            Tab::Layout => Tab::Encodings,
                            Tab::Encodings => Tab::Layout,
                        };
                    }
                    KeyCode::Up => {
                        // We send the key-up to the list state if we're looking at
                        // the Layouts tab.
                        if app.current_tab == Tab::Layout {
                            app.layouts_list_state.scroll_up_by(1);
                        }
                    }
                    KeyCode::Down => {
                        if app.current_tab == Tab::Layout {
                            app.layouts_list_state.scroll_down_by(1);
                        }
                    }
                    KeyCode::Enter => {
                        if app.current_tab == Tab::Layout {
                            // Descend into the layout subtree for the selected child.
                            let selected = app.layouts_list_state.selected().unwrap_or_default();
                            app.cursor = app.cursor.child(selected);

                            // Reset the list scroll state.
                            app.layouts_list_state = ListState::default().with_selected(Some(0));
                        }
                    }
                    KeyCode::Left => {
                        if app.current_tab == Tab::Layout {
                            // Ascend back up to the Parent node
                            app.cursor = app.cursor.parent();
                            // Reset the list scroll state.
                            app.layouts_list_state = ListState::default().with_selected(Some(0));
                        }
                    }
                    // Most events not handled
                    _ => {}
                }
            }
        }
    }
}

// TODO: add tui_logger and have a logs tab so we can see the log output from
//  doing Vortex things.¬

pub fn exec_tui(file: impl AsRef<Path>) -> VortexResult<()> {
    let app = TOKIO_RUNTIME.block_on(create_file_app(file))?;

    let mut terminal = ratatui::init();
    terminal.clear()?;

    run(terminal, app)?;

    ratatui::restore();
    Ok(())
}
