use std::path::Path;

use app::{create_file_app, AppState, KeyMode, Tab};
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

        let event = event::read()?;
        let event_result = match app.key_mode {
            KeyMode::Normal => handle_normal_mode(&mut app, event),
            KeyMode::Search => handle_search_mode(&mut app, event),
        };

        match event_result {
            HandleResult::Exit => {
                return Ok(());
            }
            HandleResult::Continue => { /* continue execution */ }
        }
    }
}

enum HandleResult {
    Continue,
    Exit,
}

fn handle_normal_mode(app: &mut AppState, event: Event) -> HandleResult {
    if let Event::Key(key) = event {
        if key.kind == KeyEventKind::Press {
            match key.code {
                KeyCode::Char('q') => {
                    // Close the process down.
                    return HandleResult::Exit;
                }
                KeyCode::Tab => {
                    // toggle between tabs
                    app.current_tab = match app.current_tab {
                        Tab::Layout => Tab::Encodings,
                        Tab::Encodings => Tab::Layout,
                    };
                }
                KeyCode::Up | KeyCode::Char('k') => {
                    // We send the key-up to the list state if we're looking at
                    // the Layouts tab.
                    if app.current_tab == Tab::Layout {
                        app.layouts_list_state.scroll_up_by(1);
                    }
                }
                KeyCode::Down | KeyCode::Char('j') => {
                    if app.current_tab == Tab::Layout {
                        app.layouts_list_state.scroll_down_by(1);
                    }
                }
                KeyCode::PageUp => {
                    if app.current_tab == Tab::Layout {
                        app.layouts_list_state.scroll_up_by(10);
                    }
                }
                KeyCode::PageDown => {
                    if app.current_tab == Tab::Layout {
                        app.layouts_list_state.scroll_down_by(10);
                    }
                }
                KeyCode::Home => {
                    if app.current_tab == Tab::Layout {
                        app.layouts_list_state.select_first();
                    }
                }
                KeyCode::End => {
                    if app.current_tab == Tab::Layout {
                        app.layouts_list_state.select_last();
                    }
                }
                KeyCode::Enter => {
                    if app.current_tab == Tab::Layout && app.cursor.layout().nchildren() > 0 {
                        // Descend into the layout subtree for the selected child.
                        let selected = app.layouts_list_state.selected().unwrap_or_default();
                        app.cursor = app.cursor.child(selected);

                        // Reset the list scroll state.
                        app.layouts_list_state = ListState::default().with_selected(Some(0));
                    }
                }
                KeyCode::Left | KeyCode::Char('h') => {
                    if app.current_tab == Tab::Layout {
                        // Ascend back up to the Parent node
                        app.cursor = app.cursor.parent();
                        // Reset the list scroll state.
                        app.layouts_list_state = ListState::default().with_selected(Some(0));
                    }
                }

                KeyCode::Char('/') => {
                    app.key_mode = KeyMode::Search;
                }

                // Most events not handled
                _ => {}
            }
        }
    }

    HandleResult::Continue
}

fn handle_search_mode(app: &mut AppState, event: Event) -> HandleResult {
    if let Event::Key(key) = event {
        match key.code {
            KeyCode::Esc => {
                // Exit search mode.
                //
                // Kill the search bar and search filtering and return to normal input processing.
                app.key_mode = KeyMode::Normal;
                app.search_filter.clear();
            }

            // Use same navigation as Normal mode
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
            KeyCode::PageUp => {
                if app.current_tab == Tab::Layout {
                    app.layouts_list_state.scroll_up_by(10);
                }
            }
            KeyCode::PageDown => {
                if app.current_tab == Tab::Layout {
                    app.layouts_list_state.scroll_down_by(10);
                }
            }
            KeyCode::Home => {
                if app.current_tab == Tab::Layout {
                    app.layouts_list_state.select_first();
                }
            }
            KeyCode::End => {
                if app.current_tab == Tab::Layout {
                    app.layouts_list_state.select_last();
                }
            }

            KeyCode::Enter => {
                // Change back to normal mode.
                //
                // We can eliminate the search filter when we do this
                if app.current_tab == Tab::Layout && app.cursor.layout().nchildren() > 0 {
                    // Descend into the layout subtree for the selected child.
                    let selected = app.layouts_list_state.selected().unwrap_or_default();
                    app.cursor = app.cursor.child(selected);

                    // Reset the list scroll state.
                    app.layouts_list_state = ListState::default().with_selected(Some(0));

                    // Clear the search filter.
                    app.search_filter.clear();
                    // Return to normal mode.
                    app.key_mode = KeyMode::Normal;
                }
            }

            KeyCode::Backspace => {
                app.search_filter.pop();
            }

            KeyCode::Char(c) => {
                // reset selection state
                app.layouts_list_state.select_first();
                // append to our search string
                app.search_filter.push(c);
            }

            // Most events unhandled.
            _ => {}
        }
    }

    HandleResult::Continue
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
