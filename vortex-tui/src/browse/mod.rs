use std::path::Path;

use app::{AppState, KeyMode, Tab, create_file_app};
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::DefaultTerminal;
use ratatui::widgets::ListState;
use ui::render_app;
use vortex::error::{VortexExpect, VortexResult};

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
            match (key.code, key.modifiers) {
                (KeyCode::Char('q'), _) => {
                    // Close the process down.
                    return HandleResult::Exit;
                }
                (KeyCode::Tab, _) => {
                    // toggle between tabs
                    app.current_tab = match app.current_tab {
                        Tab::Layout => Tab::Segments,
                        Tab::Segments => Tab::Layout,
                    };
                }
                (KeyCode::Up | KeyCode::Char('k'), _)
                | (KeyCode::Char('p'), KeyModifiers::CONTROL) => {
                    // We send the key-up to the list state if we're looking at
                    // the Layouts tab.
                    match app.current_tab {
                        Tab::Layout => {
                            app.layouts_list_state.select_previous();
                        }
                        Tab::Segments => {
                            app.segment_grid_state.vertical_scroll =
                                app.segment_grid_state.vertical_scroll.saturating_sub(10);
                            app.segment_grid_state.vertical_scroll_state = app
                                .segment_grid_state
                                .vertical_scroll_state
                                .position(app.segment_grid_state.vertical_scroll);
                        }
                    }
                }
                (KeyCode::Down | KeyCode::Char('j'), _)
                | (KeyCode::Char('n'), KeyModifiers::CONTROL) => match app.current_tab {
                    Tab::Layout => {
                        app.layouts_list_state.select_next();
                    }
                    Tab::Segments => {
                        app.segment_grid_state.vertical_scroll = app
                            .segment_grid_state
                            .vertical_scroll
                            .saturating_add(10)
                            .min(app.segment_grid_state.max_vertical_scroll);
                        app.segment_grid_state.vertical_scroll_state = app
                            .segment_grid_state
                            .vertical_scroll_state
                            .position(app.segment_grid_state.vertical_scroll);
                    }
                },
                (KeyCode::PageUp, _) | (KeyCode::Char('v'), KeyModifiers::ALT) => {
                    match app.current_tab {
                        Tab::Layout => {
                            app.layouts_list_state.scroll_up_by(10);
                        }
                        Tab::Segments => {
                            app.segment_grid_state.vertical_scroll =
                                app.segment_grid_state.vertical_scroll.saturating_sub(100);
                            app.segment_grid_state.vertical_scroll_state = app
                                .segment_grid_state
                                .vertical_scroll_state
                                .position(app.segment_grid_state.vertical_scroll);
                        }
                    }
                }
                (KeyCode::PageDown, _) | (KeyCode::Char('v'), KeyModifiers::CONTROL) => {
                    match app.current_tab {
                        Tab::Layout => {
                            app.layouts_list_state.scroll_down_by(10);
                        }
                        Tab::Segments => {
                            app.segment_grid_state.vertical_scroll = app
                                .segment_grid_state
                                .vertical_scroll
                                .saturating_add(100)
                                .min(app.segment_grid_state.max_vertical_scroll);
                            app.segment_grid_state.vertical_scroll_state = app
                                .segment_grid_state
                                .vertical_scroll_state
                                .position(app.segment_grid_state.vertical_scroll);
                        }
                    }
                }
                (KeyCode::Home, _) | (KeyCode::Char('<'), KeyModifiers::ALT) => {
                    match app.current_tab {
                        Tab::Layout => {
                            app.layouts_list_state.select_first();
                        }
                        Tab::Segments => {
                            app.segment_grid_state.horizontal_scroll =
                                app.segment_grid_state.horizontal_scroll.saturating_sub(200);
                            app.segment_grid_state.horizontal_scroll_state = app
                                .segment_grid_state
                                .horizontal_scroll_state
                                .position(app.segment_grid_state.horizontal_scroll);
                        }
                    }
                }
                (KeyCode::End, _) | (KeyCode::Char('>'), KeyModifiers::ALT) => {
                    match app.current_tab {
                        Tab::Layout => {
                            app.layouts_list_state.select_last();
                        }
                        Tab::Segments => {
                            app.segment_grid_state.horizontal_scroll = app
                                .segment_grid_state
                                .horizontal_scroll
                                .saturating_add(200)
                                .min(app.segment_grid_state.max_horizontal_scroll);
                            app.segment_grid_state.horizontal_scroll_state = app
                                .segment_grid_state
                                .horizontal_scroll_state
                                .position(app.segment_grid_state.horizontal_scroll);
                        }
                    }
                }
                (KeyCode::Enter, _) => {
                    if app.current_tab == Tab::Layout && app.cursor.layout().nchildren() > 0 {
                        // Descend into the layout subtree for the selected child.
                        let selected = app.layouts_list_state.selected().unwrap_or_default();
                        app.cursor = app.cursor.child(selected);

                        // Reset the list scroll state.
                        app.layouts_list_state = ListState::default().with_selected(Some(0));
                    }
                }
                (KeyCode::Left | KeyCode::Char('h'), _)
                | (KeyCode::Char('b'), KeyModifiers::CONTROL) => {
                    match app.current_tab {
                        Tab::Layout => {
                            // Ascend back up to the Parent node
                            app.cursor = app.cursor.parent();
                            // Reset the list scroll state.
                            app.layouts_list_state = ListState::default().with_selected(Some(0));
                        }
                        Tab::Segments => {
                            app.segment_grid_state.horizontal_scroll =
                                app.segment_grid_state.horizontal_scroll.saturating_sub(20);
                            app.segment_grid_state.horizontal_scroll_state = app
                                .segment_grid_state
                                .horizontal_scroll_state
                                .position(app.segment_grid_state.horizontal_scroll);
                        }
                    }
                }
                (KeyCode::Right | KeyCode::Char('l'), _)
                | (KeyCode::Char('b'), KeyModifiers::ALT) => match app.current_tab {
                    Tab::Layout => {}
                    Tab::Segments => {
                        app.segment_grid_state.horizontal_scroll = app
                            .segment_grid_state
                            .horizontal_scroll
                            .saturating_add(20)
                            .min(app.segment_grid_state.max_horizontal_scroll);
                        app.segment_grid_state.horizontal_scroll_state = app
                            .segment_grid_state
                            .horizontal_scroll_state
                            .position(app.segment_grid_state.horizontal_scroll);
                    }
                },

                (KeyCode::Char('/'), _) | (KeyCode::Char('s'), KeyModifiers::CONTROL) => {
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
        match (key.code, key.modifiers) {
            (KeyCode::Esc, _) | (KeyCode::Char('g'), KeyModifiers::CONTROL) => {
                // Exit search mode.
                //
                // Kill the search bar and search filtering and return to normal input processing.
                app.key_mode = KeyMode::Normal;
                app.clear_search();
            }

            // Use same navigation as Normal mode
            (KeyCode::Up, _) | (KeyCode::Char('p'), KeyModifiers::CONTROL) => {
                // We send the key-up to the list state if we're looking at
                // the Layouts tab.
                if app.current_tab == Tab::Layout {
                    app.layouts_list_state.scroll_up_by(1);
                }
            }
            (KeyCode::Down, _) | (KeyCode::Char('n'), KeyModifiers::CONTROL) => {
                if app.current_tab == Tab::Layout {
                    app.layouts_list_state.scroll_down_by(1);
                }
            }
            (KeyCode::PageUp, _) | (KeyCode::Char('v'), KeyModifiers::ALT) => {
                if app.current_tab == Tab::Layout {
                    app.layouts_list_state.scroll_up_by(10);
                }
            }
            (KeyCode::PageDown, _) | (KeyCode::Char('v'), KeyModifiers::CONTROL) => {
                if app.current_tab == Tab::Layout {
                    app.layouts_list_state.scroll_down_by(10);
                }
            }
            (KeyCode::Home, _) | (KeyCode::Char('<'), KeyModifiers::ALT) => {
                if app.current_tab == Tab::Layout {
                    app.layouts_list_state.select_first();
                }
            }
            (KeyCode::End, _) | (KeyCode::Char('>'), KeyModifiers::ALT) => {
                if app.current_tab == Tab::Layout {
                    app.layouts_list_state.select_last();
                }
            }

            (KeyCode::Enter, _) => {
                // Change back to normal mode.
                //
                // We can eliminate the search filter when we do this
                if app.current_tab == Tab::Layout && app.cursor.layout().nchildren() > 0 {
                    // Descend into the layout subtree for the selected child, do nothing if there's nothing to select.
                    if let Some(selected) = app.layouts_list_state.selected() {
                        app.cursor = match app.filter.as_ref() {
                            None => app.cursor.child(selected),
                            Some(filter) => {
                                let child_idx = filter
                                    .iter()
                                    .enumerate()
                                    .filter_map(|(idx, show)| show.then_some(idx))
                                    .nth(selected)
                                    .vortex_expect("There must be a selected item in the filter");

                                app.cursor.child(child_idx)
                            }
                        };

                        // Reset the list scroll state.
                        app.layouts_list_state = ListState::default().with_selected(Some(0));

                        app.clear_search();
                        // Return to normal mode.
                        app.key_mode = KeyMode::Normal;
                    }
                }
            }

            (KeyCode::Backspace, _) | (KeyCode::Char('h'), KeyModifiers::CONTROL) => {
                app.search_filter.pop();
            }

            (KeyCode::Char(c), _) => {
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
