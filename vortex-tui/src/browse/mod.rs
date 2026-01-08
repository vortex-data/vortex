// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Interactive TUI browser for Vortex files.

use std::path::Path;

use app::AppState;
use app::KeyMode;
use app::Tab;
use crossterm::event;
use crossterm::event::Event;
use crossterm::event::KeyCode;
use crossterm::event::KeyEventKind;
use crossterm::event::KeyModifiers;
use ratatui::DefaultTerminal;
use ui::render_app;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::layout::layouts::flat::FlatVTable;
use vortex::session::VortexSession;

pub mod app;
pub mod ui;

/// Scroll amount for single-line navigation (up/down arrows).
const SCROLL_LINE: usize = 1;
/// Scroll amount for page navigation (PageUp/PageDown).
const SCROLL_PAGE: usize = 10;
/// Scroll amount for segment grid line navigation.
const SEGMENT_SCROLL_LINE: usize = 10;
/// Scroll amount for segment grid page navigation.
const SEGMENT_SCROLL_PAGE: usize = 100;
/// Scroll amount for segment grid horizontal step.
const SEGMENT_SCROLL_HORIZONTAL_STEP: usize = 20;
/// Scroll amount for segment grid horizontal jump (Home/End).
const SEGMENT_SCROLL_HORIZONTAL_JUMP: usize = 200;

// Use the VortexResult and potentially launch a Backtrace.
async fn run(mut terminal: DefaultTerminal, mut app: AppState<'_>) -> VortexResult<()> {
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

/// Navigate the layout list up by the given amount.
fn navigate_layout_up(app: &mut AppState, amount: usize) {
    let amount_u16 = amount.try_into().unwrap_or(u16::MAX);
    if app.cursor.layout().is::<FlatVTable>() {
        app.tree_scroll_offset = app.tree_scroll_offset.saturating_sub(amount_u16);
    } else {
        app.layouts_list_state.scroll_up_by(amount_u16);
    }
}

/// Navigate the layout list down by the given amount.
fn navigate_layout_down(app: &mut AppState, amount: usize) {
    let amount_u16 = amount.try_into().unwrap_or(u16::MAX);
    if app.cursor.layout().is::<FlatVTable>() {
        app.tree_scroll_offset = app.tree_scroll_offset.saturating_add(amount_u16);
    } else {
        app.layouts_list_state.scroll_down_by(amount_u16);
    }
}

fn handle_normal_mode(app: &mut AppState, event: Event) -> HandleResult {
    if let Event::Key(key) = event
        && key.kind == KeyEventKind::Press
    {
        match (key.code, key.modifiers) {
            (KeyCode::Char('q'), _) => {
                return HandleResult::Exit;
            }
            (KeyCode::Tab, _) => {
                app.current_tab = match app.current_tab {
                    Tab::Layout => Tab::Segments,
                    Tab::Segments => Tab::Layout,
                };
            }
            (KeyCode::Up | KeyCode::Char('k'), _) | (KeyCode::Char('p'), KeyModifiers::CONTROL) => {
                match app.current_tab {
                    Tab::Layout => navigate_layout_up(app, SCROLL_LINE),
                    Tab::Segments => app.segment_grid_state.scroll_up(SEGMENT_SCROLL_LINE),
                }
            }
            (KeyCode::Down | KeyCode::Char('j'), _)
            | (KeyCode::Char('n'), KeyModifiers::CONTROL) => match app.current_tab {
                Tab::Layout => navigate_layout_down(app, SCROLL_LINE),
                Tab::Segments => app.segment_grid_state.scroll_down(SEGMENT_SCROLL_LINE),
            },
            (KeyCode::PageUp, _) | (KeyCode::Char('v'), KeyModifiers::ALT) => {
                match app.current_tab {
                    Tab::Layout => navigate_layout_up(app, SCROLL_PAGE),
                    Tab::Segments => app.segment_grid_state.scroll_up(SEGMENT_SCROLL_PAGE),
                }
            }
            (KeyCode::PageDown, _) | (KeyCode::Char('v'), KeyModifiers::CONTROL) => {
                match app.current_tab {
                    Tab::Layout => navigate_layout_down(app, SCROLL_PAGE),
                    Tab::Segments => app.segment_grid_state.scroll_down(SEGMENT_SCROLL_PAGE),
                }
            }
            (KeyCode::Home, _) | (KeyCode::Char('<'), KeyModifiers::ALT) => match app.current_tab {
                Tab::Layout => app.layouts_list_state.select_first(),
                Tab::Segments => app
                    .segment_grid_state
                    .scroll_left(SEGMENT_SCROLL_HORIZONTAL_JUMP),
            },
            (KeyCode::End, _) | (KeyCode::Char('>'), KeyModifiers::ALT) => match app.current_tab {
                Tab::Layout => app.layouts_list_state.select_last(),
                Tab::Segments => app
                    .segment_grid_state
                    .scroll_right(SEGMENT_SCROLL_HORIZONTAL_JUMP),
            },
            (KeyCode::Enter, _) => {
                if app.current_tab == Tab::Layout && app.cursor.layout().nchildren() > 0 {
                    // Descend into the layout subtree for the selected child.
                    let selected = app.layouts_list_state.selected().unwrap_or_default();
                    app.cursor = app.cursor.child(selected);
                    app.reset_layout_view_state();
                }
            }
            (KeyCode::Left | KeyCode::Char('h'), _)
            | (KeyCode::Char('b'), KeyModifiers::CONTROL) => match app.current_tab {
                Tab::Layout => {
                    app.cursor = app.cursor.parent();
                    app.reset_layout_view_state();
                }
                Tab::Segments => app
                    .segment_grid_state
                    .scroll_left(SEGMENT_SCROLL_HORIZONTAL_STEP),
            },
            (KeyCode::Right | KeyCode::Char('l'), _) | (KeyCode::Char('b'), KeyModifiers::ALT) => {
                match app.current_tab {
                    Tab::Layout => {}
                    Tab::Segments => app
                        .segment_grid_state
                        .scroll_right(SEGMENT_SCROLL_HORIZONTAL_STEP),
                }
            }

            (KeyCode::Char('/'), _) | (KeyCode::Char('s'), KeyModifiers::CONTROL) => {
                app.key_mode = KeyMode::Search;
            }

            _ => {}
        }
    }

    HandleResult::Continue
}

fn handle_search_mode(app: &mut AppState, event: Event) -> HandleResult {
    if let Event::Key(key) = event {
        match (key.code, key.modifiers) {
            (KeyCode::Esc, _) | (KeyCode::Char('g'), KeyModifiers::CONTROL) => {
                app.key_mode = KeyMode::Normal;
                app.clear_search();
            }

            (KeyCode::Up, _) | (KeyCode::Char('p'), KeyModifiers::CONTROL) => {
                if app.current_tab == Tab::Layout {
                    navigate_layout_up(app, SCROLL_LINE);
                }
            }
            (KeyCode::Down, _) | (KeyCode::Char('n'), KeyModifiers::CONTROL) => {
                if app.current_tab == Tab::Layout {
                    navigate_layout_down(app, SCROLL_LINE);
                }
            }
            (KeyCode::PageUp, _) | (KeyCode::Char('v'), KeyModifiers::ALT) => {
                if app.current_tab == Tab::Layout {
                    navigate_layout_up(app, SCROLL_PAGE);
                }
            }
            (KeyCode::PageDown, _) | (KeyCode::Char('v'), KeyModifiers::CONTROL) => {
                if app.current_tab == Tab::Layout {
                    navigate_layout_down(app, SCROLL_PAGE);
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
                if app.current_tab == Tab::Layout
                    && app.cursor.layout().nchildren() > 0
                    && let Some(selected) = app.layouts_list_state.selected()
                {
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

                    app.reset_layout_view_state();
                    app.clear_search();
                    app.key_mode = KeyMode::Normal;
                }
            }

            (KeyCode::Backspace, _) | (KeyCode::Char('h'), KeyModifiers::CONTROL) => {
                app.search_filter.pop();
            }

            (KeyCode::Char(c), _) => {
                app.layouts_list_state.select_first();
                app.search_filter.push(c);
            }

            _ => {}
        }
    }

    HandleResult::Continue
}

// TODO: add tui_logger and have a logs tab so we can see the log output from
//  doing Vortex things.

/// Launch the interactive TUI browser for a Vortex file.
///
/// # Errors
///
/// Returns an error if the file cannot be opened or if there's a terminal I/O error.
pub async fn exec_tui(session: &VortexSession, file: impl AsRef<Path>) -> VortexResult<()> {
    let app = AppState::new(session, file).await?;

    let mut terminal = ratatui::init();
    terminal.clear()?;

    run(terminal, app).await?;

    ratatui::restore();
    Ok(())
}
