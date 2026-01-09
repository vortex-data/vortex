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
use ui::QueryFocus;
use ui::SortDirection;
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

#[allow(clippy::cognitive_complexity)]
fn handle_normal_mode(app: &mut AppState, event: Event) -> HandleResult {
    if let Event::Key(key) = event
        && key.kind == KeyEventKind::Press
    {
        // Check if we're in Query tab with SQL input focus - handle text input first
        let in_sql_input =
            app.current_tab == Tab::Query && app.query_state.focus == QueryFocus::SqlInput;

        // Handle SQL input mode - most keys should type into the input
        if in_sql_input {
            match (key.code, key.modifiers) {
                // These keys exit/switch even in SQL input mode
                (KeyCode::Tab, _) => {
                    app.current_tab = Tab::Layout;
                }
                (KeyCode::Esc, _) => {
                    app.query_state.toggle_focus();
                }
                (KeyCode::Enter, _) => {
                    // Execute the SQL query with COUNT(*) for pagination
                    app.query_state.sort_column = None;
                    app.query_state.sort_direction = SortDirection::None;
                    let file_path = app.file_path.clone();
                    app.query_state
                        .execute_initial_query(app.session, &file_path);
                    // Switch focus to results table after executing
                    app.query_state.focus = QueryFocus::ResultsTable;
                }
                // Navigation keys
                (KeyCode::Left, _) => app.query_state.move_cursor_left(),
                (KeyCode::Right, _) => app.query_state.move_cursor_right(),
                (KeyCode::Home, _) => app.query_state.move_cursor_start(),
                (KeyCode::End, _) => app.query_state.move_cursor_end(),
                // Control key shortcuts
                (KeyCode::Char('a'), KeyModifiers::CONTROL) => app.query_state.move_cursor_start(),
                (KeyCode::Char('e'), KeyModifiers::CONTROL) => app.query_state.move_cursor_end(),
                (KeyCode::Char('u'), KeyModifiers::CONTROL) => app.query_state.clear_input(),
                (KeyCode::Char('b'), KeyModifiers::CONTROL) => app.query_state.move_cursor_left(),
                (KeyCode::Char('f'), KeyModifiers::CONTROL) => app.query_state.move_cursor_right(),
                (KeyCode::Char('d'), KeyModifiers::CONTROL) => {
                    app.query_state.delete_char_forward()
                }
                // Delete keys
                (KeyCode::Backspace, _) => app.query_state.delete_char(),
                (KeyCode::Delete, _) => app.query_state.delete_char_forward(),
                // All other characters get typed into the input
                (KeyCode::Char(c), KeyModifiers::NONE | KeyModifiers::SHIFT) => {
                    app.query_state.insert_char(c);
                }
                _ => {}
            }
            return HandleResult::Continue;
        }

        // Normal mode handling for all other cases
        match (key.code, key.modifiers) {
            (KeyCode::Char('q'), _) => {
                return HandleResult::Exit;
            }
            (KeyCode::Tab, _) => {
                app.current_tab = match app.current_tab {
                    Tab::Layout => Tab::Segments,
                    Tab::Segments => Tab::Query,
                    Tab::Query => Tab::Layout,
                };
            }

            // Query tab: '[' for previous page
            (KeyCode::Char('['), KeyModifiers::NONE) => {
                if app.current_tab == Tab::Query {
                    app.query_state
                        .prev_page(app.session, &app.file_path.clone());
                }
            }

            // Query tab: ']' for next page
            (KeyCode::Char(']'), KeyModifiers::NONE) => {
                if app.current_tab == Tab::Query {
                    app.query_state
                        .next_page(app.session, &app.file_path.clone());
                }
            }

            (KeyCode::Up | KeyCode::Char('k'), _) | (KeyCode::Char('p'), KeyModifiers::CONTROL) => {
                match app.current_tab {
                    Tab::Layout => navigate_layout_up(app, SCROLL_LINE),
                    Tab::Segments => app.segment_grid_state.scroll_up(SEGMENT_SCROLL_LINE),
                    Tab::Query => {
                        app.query_state.table_state.select_previous();
                    }
                }
            }
            (KeyCode::Down | KeyCode::Char('j'), _)
            | (KeyCode::Char('n'), KeyModifiers::CONTROL) => match app.current_tab {
                Tab::Layout => navigate_layout_down(app, SCROLL_LINE),
                Tab::Segments => app.segment_grid_state.scroll_down(SEGMENT_SCROLL_LINE),
                Tab::Query => {
                    app.query_state.table_state.select_next();
                }
            },
            (KeyCode::PageUp, _) | (KeyCode::Char('v'), KeyModifiers::ALT) => {
                match app.current_tab {
                    Tab::Layout => navigate_layout_up(app, SCROLL_PAGE),
                    Tab::Segments => app.segment_grid_state.scroll_up(SEGMENT_SCROLL_PAGE),
                    Tab::Query => {
                        app.query_state
                            .prev_page(app.session, &app.file_path.clone());
                    }
                }
            }
            (KeyCode::PageDown, _) | (KeyCode::Char('v'), KeyModifiers::CONTROL) => {
                match app.current_tab {
                    Tab::Layout => navigate_layout_down(app, SCROLL_PAGE),
                    Tab::Segments => app.segment_grid_state.scroll_down(SEGMENT_SCROLL_PAGE),
                    Tab::Query => {
                        app.query_state
                            .next_page(app.session, &app.file_path.clone());
                    }
                }
            }
            (KeyCode::Home, _) | (KeyCode::Char('<'), KeyModifiers::ALT) => match app.current_tab {
                Tab::Layout => app.layouts_list_state.select_first(),
                Tab::Segments => app
                    .segment_grid_state
                    .scroll_left(SEGMENT_SCROLL_HORIZONTAL_JUMP),
                Tab::Query => {
                    app.query_state.table_state.select_first();
                }
            },
            (KeyCode::End, _) | (KeyCode::Char('>'), KeyModifiers::ALT) => match app.current_tab {
                Tab::Layout => app.layouts_list_state.select_last(),
                Tab::Segments => app
                    .segment_grid_state
                    .scroll_right(SEGMENT_SCROLL_HORIZONTAL_JUMP),
                Tab::Query => {
                    app.query_state.table_state.select_last();
                }
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
                Tab::Query => {
                    app.query_state.horizontal_scroll =
                        app.query_state.horizontal_scroll.saturating_sub(1);
                }
            },
            (KeyCode::Right | KeyCode::Char('l'), _) | (KeyCode::Char('b'), KeyModifiers::ALT) => {
                match app.current_tab {
                    Tab::Layout => {}
                    Tab::Segments => app
                        .segment_grid_state
                        .scroll_right(SEGMENT_SCROLL_HORIZONTAL_STEP),
                    Tab::Query => {
                        let max_col = app.query_state.column_count().saturating_sub(1);
                        if app.query_state.horizontal_scroll < max_col {
                            app.query_state.horizontal_scroll += 1;
                        }
                    }
                }
            }

            (KeyCode::Char('/'), _) | (KeyCode::Char('s'), KeyModifiers::CONTROL) => {
                if app.current_tab != Tab::Query {
                    app.key_mode = KeyMode::Search;
                }
            }

            (KeyCode::Char('s'), KeyModifiers::NONE) => {
                if app.current_tab == Tab::Query {
                    // Sort by selected column - modifies the SQL query
                    let col = app.query_state.selected_column();
                    app.query_state.apply_sort(app.session, col, &app.file_path);
                }
            }

            (KeyCode::Esc, _) => {
                if app.current_tab == Tab::Query {
                    // Toggle focus in Query tab
                    app.query_state.toggle_focus();
                }
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
