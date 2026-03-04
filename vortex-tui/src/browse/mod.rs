// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Interactive TUI browser for Vortex files.

use app::AppState;
use app::KeyMode;
use app::Tab;
use input::InputEvent;
use input::InputKeyCode;
use vortex::error::VortexExpect;
use vortex::layout::layouts::flat::FlatVTable;

pub mod app;
pub(crate) mod input;
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

pub(crate) enum HandleResult {
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

/// Handle a key event in normal input mode.
///
/// Returns [`HandleResult::Exit`] if the user pressed the quit key.
#[allow(clippy::cognitive_complexity)]
pub(crate) fn handle_normal_mode(app: &mut AppState, event: InputEvent) -> HandleResult {
    // Check if we're in Query tab with SQL input focus - handle text input first
    #[cfg(not(target_arch = "wasm32"))]
    {
        use ui::QueryFocus;
        use ui::SortDirection;

        let in_sql_input =
            app.current_tab == Tab::Query && app.query_state.focus == QueryFocus::SqlInput;

        if in_sql_input {
            match (&event.code, event.ctrl, event.alt, event.shift) {
                (InputKeyCode::Tab, ..) => {
                    app.current_tab = Tab::Layout;
                }
                (InputKeyCode::Esc, ..) => {
                    app.query_state.toggle_focus();
                }
                (InputKeyCode::Enter, ..) => {
                    app.query_state.sort_column = None;
                    app.query_state.sort_direction = SortDirection::None;
                    let file_path = app.file_path.clone();
                    app.query_state
                        .execute_initial_query(&app.session, &file_path);
                    app.query_state.focus = QueryFocus::ResultsTable;
                }
                (InputKeyCode::Left, ..) => app.query_state.move_cursor_left(),
                (InputKeyCode::Right, ..) => app.query_state.move_cursor_right(),
                (InputKeyCode::Home, ..) => app.query_state.move_cursor_start(),
                (InputKeyCode::End, ..) => app.query_state.move_cursor_end(),
                (InputKeyCode::Char('a'), true, ..) => app.query_state.move_cursor_start(),
                (InputKeyCode::Char('e'), true, ..) => app.query_state.move_cursor_end(),
                (InputKeyCode::Char('u'), true, ..) => app.query_state.clear_input(),
                (InputKeyCode::Char('b'), true, ..) => app.query_state.move_cursor_left(),
                (InputKeyCode::Char('f'), true, ..) => app.query_state.move_cursor_right(),
                (InputKeyCode::Char('d'), true, ..) => app.query_state.delete_char_forward(),
                (InputKeyCode::Backspace, ..) => app.query_state.delete_char(),
                (InputKeyCode::Delete, ..) => app.query_state.delete_char_forward(),
                (InputKeyCode::Char(c), false, false, _) => {
                    app.query_state.insert_char(*c);
                }
                _ => {}
            }
            return HandleResult::Continue;
        }
    }

    match (&event.code, event.ctrl, event.alt, event.shift) {
        (InputKeyCode::Char('q'), ..) => {
            return HandleResult::Exit;
        }
        (InputKeyCode::Tab, ..) => {
            app.current_tab = match app.current_tab {
                Tab::Layout => Tab::Segments,
                #[cfg(not(target_arch = "wasm32"))]
                Tab::Segments => Tab::Query,
                #[cfg(not(target_arch = "wasm32"))]
                Tab::Query => Tab::Layout,
                #[cfg(target_arch = "wasm32")]
                Tab::Segments => Tab::Layout,
            };
        }

        #[cfg(not(target_arch = "wasm32"))]
        (InputKeyCode::Char('['), false, false, _) => {
            if app.current_tab == Tab::Query {
                app.query_state
                    .prev_page(&app.session, &app.file_path.clone());
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        (InputKeyCode::Char(']'), false, false, _) => {
            if app.current_tab == Tab::Query {
                app.query_state
                    .next_page(&app.session, &app.file_path.clone());
            }
        }

        (InputKeyCode::Up, ..)
        | (InputKeyCode::Char('k'), false, false, _)
        | (InputKeyCode::Char('p'), true, ..) => match app.current_tab {
            Tab::Layout => navigate_layout_up(app, SCROLL_LINE),
            Tab::Segments => app.segment_grid_state.scroll_up(SEGMENT_SCROLL_LINE),
            #[cfg(not(target_arch = "wasm32"))]
            Tab::Query => {
                app.query_state.table_state.select_previous();
            }
        },
        (InputKeyCode::Down, ..)
        | (InputKeyCode::Char('j'), false, false, _)
        | (InputKeyCode::Char('n'), true, ..) => match app.current_tab {
            Tab::Layout => navigate_layout_down(app, SCROLL_LINE),
            Tab::Segments => app.segment_grid_state.scroll_down(SEGMENT_SCROLL_LINE),
            #[cfg(not(target_arch = "wasm32"))]
            Tab::Query => {
                app.query_state.table_state.select_next();
            }
        },
        (InputKeyCode::PageUp, ..) | (InputKeyCode::Char('v'), _, true, _) => {
            match app.current_tab {
                Tab::Layout => navigate_layout_up(app, SCROLL_PAGE),
                Tab::Segments => app.segment_grid_state.scroll_up(SEGMENT_SCROLL_PAGE),
                #[cfg(not(target_arch = "wasm32"))]
                Tab::Query => {
                    app.query_state
                        .prev_page(&app.session, &app.file_path.clone());
                }
            }
        }
        (InputKeyCode::PageDown, ..) | (InputKeyCode::Char('v'), true, ..) => {
            match app.current_tab {
                Tab::Layout => navigate_layout_down(app, SCROLL_PAGE),
                Tab::Segments => app.segment_grid_state.scroll_down(SEGMENT_SCROLL_PAGE),
                #[cfg(not(target_arch = "wasm32"))]
                Tab::Query => {
                    app.query_state
                        .next_page(&app.session, &app.file_path.clone());
                }
            }
        }
        (InputKeyCode::Home, ..) | (InputKeyCode::Char('<'), _, true, _) => match app.current_tab {
            Tab::Layout => app.layouts_list_state.select_first(),
            Tab::Segments => app
                .segment_grid_state
                .scroll_left(SEGMENT_SCROLL_HORIZONTAL_JUMP),
            #[cfg(not(target_arch = "wasm32"))]
            Tab::Query => {
                app.query_state.table_state.select_first();
            }
        },
        (InputKeyCode::End, ..) | (InputKeyCode::Char('>'), _, true, _) => match app.current_tab {
            Tab::Layout => app.layouts_list_state.select_last(),
            Tab::Segments => app
                .segment_grid_state
                .scroll_right(SEGMENT_SCROLL_HORIZONTAL_JUMP),
            #[cfg(not(target_arch = "wasm32"))]
            Tab::Query => {
                app.query_state.table_state.select_last();
            }
        },
        (InputKeyCode::Enter, ..) => {
            if app.current_tab == Tab::Layout && app.cursor.layout().nchildren() > 0 {
                let selected = app.layouts_list_state.selected().unwrap_or_default();
                app.cursor = app.cursor.child(selected);
                app.reset_layout_view_state();
            }
        }
        (InputKeyCode::Left, ..)
        | (InputKeyCode::Char('h'), false, false, _)
        | (InputKeyCode::Char('b'), true, ..) => match app.current_tab {
            Tab::Layout => {
                app.cursor = app.cursor.parent();
                app.reset_layout_view_state();
            }
            Tab::Segments => app
                .segment_grid_state
                .scroll_left(SEGMENT_SCROLL_HORIZONTAL_STEP),
            #[cfg(not(target_arch = "wasm32"))]
            Tab::Query => {
                app.query_state.horizontal_scroll =
                    app.query_state.horizontal_scroll.saturating_sub(1);
            }
        },
        (InputKeyCode::Right, ..)
        | (InputKeyCode::Char('l'), false, false, _)
        | (InputKeyCode::Char('b'), _, true, _) => match app.current_tab {
            Tab::Layout => {}
            Tab::Segments => app
                .segment_grid_state
                .scroll_right(SEGMENT_SCROLL_HORIZONTAL_STEP),
            #[cfg(not(target_arch = "wasm32"))]
            Tab::Query => {
                let max_col = app.query_state.column_count().saturating_sub(1);
                if app.query_state.horizontal_scroll < max_col {
                    app.query_state.horizontal_scroll += 1;
                }
            }
        },

        (InputKeyCode::Char('/'), ..) | (InputKeyCode::Char('s'), true, ..) => {
            #[cfg(not(target_arch = "wasm32"))]
            if app.current_tab == Tab::Query {
                // Don't enter search mode from query tab
            } else {
                app.key_mode = KeyMode::Search;
            }
            #[cfg(target_arch = "wasm32")]
            {
                app.key_mode = KeyMode::Search;
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        (InputKeyCode::Char('s'), false, false, _) => {
            if app.current_tab == Tab::Query {
                let col = app.query_state.selected_column();
                app.query_state
                    .apply_sort(&app.session, col, &app.file_path);
            }
        }

        #[cfg(not(target_arch = "wasm32"))]
        (InputKeyCode::Esc, ..) => {
            if app.current_tab == Tab::Query {
                app.query_state.toggle_focus();
            }
        }

        _ => {}
    }

    HandleResult::Continue
}

/// Handle a key event in search mode.
pub(crate) fn handle_search_mode(app: &mut AppState, event: InputEvent) -> HandleResult {
    match (&event.code, event.ctrl, event.alt, event.shift) {
        (InputKeyCode::Esc, ..) | (InputKeyCode::Char('g'), true, ..) => {
            app.key_mode = KeyMode::Normal;
            app.clear_search();
        }

        (InputKeyCode::Up, ..) | (InputKeyCode::Char('p'), true, ..) => {
            if app.current_tab == Tab::Layout {
                navigate_layout_up(app, SCROLL_LINE);
            }
        }
        (InputKeyCode::Down, ..) | (InputKeyCode::Char('n'), true, ..) => {
            if app.current_tab == Tab::Layout {
                navigate_layout_down(app, SCROLL_LINE);
            }
        }
        (InputKeyCode::PageUp, ..) | (InputKeyCode::Char('v'), _, true, _) => {
            if app.current_tab == Tab::Layout {
                navigate_layout_up(app, SCROLL_PAGE);
            }
        }
        (InputKeyCode::PageDown, ..) | (InputKeyCode::Char('v'), true, ..) => {
            if app.current_tab == Tab::Layout {
                navigate_layout_down(app, SCROLL_PAGE);
            }
        }
        (InputKeyCode::Home, ..) | (InputKeyCode::Char('<'), _, true, _) => {
            if app.current_tab == Tab::Layout {
                app.layouts_list_state.select_first();
            }
        }
        (InputKeyCode::End, ..) | (InputKeyCode::Char('>'), _, true, _) => {
            if app.current_tab == Tab::Layout {
                app.layouts_list_state.select_last();
            }
        }

        (InputKeyCode::Enter, ..) => {
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

        (InputKeyCode::Backspace, ..) | (InputKeyCode::Char('h'), true, ..) => {
            app.search_filter.pop();
        }

        (InputKeyCode::Char(c), false, false, _) => {
            app.layouts_list_state.select_first();
            app.search_filter.push(*c);
        }

        _ => {}
    }

    HandleResult::Continue
}

// --- Native-only crossterm event loop ---

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use crossterm::event;
    use crossterm::event::Event;
    use crossterm::event::KeyEventKind;
    use ratatui::DefaultTerminal;
    use vortex::error::VortexResult;
    use vortex::session::VortexSession;

    use super::ui::render_app;
    use super::*;

    async fn run(mut terminal: DefaultTerminal, mut app: AppState) -> VortexResult<()> {
        loop {
            terminal.draw(|frame| render_app(&mut app, frame))?;

            let raw_event = event::read()?;
            if let Event::Key(key) = raw_event {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                let input = InputEvent::from(key);
                let result = match app.key_mode {
                    KeyMode::Normal => handle_normal_mode(&mut app, input),
                    KeyMode::Search => handle_search_mode(&mut app, input),
                };

                if matches!(result, HandleResult::Exit) {
                    return Ok(());
                }
            }
        }
    }

    /// Launch the interactive TUI browser for a Vortex file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be opened or if there's a terminal I/O error.
    pub async fn exec_tui(
        session: &VortexSession,
        file: impl AsRef<std::path::Path>,
    ) -> VortexResult<()> {
        let app = AppState::new(session, file).await?;

        let mut terminal = ratatui::init();
        terminal.clear()?;

        run(terminal, app).await?;

        ratatui::restore();
        Ok(())
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::exec_tui;
