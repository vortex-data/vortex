// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_array::RecordBatch;
use ratatui::buffer::Buffer;
use ratatui::layout::Constraint;
use ratatui::layout::Layout;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::style::Style;
use ratatui::text::Line;
use ratatui::text::Span;
use ratatui::widgets::Block;
use ratatui::widgets::BorderType;
use ratatui::widgets::Borders;
use ratatui::widgets::Cell;
use ratatui::widgets::Paragraph;
use ratatui::widgets::Row;
use ratatui::widgets::Scrollbar;
use ratatui::widgets::ScrollbarOrientation;
use ratatui::widgets::ScrollbarState;
use ratatui::widgets::StatefulWidget;
use ratatui::widgets::Table;
use ratatui::widgets::TableState;
use ratatui::widgets::Widget;
use tokio::sync::oneshot;
use vortex::session::VortexSession;

use crate::browse::app::AppState;
use crate::datafusion_helper::arrow_value_to_json;
use crate::datafusion_helper::execute_vortex_query;
use crate::datafusion_helper::json_value_to_display;

/// Sort direction for table columns.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SortDirection {
    /// No sorting applied.
    #[default]
    None,
    /// Sort in ascending order.
    Ascending,
    /// Sort in descending order.
    Descending,
}

impl SortDirection {
    /// Cycle to the next sort direction: None -> Ascending -> Descending -> None.
    pub fn cycle(self) -> Self {
        match self {
            SortDirection::None => SortDirection::Ascending,
            SortDirection::Ascending => SortDirection::Descending,
            SortDirection::Descending => SortDirection::None,
        }
    }

    /// Get the sort direction indicator character for display.
    pub fn indicator(self) -> &'static str {
        match self {
            SortDirection::None => "",
            SortDirection::Ascending => " ▲",
            SortDirection::Descending => " ▼",
        }
    }
}

/// Focus state within the Query tab.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum QueryFocus {
    /// Focus is on the SQL input field.
    #[default]
    SqlInput,
    /// Focus is on the results table.
    ResultsTable,
}

/// Result from a background query task.
pub(crate) struct PendingQueryResult {
    pub row_count: Option<Result<usize, String>>,
    pub query_result: Result<QueryResults, String>,
}

/// State for the SQL query interface.
pub struct QueryState {
    /// The SQL query input text.
    pub sql_input: String,
    /// Cursor position in the SQL input.
    pub cursor_position: usize,
    /// Current focus within the Query tab.
    pub focus: QueryFocus,
    /// Query results as RecordBatches.
    pub results: Option<QueryResults>,
    /// Error message if query failed.
    pub error: Option<String>,
    /// Whether a query is currently running.
    pub running: bool,
    /// Table state for the results view.
    pub table_state: TableState,
    /// Horizontal scroll offset for the results table.
    pub horizontal_scroll: usize,
    /// Column being sorted (if any).
    pub sort_column: Option<usize>,
    /// Sort direction.
    pub sort_direction: SortDirection,
    /// Current page (0-indexed).
    pub current_page: usize,
    /// Rows per page (parsed from LIMIT clause).
    pub page_size: usize,
    /// Total row count from COUNT(*) query.
    pub total_row_count: Option<usize>,
    /// Base SQL query (without LIMIT/OFFSET) for pagination.
    pub base_query: String,
    /// ORDER BY clause if any.
    pub order_clause: Option<String>,
    /// Whether a query execution is pending (needs to be spawned).
    pending_execution: bool,
    /// Whether a row count query is needed on next spawn.
    needs_row_count: bool,
    /// Receiver for in-flight background query result.
    pub(crate) pending_rx: Option<oneshot::Receiver<PendingQueryResult>>,
}

impl Default for QueryState {
    fn default() -> Self {
        let default_sql = "SELECT * FROM data LIMIT 20";
        Self {
            sql_input: default_sql.to_string(),
            cursor_position: default_sql.len(),
            focus: QueryFocus::default(),
            results: None,
            error: None,
            running: false,
            table_state: TableState::default(),
            horizontal_scroll: 0,
            sort_column: None,
            sort_direction: SortDirection::default(),
            current_page: 0,
            page_size: 20,
            total_row_count: None,
            base_query: "SELECT * FROM data".to_string(),
            order_clause: None,
            pending_execution: false,
            needs_row_count: false,
            pending_rx: None,
        }
    }
}

impl QueryState {
    /// Insert a character at the cursor position.
    pub fn insert_char(&mut self, c: char) {
        self.sql_input.insert(self.cursor_position, c);
        self.cursor_position += 1;
    }

    /// Delete the character before the cursor.
    pub fn delete_char(&mut self) {
        if self.cursor_position > 0 {
            self.cursor_position -= 1;
            self.sql_input.remove(self.cursor_position);
        }
    }

    /// Delete the character at the cursor.
    pub fn delete_char_forward(&mut self) {
        if self.cursor_position < self.sql_input.len() {
            self.sql_input.remove(self.cursor_position);
        }
    }

    /// Move cursor left.
    pub fn move_cursor_left(&mut self) {
        self.cursor_position = self.cursor_position.saturating_sub(1);
    }

    /// Move cursor right.
    pub fn move_cursor_right(&mut self) {
        if self.cursor_position < self.sql_input.len() {
            self.cursor_position += 1;
        }
    }

    /// Move cursor to start.
    pub fn move_cursor_start(&mut self) {
        self.cursor_position = 0;
    }

    /// Move cursor to end.
    pub fn move_cursor_end(&mut self) {
        self.cursor_position = self.sql_input.len();
    }

    /// Clear the SQL input.
    pub fn clear_input(&mut self) {
        self.sql_input.clear();
        self.cursor_position = 0;
    }

    /// Toggle focus between SQL input and results table.
    pub fn toggle_focus(&mut self) {
        self.focus = match self.focus {
            QueryFocus::SqlInput => QueryFocus::ResultsTable,
            QueryFocus::ResultsTable => QueryFocus::SqlInput,
        };
    }

    /// Prepare initial query - parses SQL, sets flags for async execution.
    pub fn prepare_initial_query(&mut self) {
        self.error = None;

        // Parse the SQL to extract base query, order clause, and page size
        let (base_sql, order_clause, limit) = self.parse_sql_parts();
        self.base_query = base_sql;
        self.order_clause = order_clause;
        self.page_size = limit.unwrap_or(20);
        self.current_page = 0;

        self.needs_row_count = true;
        self.rebuild_sql();
    }

    /// Prepare navigation to next page.
    pub fn prepare_next_page(&mut self) {
        let total_pages = self.total_pages();
        if self.current_page + 1 < total_pages {
            self.current_page += 1;
            self.rebuild_sql();
        }
    }

    /// Prepare navigation to previous page.
    pub fn prepare_prev_page(&mut self) {
        if self.current_page > 0 {
            self.current_page -= 1;
            self.rebuild_sql();
        }
    }

    /// Get total number of pages.
    pub fn total_pages(&self) -> usize {
        match self.total_row_count {
            Some(total) if total > 0 => total.div_ceil(self.page_size),
            _ => 1,
        }
    }

    /// Build SQL query from current state and set the pending execution flag.
    fn rebuild_sql(&mut self) {
        let offset = self.current_page * self.page_size;

        let new_sql = match &self.order_clause {
            Some(order) => {
                format!(
                    "{} {} LIMIT {} OFFSET {}",
                    self.base_query, order, self.page_size, offset
                )
            }
            None => {
                format!(
                    "{} LIMIT {} OFFSET {}",
                    self.base_query, self.page_size, offset
                )
            }
        };

        self.sql_input = new_sql;
        self.cursor_position = self.sql_input.len();

        self.running = true;
        self.error = None;
        self.pending_execution = true;
    }

    /// Spawn a background task for the pending query, if any.
    ///
    /// After calling `prepare_*` methods, call this to kick off execution.
    /// The result will arrive on [`pending_rx`] and should be applied with
    /// [`apply_query_result`].
    pub(crate) fn spawn_pending(&mut self, session: &VortexSession, file_path: &str) {
        if !self.pending_execution {
            return;
        }
        self.pending_execution = false;

        let (tx, rx) = oneshot::channel();
        let session = session.clone();
        let file_path = file_path.to_string();
        let sql = self.sql_input.clone();
        let base_query = self.base_query.clone();
        let needs_row_count = self.needs_row_count;
        self.needs_row_count = false;

        tokio::spawn(async move {
            let row_count = match needs_row_count {
                true => Some(get_row_count(&session, &file_path, &base_query).await),
                false => None,
            };
            let query_result = execute_query(&session, &file_path, &sql).await;
            drop(tx.send(PendingQueryResult {
                row_count,
                query_result,
            }));
        });

        self.pending_rx = Some(rx);
    }

    /// Apply a completed background query result to the state.
    pub(crate) fn apply_query_result(&mut self, result: PendingQueryResult) {
        if let Some(row_count) = result.row_count {
            self.total_row_count = row_count.ok();
        }
        match result.query_result {
            Ok(results) => {
                self.results = Some(results);
                self.table_state.select(Some(0));
            }
            Err(e) => {
                self.error = Some(e);
            }
        }
        self.running = false;
    }

    /// Parse SQL to extract base query, ORDER BY clause, and LIMIT value.
    fn parse_sql_parts(&self) -> (String, Option<String>, Option<usize>) {
        let sql = &self.sql_input;
        let sql_upper = sql.to_uppercase();

        // Find positions of clauses
        let order_idx = sql_upper.find(" ORDER BY ");
        let limit_idx = sql_upper.find(" LIMIT ");
        let offset_idx = sql_upper.find(" OFFSET ");

        // Extract limit value if present
        let limit_value = if let Some(li) = limit_idx {
            let after_limit = &sql[li + 7..]; // Skip " LIMIT "
            let end_idx = after_limit
                .find(|c: char| !c.is_ascii_digit() && c != ' ')
                .unwrap_or(after_limit.len());
            after_limit[..end_idx].trim().parse::<usize>().ok()
        } else {
            None
        };

        // Find the earliest of LIMIT or OFFSET to know where to cut
        let cut_idx = match (limit_idx, offset_idx) {
            (Some(li), Some(oi)) => Some(li.min(oi)),
            (Some(li), None) => Some(li),
            (None, Some(oi)) => Some(oi),
            (None, None) => None,
        };

        match (order_idx, cut_idx) {
            (Some(oi), Some(ci)) if oi < ci => {
                // ORDER BY comes before LIMIT/OFFSET
                let base = sql[..oi].trim().to_string();
                let order = sql[oi..ci].trim().to_string();
                (base, Some(order), limit_value)
            }
            (Some(oi), None) => {
                // Only ORDER BY, no LIMIT/OFFSET
                let base = sql[..oi].trim().to_string();
                let order = sql[oi..].trim().to_string();
                (base, Some(order), limit_value)
            }
            (None, Some(ci)) => {
                // No ORDER BY, just LIMIT/OFFSET
                let base = sql[..ci].trim().to_string();
                (base, None, limit_value)
            }
            (Some(_oi), Some(ci)) => {
                // ORDER BY comes after LIMIT (unusual) - just cut at LIMIT
                let base = sql[..ci].trim().to_string();
                (base, None, limit_value)
            }
            (None, None) => {
                // No ORDER BY or LIMIT/OFFSET
                (sql.clone(), None, limit_value)
            }
        }
    }

    /// Get the currently selected column index.
    pub fn selected_column(&self) -> usize {
        self.horizontal_scroll
    }

    /// Total number of columns in results.
    pub fn column_count(&self) -> usize {
        self.results
            .as_ref()
            .and_then(|r| r.batches.first())
            .map(|b| b.num_columns())
            .unwrap_or(0)
    }

    /// Prepare sort on a column by modifying the ORDER BY clause and setting execution flag.
    pub fn prepare_sort(&mut self, column: usize) {
        // Get the column name from results
        let column_name = match &self.results {
            Some(results) if column < results.column_names.len() => {
                results.column_names[column].clone()
            }
            _ => return,
        };

        // Cycle sort direction
        if self.sort_column == Some(column) {
            self.sort_direction = self.sort_direction.cycle();
            if self.sort_direction == SortDirection::None {
                self.sort_column = None;
            }
        } else {
            self.sort_column = Some(column);
            self.sort_direction = SortDirection::Ascending;
        }

        // Update the ORDER BY clause
        self.order_clause = if self.sort_direction == SortDirection::None {
            None
        } else {
            let direction = match self.sort_direction {
                SortDirection::Ascending => "ASC",
                SortDirection::Descending => "DESC",
                SortDirection::None => unreachable!(),
            };
            Some(format!("ORDER BY \"{column_name}\" {direction}"))
        };

        // Reset to first page and set pending execution
        self.current_page = 0;
        self.rebuild_sql();
    }
}

/// Holds query results for display.
pub struct QueryResults {
    pub batches: Vec<RecordBatch>,
    pub total_rows: usize,
    pub column_names: Vec<String>,
}

/// Execute a SQL query against the Vortex file.
async fn execute_query(
    session: &VortexSession,
    file_path: &str,
    sql: &str,
) -> Result<QueryResults, String> {
    let batches = execute_vortex_query(session, file_path, sql).await?;

    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

    let column_names = if let Some(batch) = batches.first() {
        let schema = batch.schema();
        schema.fields().iter().map(|f| f.name().clone()).collect()
    } else {
        vec![]
    };

    Ok(QueryResults {
        batches,
        total_rows,
        column_names,
    })
}

/// Get total row count for a base query using COUNT(*).
async fn get_row_count(
    session: &VortexSession,
    file_path: &str,
    base_query: &str,
) -> Result<usize, String> {
    let count_sql = format!("SELECT COUNT(*) as count FROM ({base_query}) AS subquery");

    let batches = execute_vortex_query(session, file_path, &count_sql).await?;

    // Extract count from result
    if let Some(batch) = batches.first()
        && batch.num_rows() > 0
        && batch.num_columns() > 0
    {
        use arrow_array::Int64Array;
        if let Some(arr) = batch.column(0).as_any().downcast_ref::<Int64Array>() {
            #[expect(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            return Ok(arr.value(0) as usize);
        }
    }

    Ok(0)
}

/// Render the Query tab UI.
pub fn render_query(app: &mut AppState, area: Rect, buf: &mut Buffer) {
    let [input_area, results_area] =
        Layout::vertical([Constraint::Length(5), Constraint::Min(10)]).areas(area);

    render_sql_input(app, input_area, buf);
    render_results_table(app, results_area, buf);
}

fn render_sql_input(app: &mut AppState, area: Rect, buf: &mut Buffer) {
    let is_focused = app.query_state.focus == QueryFocus::SqlInput;

    let border_color = if is_focused {
        Color::Cyan
    } else {
        Color::DarkGray
    };

    let block = Block::default()
        .title("SQL Query (Enter to execute, Esc to switch focus)")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    block.render(area, buf);

    // Create the input text with cursor
    let sql = &app.query_state.sql_input;
    let cursor_pos = app.query_state.cursor_position;

    let (before_cursor, after_cursor) = sql.split_at(cursor_pos.min(sql.len()));

    let first_char = after_cursor.chars().next();
    let cursor_char = if is_focused {
        match first_char {
            None => Span::styled(" ", Style::default().bg(Color::White).fg(Color::Black)),
            Some(c) => Span::styled(
                c.to_string(),
                Style::default().bg(Color::White).fg(Color::Black),
            ),
        }
    } else {
        match first_char {
            None => Span::raw(""),
            Some(c) => Span::raw(c.to_string()),
        }
    };

    let rest = match first_char {
        Some(c) if after_cursor.len() > c.len_utf8() => &after_cursor[c.len_utf8()..],
        _ => "",
    };

    let line = Line::from(vec![Span::raw(before_cursor), cursor_char, Span::raw(rest)]);

    let paragraph = Paragraph::new(line).style(Style::default().fg(Color::White));

    paragraph.render(inner, buf);
}

fn render_results_table(app: &mut AppState, area: Rect, buf: &mut Buffer) {
    let is_focused = app.query_state.focus == QueryFocus::ResultsTable;

    let border_color = if is_focused {
        Color::Cyan
    } else {
        Color::DarkGray
    };

    // Show status in title
    let title = if app.query_state.running {
        "Results (running...)".to_string()
    } else if let Some(ref error) = app.query_state.error {
        format!("Results (error: {})", truncate_str(error, 50))
    } else if let Some(ref _results) = app.query_state.results {
        let total_rows = app.query_state.total_row_count.unwrap_or(0);
        let total_pages = app.query_state.total_pages();
        format!(
            "Results ({} rows, page {}/{}) [hjkl navigate, [/] pages, s sort]",
            total_rows,
            app.query_state.current_page + 1,
            total_pages,
        )
    } else {
        "Results (press Enter to execute query)".to_string()
    };

    let block = Block::default()
        .title(title)
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(border_color));

    let inner = block.inner(area);
    block.render(area, buf);

    if let Some(ref error) = app.query_state.error {
        let error_text = Paragraph::new(error.as_str())
            .style(Style::default().fg(Color::Red))
            .wrap(ratatui::widgets::Wrap { trim: true });
        error_text.render(inner, buf);
        return;
    }

    let Some(ref results) = app.query_state.results else {
        let help = Paragraph::new("Enter a SQL query above and press Enter to execute.\nThe table is available as 'data'.\n\nExample: SELECT * FROM data WHERE column > 10 LIMIT 100")
            .style(Style::default().fg(Color::Gray));
        help.render(inner, buf);
        return;
    };

    if results.batches.is_empty() || results.total_rows == 0 {
        let empty =
            Paragraph::new("Query returned no results.").style(Style::default().fg(Color::Yellow));
        empty.render(inner, buf);
        return;
    }

    // Build header row with sort indicators
    let header_cells: Vec<Cell> = results
        .column_names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let indicator = if app.query_state.sort_column == Some(i) {
                app.query_state.sort_direction.indicator()
            } else {
                ""
            };

            let style = if is_focused && i == app.query_state.horizontal_scroll {
                Style::default().fg(Color::Black).bg(Color::Cyan).bold()
            } else {
                Style::default().fg(Color::Green).bold()
            };

            Cell::from(format!("{name}{indicator}")).style(style)
        })
        .collect();

    let header = Row::new(header_cells).height(1);

    // Since we use LIMIT/OFFSET in SQL, batches contain only the current page's data
    // Display all rows from the batches
    let rows = get_all_rows(results, &app.query_state);

    // Calculate column widths
    #[expect(clippy::cast_possible_truncation)]
    let widths: Vec<Constraint> = results
        .column_names
        .iter()
        .map(|name| Constraint::Min((name.len() + 3).max(10) as u16))
        .collect();

    let table = Table::new(rows, widths)
        .header(header)
        .row_highlight_style(Style::default().bg(Color::DarkGray));

    // Split area for table and scrollbar
    let [table_area, scrollbar_area] =
        Layout::horizontal([Constraint::Min(0), Constraint::Length(1)]).areas(inner);

    StatefulWidget::render(table, table_area, buf, &mut app.query_state.table_state);

    // Render vertical scrollbar
    let total_pages = app.query_state.total_pages();
    if total_pages > 1 {
        let mut scrollbar_state = ScrollbarState::new(total_pages)
            .position(app.query_state.current_page)
            .viewport_content_length(1);

        Scrollbar::new(ScrollbarOrientation::VerticalRight)
            .begin_symbol(Some("▲"))
            .end_symbol(Some("▼"))
            .render(scrollbar_area, buf, &mut scrollbar_state);
    }
}

/// Get all rows from batches (pagination is handled via SQL LIMIT/OFFSET).
fn get_all_rows<'a>(results: &'a QueryResults, query_state: &QueryState) -> Vec<Row<'a>> {
    let mut rows = Vec::new();

    for batch in &results.batches {
        for row_idx in 0..batch.num_rows() {
            let cells: Vec<Cell> = (0..batch.num_columns())
                .map(|col_idx| {
                    let json_value = arrow_value_to_json(batch.column(col_idx).as_ref(), row_idx);
                    let value = json_value_to_display(json_value);
                    let style = if query_state.sort_column == Some(col_idx) {
                        Style::default().fg(Color::Cyan)
                    } else {
                        Style::default()
                    };
                    Cell::from(truncate_str(&value, 30).to_string()).style(style)
                })
                .collect();
            rows.push(Row::new(cells));
        }
    }

    rows
}

fn truncate_str(s: &str, max_len: usize) -> &str {
    if s.len() <= max_len {
        s
    } else {
        &s[..max_len.saturating_sub(3)]
    }
}
