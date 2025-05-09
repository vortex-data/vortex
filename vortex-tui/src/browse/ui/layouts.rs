use humansize::{DECIMAL, make_format};
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::Text;
use ratatui::widgets::{
    Block, BorderType, Borders, Cell, List, Paragraph, Row, StatefulWidget, Table, Widget, Wrap,
};
use vortex::error::VortexExpect;
use vortex::expr::Identity;
use vortex::layout::{
    CHUNKED_LAYOUT_ID, DICT_LAYOUT_ID, FLAT_LAYOUT_ID, STATS_LAYOUT_ID, STRUCT_LAYOUT_ID,
};
use vortex::mask::Mask;
use vortex::stats::stats_from_bitset_bytes;
use vortex::{Array, ArrayRef, ToCanonical};
use vortex_layout::layouts::stats::StatsLayout;
use vortex_layout::{ExprEvaluator, LayoutVTable};

use crate::TOKIO_RUNTIME;
use crate::browse::app::{AppState, LayoutCursor};

/// Render the Layouts tab.
pub fn render_layouts(app_state: &mut AppState, area: Rect, buf: &mut Buffer) {
    let [header_area, detail_area] =
        Layout::vertical([Constraint::Length(10), Constraint::Min(1)]).areas(area);

    // Render the header area.
    render_layout_header(&app_state.cursor, header_area, buf);

    // Render the list view if the layout has children
    if app_state.cursor.encoding().id() == FLAT_LAYOUT_ID {
        render_array(
            app_state,
            detail_area,
            buf,
            app_state.cursor.is_stats_table(),
        );
    } else {
        render_children_list(app_state, detail_area, buf);
    }
}

fn render_layout_header(cursor: &LayoutCursor, area: Rect, buf: &mut Buffer) {
    let layout_kind = cursor.layout().id().to_string();
    let row_count = cursor.layout().row_count();
    let size_formatter = make_format(DECIMAL);
    let size = size_formatter(cursor.total_size());

    let mut rows = vec![
        Text::from(format!("Kind: {layout_kind}")).bold(),
        Text::from(format!("Row Count: {row_count}")).bold(),
        Text::from(format!("Schema: {}", cursor.dtype()))
            .bold()
            .green(),
        Text::from(format!("Children: {}", cursor.layout().nchildren())).bold(),
        Text::from(format!("Segment data size: {}", size)).bold(),
    ];

    if cursor.encoding().id() == FLAT_LAYOUT_ID {
        rows.push(Text::from(format!(
            "FlatBuffer Size: {}",
            size_formatter(cursor.flatbuffer_size())
        )));
    }

    if cursor.encoding().id() == StatsLayout.id() {
        // Push any columnar stats.
        if let Some(available_stats) = cursor
            .layout()
            .metadata()
            .map(|metadata| stats_from_bitset_bytes(&metadata[4..]))
        {
            let mut line = String::new();
            line.push_str("Statistics: ");
            for stat in available_stats {
                line.push_str(stat.to_string().as_str());
                line.push(' ');
            }

            rows.push(Text::from(line));
        } else {
            rows.push(Text::from("No chunk statistics found"));
        }
    }

    let container = Block::new()
        .title("Layout Info")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::DarkGray));

    let inner_area = container.inner(area);

    container.render(area, buf);

    Widget::render(List::new(rows), inner_area, buf);
}

// Render the inner Array for a FlatLayout
fn render_array(app: &AppState, area: Rect, buf: &mut Buffer, is_stats_table: bool) {
    let reader = app
        .cursor
        .layout()
        .reader(&app.vxf.segment_source(), app.vxf.footer().ctx())
        .vortex_expect("Failed to create reader");

    let array = TOKIO_RUNTIME
        .block_on(
            reader
                .projection_evaluation(&(0..reader.row_count()), &Identity::new_expr())
                .vortex_expect("Failed to construct projection")
                .invoke(Mask::new_true(
                    reader.row_count().try_into().vortex_expect("row count"),
                )),
        )
        .vortex_expect("Failed to read flat array");

    // Show the metadata as JSON. (show count of encoded bytes as well)
    // let metadata_size = array.metadata_bytes().unwrap_or_default().len();
    let container = Block::new()
        .title("Array Info")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::DarkGray));

    let widget_area = container.inner(area);

    container.render(area, buf);

    if is_stats_table {
        // Render the stats table horizontally
        let struct_array = array.to_struct().vortex_expect("stats table");
        // add 1 for the chunk column
        let field_count = struct_array.struct_dtype().nfields() + 1;
        let header = std::iter::once("chunk")
            .chain(struct_array.names().iter().map(|x| x.as_ref()))
            .map(Cell::from)
            .collect::<Row>()
            .style(Style::default().fg(Color::Green).bg(Color::DarkGray))
            .height(1);

        assert_eq!(app.cursor.dtype(), array.dtype());

        let field_arrays: Vec<ArrayRef> = struct_array.fields().to_vec();

        // TODO: trim the number of displayed rows and allow paging through column stats.
        let rows = (0..array.len()).map(|chunk_id| {
            std::iter::once(Cell::from(Text::from(format!("{chunk_id}"))))
                .chain(field_arrays.iter().map(|arr| {
                    Cell::from(Text::from(
                        arr.scalar_at(chunk_id)
                            .vortex_expect("stats table scalar_at")
                            .to_string(),
                    ))
                }))
                .collect::<Row>()
        });

        Widget::render(
            Table::new(rows, (0..field_count).map(|_| Constraint::Min(6))).header(header),
            widget_area,
            buf,
        );
    } else {
        let header = ["Name", "Value"]
            .into_iter()
            .map(Cell::from)
            .collect::<Row>()
            .style(Style::new().bold())
            .height(1);

        let rows = array.statistics().into_iter().map(|(stat, value)| {
            let value = value.into_scalar(
                stat.dtype(array.dtype())
                    .vortex_expect("stat invalid for dtype"),
            );
            let stat = Cell::from(Text::from(format!("{stat}")));
            let value = Cell::from(Text::from(format!("{value}")));
            Row::new(vec![stat, value])
        });

        let layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(vec![Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(widget_area);
        let table = Table::new(rows, [Constraint::Min(6), Constraint::Min(6)]).header(header);
        // Tree-display the active array
        let tree = Paragraph::new(array.tree_display().to_string()).wrap(Wrap { trim: false });

        let stats_container = Block::new()
            .title("Statistics")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::DarkGray));

        let tree_container = Block::new()
            .title("Encoding Tree Display")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::DarkGray));

        let tree_inner = tree_container.inner(layout[0]);
        let stats_inner = stats_container.inner(layout[1]);

        tree_container.render(layout[0], buf);
        stats_container.render(layout[1], buf);

        Widget::render(tree, tree_inner, buf);
        Widget::render(table, stats_inner, buf);

        // Split view, show information about the child arrays (metadata, count, etc.)
    };
}

fn render_children_list(app: &mut AppState, area: Rect, buf: &mut Buffer) {
    // TODO: add selection state.
    let search_filter = app.search_filter.clone();

    if app.cursor.layout().nchildren() > 0 {
        let filter: Vec<bool> = (0..app.cursor.layout().nchildren())
            .map(|idx| child_name(app, idx))
            .map(|name| {
                if search_filter.is_empty() {
                    true
                } else {
                    name.contains(&search_filter)
                }
            })
            .collect();

        let list_items: Vec<String> = (0..app.cursor.layout().nchildren())
            .zip(filter.iter())
            .filter(|&(_, keep)| *keep)
            .map(|(idx, _)| child_name(app, idx))
            .collect();

        if !app.search_filter.is_empty() {
            app.filter = Some(filter);
        }

        let container = Block::new()
            .title("Child Layouts")
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(Color::DarkGray));

        let inner_area = container.inner(area);

        container.render(area, buf);

        // Render the List view.
        // TODO: add state so we can scroll
        StatefulWidget::render(
            List::new(list_items).highlight_style(Style::default().black().on_white().bold()),
            inner_area,
            buf,
            &mut app.layouts_list_state,
        );
    }
}

fn child_name(app: &mut AppState, nth: usize) -> String {
    let cursor = &app.cursor;
    let formatter = make_format(DECIMAL);
    // TODO(ngates): layout visitors
    if cursor.layout().id() == STRUCT_LAYOUT_ID {
        let struct_dtype = cursor.dtype().as_struct().expect("struct dtype");
        let field_name = struct_dtype.field_name(nth).expect("field name");
        let field_dtype = struct_dtype.field_by_index(nth).expect("dtype value");

        let total_size = formatter(app.cursor.child(nth).total_size());

        format!("Column {nth} - {field_name} ({field_dtype}) - {total_size}")
    } else if cursor.layout().id() == CHUNKED_LAYOUT_ID {
        let name = format!("Chunk {nth}");
        let child_cursor = app.cursor.child(nth);

        let total_size = formatter(child_cursor.total_size());
        let row_count = child_cursor.layout().row_count();

        format!("{name} - {row_count} - {total_size}")
    } else if cursor.layout().id() == FLAT_LAYOUT_ID {
        format!("Page {nth}")
    } else if cursor.layout().id() == STATS_LAYOUT_ID {
        // 0th child is the data, 1st child is stats.
        if nth == 0 {
            "Data".to_string()
        } else if nth == 1 {
            "Stats".to_string()
        } else {
            format!("Unknown {nth}")
        }
    } else if cursor.layout().id() == DICT_LAYOUT_ID {
        match nth {
            0 => "Values".to_string(),
            1 => "Codes".to_string(),
            _ => format!("unknown {nth}"),
        }
    } else {
        format!("Unknown {nth}")
    }
}
