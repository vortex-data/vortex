use humansize::{DECIMAL, format_size, make_format};
use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::Text;
use ratatui::widgets::{
    Block, BorderType, Borders, Cell, List, Paragraph, Row, StatefulWidget, Table, Widget,
};
use vortex::ArrayRef;
use vortex::compute::scalar_at;
use vortex::error::{VortexExpect, VortexResult, VortexUnwrap, vortex_bail};
use vortex::expr::Identity;
use vortex::file::SegmentSpec;
use vortex::layout::{CHUNKED_LAYOUT_ID, FLAT_LAYOUT_ID, STATS_LAYOUT_ID, STRUCT_LAYOUT_ID};
use vortex::stats::stats_from_bitset_bytes;
use vortex_layout::layouts::chunked::ChunkedLayout;
use vortex_layout::layouts::flat::FlatLayout;
use vortex_layout::layouts::stats::StatsLayout;
use vortex_layout::layouts::struct_::StructLayout;
use vortex_layout::segments::SegmentId;
use vortex_layout::{ExprEvaluator, LayoutReaderExt, LayoutVTable, RowMask};

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

    let segments = collect_segment_ids(cursor.layout());
    let size =
        total_size(cursor.segment_map(), segments.0) + total_size(cursor.segment_map(), segments.1);
    let size = format_size(size, DECIMAL);

    let mut rows = vec![
        Text::from(format!("Kind: {layout_kind}")).bold(),
        Text::from(format!("Row Count: {row_count}")).bold(),
        Text::from(format!("Schema: {}", cursor.dtype()))
            .bold()
            .green(),
        Text::from(format!("Children: {}", cursor.layout().nchildren())).bold(),
        // Text::from(format!("Segments: {}", cursor.layout().nsegments())),
        Text::from(format!("Segment data size: {}", size)).bold(),
    ];

    if cursor.encoding().id() == FLAT_LAYOUT_ID {
        rows.push(Text::from(format!(
            "FlatBuffer Size: {} bytes",
            cursor.flatbuffer_size()
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
        .reader(app.reader.clone(), app.footer.ctx().clone())
        .vortex_expect("Failed to create reader");

    let array = TOKIO_RUNTIME
        .block_on(reader.evaluate_expr(
            RowMask::new_valid_between(0, reader.row_count()),
            Identity::new_expr(),
        ))
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
        let struct_array = array.as_struct_typed().vortex_expect("stats table");
        // add 1 for the chunk column
        let field_count = struct_array.nfields() + 1;
        let header = std::iter::once("chunk")
            .chain(struct_array.names().iter().map(|x| x.as_ref()))
            .map(Cell::from)
            .collect::<Row>()
            .style(Style::default().fg(Color::Green).bg(Color::DarkGray))
            .height(1);

        assert_eq!(app.cursor.dtype(), array.dtype());

        let field_arrays: Vec<ArrayRef> = (0..struct_array.nfields())
            .map(|x| {
                struct_array
                    .maybe_null_field_by_idx(x)
                    .vortex_expect("stats table field")
            })
            .collect();

        // TODO: trim the number of displayed rows and allow paging through column stats.
        let rows = (0..array.len()).map(|chunk_id| {
            std::iter::once(Cell::from(Text::from(format!("{chunk_id}"))))
                .chain(field_arrays.iter().map(|arr| {
                    Cell::from(Text::from(
                        scalar_at(arr, chunk_id)
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
        // Tree-display the active array
        Paragraph::new(array.tree_display().to_string()).render(widget_area, buf);
        // Split view, show information about the child arrays (metadata, count, etc.)
    };
}

fn render_children_list(app: &mut AppState, area: Rect, buf: &mut Buffer) {
    // let cursor = &;
    // TODO: add selection state.
    // let layout = app.cursor.layout();
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

fn struct_child_layout_size(
    nth: usize,
    layout: &vortex_layout::Layout,
    segment_map: &[SegmentSpec],
) -> (usize, usize) {
    let struct_dtype = layout.dtype().as_struct().expect("struct dtype");
    let field_name = struct_dtype.field_name(nth).expect("field name");
    let field_dtype = struct_dtype.field_by_index(nth).expect("dtype value");

    let child_layout = layout
        .child(nth, field_dtype.clone(), field_name)
        .vortex_unwrap();

    let (data_segment_ids, stats_segment_ids) = collect_segment_ids(&child_layout);

    let data_size = total_size(segment_map, data_segment_ids);
    let stats_size = total_size(segment_map, stats_segment_ids);

    (data_size, stats_size)
}

fn child_name(app: &mut AppState, nth: usize) -> String {
    let cursor = &app.cursor;
    let formatter = make_format(DECIMAL);
    // TODO(ngates): layout visitors
    if cursor.layout().id() == STRUCT_LAYOUT_ID {
        let struct_dtype = cursor.dtype().as_struct().expect("struct dtype");
        let field_name = struct_dtype.field_name(nth).expect("field name");
        let field_dtype = struct_dtype.field_by_index(nth).expect("dtype value");

        let (data_size, stats_size) =
            struct_child_layout_size(nth, cursor.layout(), cursor.segment_map());

        let total_size = data_size + stats_size;

        let data_size = formatter(data_size);
        let stats_size = formatter(stats_size);
        let total_size = formatter(total_size);

        format!(
            "Column {nth} - {field_name} ({field_dtype}) - {data_size} + {stats_size} = {total_size}"
        )
    } else if cursor.layout().id() == CHUNKED_LAYOUT_ID {
        let name = format!("Chunk {nth}");
        let child_layout = app
            .cursor
            .layout()
            .child(nth, app.cursor.dtype().clone(), name.clone())
            .vortex_unwrap();

        let (data_segment_ids, stats_segment_ids) = collect_segment_ids(&child_layout);

        let data_size = total_size(app.cursor.segment_map(), data_segment_ids);
        let stats_size = total_size(app.cursor.segment_map(), stats_segment_ids);
        let total_size = formatter(data_size + stats_size);

        let row_count = child_layout.row_count();

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
    } else {
        format!("Unknown {nth}")
    }
}

fn collect_segment_ids(root_layout: &vortex_layout::Layout) -> (Vec<SegmentId>, Vec<SegmentId>) {
    let mut data_segment_ids = Vec::default();
    let mut stats_segment_ids = Some(Vec::default());

    collect_segment_ids_impl(root_layout, &mut data_segment_ids, &mut stats_segment_ids)
        .vortex_unwrap();

    (data_segment_ids, stats_segment_ids.unwrap())
}

fn total_size(segment_map: &[SegmentSpec], segment_ids: Vec<SegmentId>) -> usize {
    segment_ids
        .iter()
        .map(|seg_id| segment_map[**seg_id as usize].length as usize)
        .sum::<usize>()
}

fn collect_segment_ids_impl(
    root: &vortex_layout::Layout,
    data_segments: &mut Vec<SegmentId>,
    stats_segments: &mut Option<Vec<SegmentId>>,
) -> VortexResult<()> {
    let layout_id = root.id();

    if layout_id == StructLayout.id() {
        let dtype = root.dtype().as_struct().vortex_expect("");
        for child_idx in 0..dtype.fields().len() {
            let name = dtype.field_name(child_idx)?;
            let child_dtype = dtype.field_by_index(child_idx)?;
            let child_layout = root.child(child_idx, child_dtype, name)?;
            collect_segment_ids_impl(&child_layout, data_segments, stats_segments)?;
        }
    } else if layout_id == ChunkedLayout.id() {
        for child_idx in 0..root.nchildren() {
            let child_layout =
                root.child(child_idx, root.dtype().clone(), format!("[{child_idx}]"))?;
            collect_segment_ids_impl(&child_layout, data_segments, stats_segments)?;
        }
    } else if layout_id == StatsLayout.id() {
        let data_layout = root.child(0, root.dtype().clone(), format!("data"))?;
        collect_segment_ids_impl(&data_layout, data_segments, stats_segments)?;

        if let Some(stats_segments) = stats_segments.as_mut() {
            let stats_layout = root.child(1, root.dtype().clone(), format!("stats"))?;
            collect_segment_ids_impl(&stats_layout, stats_segments, &mut None)?;
        }
    } else if layout_id == FlatLayout.id() {
        data_segments.extend(root.segments());
    } else {
        vortex_bail!("IDK what I'm doing with my life")
    };

    Ok(())
}
