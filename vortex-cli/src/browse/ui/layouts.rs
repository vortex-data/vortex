use std::sync::Arc;

use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::Text;
use ratatui::widgets::{
    Block, BorderType, Borders, Cell, List, ListState, Paragraph, Row, StatefulWidget, Table,
    Widget,
};
use vortex::buffer::ByteBuffer;
use vortex::compute::scalar_at;
use vortex::dtype::{DType, Field};
use vortex::error::VortexExpect;
use vortex::file::{CHUNKED_LAYOUT_ID, COLUMNAR_LAYOUT_ID, FLAT_LAYOUT_ID};
use vortex::flatbuffers::array::root_as_array;
use vortex::flatbuffers::FlatBuffer;
use vortex::parts::ArrayParts;
use vortex::sampling_compressor::ALL_ENCODINGS_CONTEXT;
use vortex::stats::stats_from_bitset_bytes;
use vortex::{ArrayData, Context};
use vortex_layout::segments::SegmentId;
use vortex_layout::LayoutData;

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
        render_children_list(
            &app_state.cursor,
            detail_area,
            buf,
            &mut app_state.layouts_list_state,
        );
    }
}

fn render_layout_header(cursor: &LayoutCursor, area: Rect, buf: &mut Buffer) {
    let layout_kind = match cursor.encoding().id() {
        FLAT_LAYOUT_ID => "FLAT".to_string(),
        CHUNKED_LAYOUT_ID => "CHUNKED".to_string(),
        COLUMNAR_LAYOUT_ID => "COLUMNAR".to_string(),
        _ => "UNKNOWN".to_string(),
    };

    let row_count = cursor.layout().row_count();

    let mut rows = vec![
        Text::from(format!("Kind: {layout_kind}")).bold(),
        Text::from(format!("Row Count: {row_count}")).bold(),
        Text::from(format!("Schema: {}", cursor.dtype()))
            .bold()
            .green(),
    ];

    if cursor.encoding().id() == CHUNKED_LAYOUT_ID {
        // Push any columnar stats.
        if let Some(metadata) = cursor.layout().metadata() {
            let available_stats = stats_from_bitset_bytes(&metadata);
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
    let segment_ids: Vec<SegmentId> = app.cursor.layout().segments().collect();
    let buffers: Vec<ByteBuffer> = segment_ids
        .into_iter()
        .map(|id| app.read_segment(id))
        .collect();

    let array = read_array(
        buffers,
        app.cursor.layout(),
        ALL_ENCODINGS_CONTEXT.clone(),
        app.cursor.layout().dtype(),
    );

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
        let struct_array = array.as_struct_array().vortex_expect("stats table");
        let field_count = struct_array.nfields();
        let header = std::iter::once("chunk")
            .chain(struct_array.names().iter().map(|x| x.as_ref()))
            .map(Cell::from)
            .collect::<Row>()
            .style(Style::default().fg(Color::Green).bg(Color::DarkGray))
            .height(1);

        let field_arrays: Vec<ArrayData> = (0..struct_array.nfields())
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
            Table::new(rows, (0..field_count).map(|_| Constraint::Length(6))).header(header),
            widget_area,
            buf,
        );
    } else {
        // Tree-display the active array
        Paragraph::new(array.tree_display().to_string()).render(widget_area, buf);
    };
}

fn render_children_list(
    cursor: &LayoutCursor,
    area: Rect,
    buf: &mut Buffer,
    state: &mut ListState,
) {
    // TODO: add selection state.
    let layout = cursor.layout();

    if layout.nchildren() > 0 {
        let list_items: Vec<String> = (0..layout.nchildren())
            .map(|idx| child_name(cursor, idx))
            .collect();

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
            state,
        );
    }
}

fn child_name(cursor: &LayoutCursor, nth: usize) -> String {
    match cursor.encoding().id() {
        COLUMNAR_LAYOUT_ID => {
            let field_info = cursor
                .dtype()
                .as_struct()
                .expect("struct dtype")
                .field_info(&Field::Index(nth))
                .expect("struct dtype component");
            let field_name = field_info.name;
            let field_dtype = field_info.dtype.value().expect("dtype value");
            format!("Column {nth} - {field_name} ({field_dtype})")
        }
        CHUNKED_LAYOUT_ID => {
            // 0th child of a ChunkedLayout is the chunk stats array.
            // The rest of the chunks are child arrays
            if cursor.layout().metadata().is_none() {
                format!("Chunk {nth}")
            } else if nth == (cursor.layout().nchildren() - 1) {
                "Chunk Statistics".to_string()
            } else {
                format!("Chunk {}", nth)
            }
        }
        FLAT_LAYOUT_ID => format!("Page {nth}"),
        _ => format!("Unknown {nth}"),
    }
}

fn read_array(
    mut buffers: Vec<ByteBuffer>,
    layout: &LayoutData,
    ctx: Arc<Context>,
    dtype: &DType,
) -> ArrayData {
    let flatbuffer = FlatBuffer::try_from(buffers.pop().vortex_expect("buffers pop"))
        .vortex_expect("flatbuffers");

    let fb_array =
        root_as_array(flatbuffer.as_ref()).vortex_expect("Invalid fba::Array flatbuffer");

    let array_parts = ArrayParts::new(
        layout.row_count().try_into().vortex_expect("row_count"),
        fb_array,
        flatbuffer.clone(),
        buffers,
    );

    // // Decode into an ArrayData.
    array_parts
        .decode(ctx, dtype.clone())
        .vortex_expect("decoding ArrayParts")
}
