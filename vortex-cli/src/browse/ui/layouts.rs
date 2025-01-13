use ratatui::buffer::Buffer;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::Text;
use ratatui::widgets::{
    Block, BorderType, Borders, List, ListState, Paragraph, StatefulWidget, Widget,
};
use vortex::buffer::ByteBuffer;
use vortex::dtype::Field;
use vortex::error::VortexExpect;
use vortex::file::{CHUNKED_LAYOUT_ID, COLUMNAR_LAYOUT_ID, FLAT_LAYOUT_ID};
use vortex::flatbuffers::array::root_as_array;
use vortex::flatbuffers::FlatBuffer;
use vortex::stats::stats_from_bitset_bytes;
use vortex_layout::segments::SegmentId;

use crate::browse::app::{AppState, LayoutCursor};

/// Render the Layouts tab.
pub fn render_layouts(app_state: &mut AppState, area: Rect, buf: &mut Buffer) {
    let [header_area, detail_area] =
        Layout::vertical([Constraint::Length(10), Constraint::Min(1)]).areas(area);

    // Render the header area.
    render_layout_header(&app_state.cursor, header_area, buf);

    // Render the list view if the layout has children
    if app_state.cursor.encoding().id() == FLAT_LAYOUT_ID {
        render_array(app_state, detail_area, buf);
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
    // We want the header to have some padding, and all elements to be horizontally aligned.
    // let [area] = Layout::default()
    //     .constraints([Constraint::Min(0)])
    //     .margin(10)
    //     .areas(area);

    let layout_kind = match cursor.encoding().id() {
        FLAT_LAYOUT_ID => "FLAT".to_string(),
        CHUNKED_LAYOUT_ID => "CHUNKED".to_string(),
        COLUMNAR_LAYOUT_ID => "COLUMNAR".to_string(),
        _ => "UNKNOWN".to_string(),
    };

    // If using a FlatLayout, read the array and parse the metadata.

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
fn render_array(app: &AppState, area: Rect, buf: &mut Buffer) {
    let segment_ids: Vec<SegmentId> = app.cursor.layout().segments().collect();
    let buffers: Vec<ByteBuffer> = segment_ids
        .into_iter()
        .map(|id| app.read_segment(id))
        .collect();

    let array_fmt = read_array(
        buffers,
        // app.cursor.layout(),
        // ALL_ENCODINGS_CONTEXT.clone(),
        // app.cursor.layout().dtype(),
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

    Paragraph::new(array_fmt).render(widget_area, buf);
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
            } else if nth == 0 {
                "Chunk Statistics".to_string()
            } else {
                format!("Chunk {}", nth - 1)
            }
        }
        FLAT_LAYOUT_ID => format!("Page {nth}"),
        _ => format!("Unknown {nth}"),
    }
}

fn read_array(
    mut buffers: Vec<ByteBuffer>,
    // layout: &LayoutData,
    // ctx: Arc<Context>,
    // dtype: &DType,
) -> String {
    #[allow(clippy::unwrap_used)]
    let flatbuffer = FlatBuffer::try_from(buffers.pop().unwrap()).unwrap();

    let fb_array =
        root_as_array(flatbuffer.as_ref()).vortex_expect("Invalid fba::Array flatbuffer");
    format!("{fb_array:?}")

    // TODO(aduffy): for some reason this fails with schema unexpected error.
    // let mut log = std::fs::File::create("fb.log").unwrap();
    // write!(log, "flatbuffer: {fb_array:?}").unwrap();
    // let row_count = usize::try_from(layout.row_count())
    //     .vortex_expect("FlatLayout row count does not fit within usize");

    // let array_parts = ArrayParts::new(
    //     row_count,
    //     root_as_array(flatbuffer.as_ref()).vortex_expect("Invalid fba::Array flatbuffer"),
    //     flatbuffer.clone(),
    //     buffers,
    // );

    // // Decode into an ArrayData.
    // array_parts.decode(ctx, dtype.clone()).unwrap()
}
