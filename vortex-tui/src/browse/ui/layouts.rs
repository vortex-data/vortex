// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fuzzy_matcher::FuzzyMatcher;
use fuzzy_matcher::skim::SkimMatcherV2;
use humansize::DECIMAL;
use humansize::make_format;
use itertools::Itertools;
use ratatui::buffer::Buffer;
use ratatui::layout::Constraint;
use ratatui::layout::Direction;
use ratatui::layout::Layout;
use ratatui::layout::Rect;
use ratatui::style::Color;
use ratatui::style::Style;
use ratatui::style::Stylize;
use ratatui::text::Text;
use ratatui::widgets::Block;
use ratatui::widgets::BorderType;
use ratatui::widgets::Borders;
use ratatui::widgets::Cell;
use ratatui::widgets::List;
use ratatui::widgets::Paragraph;
use ratatui::widgets::Row;
use ratatui::widgets::StatefulWidget;
use ratatui::widgets::Table;
use ratatui::widgets::Widget;
use ratatui::widgets::Wrap;
use vortex::array::ArrayRef;
use vortex::array::LEGACY_SESSION;
#[expect(deprecated)]
use vortex::array::ToCanonical;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::struct_::StructArrayExt;
use vortex::error::VortexExpect;
use vortex::layout::layouts::flat::Flat;
use vortex::layout::layouts::zoned::Zoned;

use crate::browse::app::AppState;

/// Render the Layouts tab.
pub fn render_layouts(app_state: &mut AppState, area: Rect, buf: &mut Buffer) {
    let [header_area, detail_area] =
        Layout::vertical([Constraint::Length(10), Constraint::Min(1)]).areas(area);

    // Render the header area.
    render_layout_header(app_state, header_area, buf);

    // Render the list view if the layout has children
    if app_state.cursor.layout().is::<Flat>() {
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

fn render_layout_header(app: &AppState, area: Rect, buf: &mut Buffer) {
    let cursor = &app.cursor;
    let layout_id = cursor.layout().encoding_id();
    let row_count = cursor.layout().row_count();
    let size_formatter = make_format(DECIMAL);
    let size = size_formatter(cursor.total_size());

    let mut rows = vec![
        Text::from(format!("Kind: {layout_id}")).bold(),
        Text::from(format!("Row Count: {row_count}")).bold(),
        Text::from(format!("Schema: {}", cursor.dtype()))
            .bold()
            .green(),
        Text::from(format!("Children: {}", cursor.layout().nchildren())).bold(),
        Text::from(format!("Segment data size: {size}")).bold(),
    ];

    if cursor.layout().is::<Flat>() {
        if let Some(fb_size) = app.cached_flatbuffer_size {
            rows.push(Text::from(format!(
                "FlatBuffer Size: {}",
                size_formatter(fb_size)
            )));
        }

        // Display metadata info about the flat layout
        let metadata_info = cursor.flat_layout_metadata_info();
        rows.push(Text::from(metadata_info));
    }

    if let Some(layout) = cursor.layout().as_opt::<Zoned>() {
        // Push any zone stats.
        let mut line = String::new();
        line.push_str("Statistics: ");
        for stat in layout.present_stats().as_ref() {
            line.push_str(stat.to_string().as_str());
            line.push(' ');
        }

        rows.push(Text::from(line));
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

/// Render the inner Array for a FlatLayout.
fn render_array(app: &AppState, area: Rect, buf: &mut Buffer, is_stats_table: bool) {
    // Array data is loaded eagerly when navigating to a FlatLayout (synchronously on
    // native, asynchronously on WASM) and cached in AppState. The render loop never
    // performs I/O.
    let array = match app.cached_flat_array.as_ref() {
        Some(arr) => arr.clone(),
        None => {
            let loading =
                Paragraph::new("Loading array data...").style(Style::default().fg(Color::DarkGray));
            Widget::render(loading, area, buf);
            return;
        }
    };

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
        #[expect(deprecated)]
        let struct_array = array.to_struct();
        // add 1 for the chunk column
        let field_count = struct_array.struct_fields().nfields() + 1;
        let header = std::iter::once("chunk")
            .chain(struct_array.names().iter().map(|x| x.as_ref()))
            .map(Cell::from)
            .collect::<Row>()
            .style(
                Style::default()
                    .fg(Color::Rgb(206, 229, 98))
                    .bg(Color::DarkGray),
            )
            .height(1);

        assert_eq!(app.cursor.dtype(), array.dtype());

        let field_arrays: Vec<ArrayRef> = struct_array.unmasked_fields().to_vec();

        // TODO: trim the number of displayed rows and allow paging through column stats.
        let rows = (0..array.len()).map(|chunk_id| {
            std::iter::once(Cell::from(Text::from(format!("{chunk_id}"))))
                .chain(field_arrays.iter().map(|arr| {
                    Cell::from(Text::from(
                        arr.execute_scalar(chunk_id, &mut LEGACY_SESSION.create_execution_ctx())
                            .vortex_expect("scalar_at failed")
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

        let rows = array.statistics().with_iter(|iter| {
            iter.map(|(stat, value)| {
                let value = value.clone().into_scalar(
                    stat.dtype(array.dtype())
                        .vortex_expect("stat invalid for dtype"),
                );
                let stat = Cell::from(Text::from(format!("{stat}")));
                let value = Cell::from(Text::from(format!("{value}")));
                Row::new(vec![stat, value])
            })
            .collect::<Vec<_>>()
        });

        let layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints(vec![Constraint::Percentage(70), Constraint::Percentage(30)])
            .split(widget_area);
        let table = Table::new(rows, [Constraint::Min(6), Constraint::Min(6)]).header(header);
        // Tree-display the active array with scroll support
        let tree = Paragraph::new(array.display_tree().to_string())
            .wrap(Wrap { trim: false })
            .scroll((app.tree_scroll_offset, 0));

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
    let search_filter = &app.search_filter;
    let layout = app.cursor.layout();

    if layout.nchildren() > 0 {
        if search_filter.is_empty() {
            // No search filter, show all items
            let list_items = layout
                .child_names()
                .map(|name| name.to_string())
                .collect_vec();

            app.filter = None;
            render_child_list_items(app, area, buf, list_items);
        } else {
            // Use fuzzy matching to rank and filter results
            let matcher = SkimMatcherV2::default();

            // Collect matches
            let matches = layout
                .child_names()
                .enumerate()
                .filter_map(|(idx, name)| {
                    matcher
                        .fuzzy_match(&name, search_filter)
                        .map(|_| (idx, name.to_string()))
                })
                .collect_vec();

            // Create filter based on fuzzy matches
            let mut filter = vec![false; layout.nchildren()];
            let list_items = matches
                .iter()
                .map(|(idx, name)| {
                    filter[*idx] = true;
                    name.clone()
                })
                .collect_vec();

            app.filter = Some(filter);
            render_child_list_items(app, area, buf, list_items);
        }
    }
}

fn render_child_list_items(
    app: &mut AppState,
    area: Rect,
    buf: &mut Buffer,
    list_items: Vec<String>,
) {
    let container = Block::new()
        .title("Child Layouts")
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(Style::default().fg(Color::DarkGray));

    let inner_area = container.inner(area);

    container.render(area, buf);

    // Render the List view.
    StatefulWidget::render(
        List::new(list_items).highlight_style(
            Style::default()
                .fg(Color::Rgb(16, 16, 16))
                .bg(Color::Rgb(89, 113, 253))
                .bold(),
        ),
        inner_area,
        buf,
        &mut app.layouts_list_state,
    );
}
