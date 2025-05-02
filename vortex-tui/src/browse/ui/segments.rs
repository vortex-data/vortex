use std::sync::Arc;

use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::prelude::{Alignment, Line, Margin, StatefulWidget, Widget};
use ratatui::widgets::{
    Block, Borders, Paragraph, Scrollbar, ScrollbarOrientation, ScrollbarState, Wrap,
};
use taffy::{
    AvailableSpace, Dimension, FlexDirection, LengthPercentage, NodeId, PrintTree, Size, Style,
    TaffyTree, TraversePartialTree,
};
use vortex::aliases::hash_map::HashMap;
use vortex::error::{VortexExpect, VortexResult, VortexUnwrap, vortex_err};
use vortex::file::SegmentSpec;
use vortex_layout::layouts::chunked::ChunkedLayout;
use vortex_layout::layouts::dict::DictLayout;
use vortex_layout::layouts::flat::FlatLayout;
use vortex_layout::layouts::stats::StatsLayout;
use vortex_layout::layouts::struct_::StructLayout;
use vortex_layout::{Layout, LayoutVTable};

use crate::browse::app::AppState;

#[derive(Debug, Clone, Default)]
pub struct SegmentGridState<'a> {
    /// state for the segment grid layout
    pub segment_tree: Option<(TaffyTree<()>, NodeId, HashMap<NodeId, NodeContents<'a>>)>,
    pub horizontal_scroll_state: ScrollbarState,
    pub vertical_scroll_state: ScrollbarState,
    pub vertical_scroll: usize,
    pub horizontal_scroll: usize,
    pub max_horizontal_scroll: usize,
    pub max_vertical_scroll: usize,
}

impl SegmentGridState<'_> {
    pub fn scroll_up(&mut self, amount: usize) {
        self.vertical_scroll = self.vertical_scroll.saturating_sub(amount);
        self.vertical_scroll_state = self.vertical_scroll_state.position(self.vertical_scroll);
    }

    pub fn scroll_down(&mut self, amount: usize) {
        self.vertical_scroll = self
            .vertical_scroll
            .saturating_add(amount)
            .min(self.max_vertical_scroll);
        self.vertical_scroll_state = self.vertical_scroll_state.position(self.vertical_scroll);
    }

    pub fn scroll_left(&mut self, amount: usize) {
        self.horizontal_scroll = self.horizontal_scroll.saturating_sub(amount);
        self.horizontal_scroll_state = self
            .horizontal_scroll_state
            .position(self.horizontal_scroll);
    }

    pub fn scroll_right(&mut self, amount: usize) {
        self.horizontal_scroll = self
            .horizontal_scroll
            .saturating_add(amount)
            .min(self.max_horizontal_scroll);
        self.horizontal_scroll_state = self
            .horizontal_scroll_state
            .position(self.horizontal_scroll);
    }
}

#[derive(Debug, Clone)]
pub struct NodeContents<'a> {
    title: Arc<str>,
    contents: Vec<Line<'a>>,
}

pub struct SegmentDisplay {
    name: Arc<str>,
    spec: SegmentSpec,
    row_offset: u64,
    row_count: u64,
    byte_gap: u64,
}

#[allow(clippy::cast_possible_truncation)]
pub fn segments_ui(app_state: &mut AppState, area: Rect, buf: &mut Buffer) {
    if app_state.segment_grid_state.segment_tree.is_none() {
        let segment_tree = collect_segment_tree(
            app_state.vxf.footer().layout(),
            app_state.vxf.footer().segment_map(),
        );
        app_state.segment_grid_state.segment_tree = Some(
            to_display_segment_tree(segment_tree)
                .map_err(|e| vortex_err!("Fail to compute segment tree {e}"))
                .vortex_unwrap(),
        );
    }

    let Some((tree, root_node, contents)) = &mut app_state.segment_grid_state.segment_tree else {
        unreachable!("uninitialized state")
    };

    if app_state.frame_size != area.as_size() {
        let viewport_size = Size {
            width: AvailableSpace::Definite(area.width as f32),
            height: AvailableSpace::Definite(area.height as f32),
        };
        tree.compute_layout(*root_node, viewport_size)
            .map_err(|e| vortex_err!("Fail to compute layout {e}"))
            .vortex_unwrap();
        app_state.frame_size = area.as_size();

        let root_layout = tree.get_final_layout(*root_node);

        app_state.segment_grid_state.max_horizontal_scroll = root_layout.scroll_width() as usize;
        app_state.segment_grid_state.max_vertical_scroll = root_layout.scroll_height() as usize;

        app_state.segment_grid_state.horizontal_scroll_state = app_state
            .segment_grid_state
            .horizontal_scroll_state
            .content_length(root_layout.scroll_width() as usize)
            .viewport_content_length(app_state.frame_size.width as usize)
            .position(app_state.segment_grid_state.horizontal_scroll);
        app_state.segment_grid_state.vertical_scroll_state = app_state
            .segment_grid_state
            .vertical_scroll_state
            .content_length(root_layout.scroll_height() as usize)
            .viewport_content_length(app_state.frame_size.height as usize)
            .position(app_state.segment_grid_state.vertical_scroll);
    }

    render_tree(
        tree,
        *root_node,
        contents,
        (
            app_state.segment_grid_state.horizontal_scroll,
            app_state.segment_grid_state.vertical_scroll,
        ),
        area,
        buf,
    );

    let horizontal_scroll = Scrollbar::new(ScrollbarOrientation::HorizontalBottom)
        .begin_symbol(Some("◄"))
        .end_symbol(Some("►"));
    horizontal_scroll.render(
        area,
        buf,
        &mut app_state.segment_grid_state.horizontal_scroll_state,
    );

    let vertical_scroll = Scrollbar::new(ScrollbarOrientation::VerticalRight)
        .begin_symbol(Some("▲"))
        .end_symbol(Some("▼"));
    vertical_scroll.render(
        area.inner(Margin {
            horizontal: 0,
            vertical: 1,
        }),
        buf,
        &mut app_state.segment_grid_state.vertical_scroll_state,
    );
}

#[allow(clippy::cast_possible_truncation)]
fn render_tree(
    tree: &TaffyTree<()>,
    node: NodeId,
    contents: &HashMap<NodeId, NodeContents>,
    viewport_top_left: (usize, usize),
    bounding_box: Rect,
    buf: &mut Buffer,
) -> Option<Rect> {
    let layout = tree.get_final_layout(node);

    let object_x = layout.location.x as usize;
    let object_y = layout.location.y as usize;

    let x_viewport = object_x.saturating_sub(viewport_top_left.0);
    let y_viewport = object_y.saturating_sub(viewport_top_left.1);

    let block_contents = contents.get(&node);
    if (viewport_top_left.0
        > layout.size.width as usize + layout.scroll_width() as usize + object_x
        || viewport_top_left.1
            > layout.size.height as usize + layout.scroll_height() as usize + object_y)
        && block_contents.is_some_and(|c| !c.contents.is_empty())
    {
        return None;
    }

    let r = bounding_box.intersection(Rect::new(
        x_viewport as u16 + bounding_box.x,
        y_viewport as u16 + bounding_box.y,
        layout.size.width as u16,
        layout.size.height as u16,
    ));

    if r.is_empty() {
        return None;
    }

    let mut block_to_render = None;
    if let Some(blk_content) = block_contents {
        for p in r.positions() {
            buf[p].reset()
        }

        let block = Block::new()
            .title(&*blk_content.title)
            .borders(Borders::ALL);

        if !blk_content.contents.is_empty() {
            Paragraph::new(blk_content.contents.clone())
                .block(block)
                .alignment(Alignment::Left)
                .wrap(Wrap { trim: true })
                .render(r, buf);
        } else {
            block_to_render = Some(block);
        }
    }

    let _child_area = tree
        .child_ids(node)
        .flat_map(|child_node_id| {
            render_tree(
                tree,
                child_node_id,
                contents,
                (
                    viewport_top_left.0.saturating_sub(object_x),
                    viewport_top_left.1.saturating_sub(object_y),
                ),
                r,
                buf,
            )
        })
        .reduce(|a, b| a.union(b));

    if let Some(block) = block_to_render {
        block.render(r, buf);
    }

    Some(r)
}

fn to_display_segment_tree<'a>(
    segment_tree: SegmentTree,
) -> anyhow::Result<(TaffyTree<()>, NodeId, HashMap<NodeId, NodeContents<'a>>)> {
    // Extra node for the parent node of the segment specs, and one parent node as the root
    let mut tree = TaffyTree::with_capacity(
        segment_tree
            .segments
            .iter()
            .map(|(_, v)| v.len() + 1)
            .sum::<usize>()
            + 1,
    );

    let mut node_contents: HashMap<NodeId, NodeContents> = HashMap::new();

    let children = segment_tree
        .segment_ordering
        .into_iter()
        .map(|name| {
            let chunks = segment_tree
                .segments
                .get(&name)
                .vortex_expect("Must have segment for name");
            let leaves = chunks
                .iter()
                .map(|segment| {
                    let node_id = tree.new_leaf(Style {
                        min_size: Size {
                            width: Dimension::percent(1.0),
                            height: Dimension::length(7.0),
                        },
                        size: Size {
                            width: Dimension::percent(1.0),
                            height: Dimension::length(15.0),
                        },
                        ..Default::default()
                    })?;
                    node_contents.insert(
                        node_id,
                        NodeContents {
                            title: segment.name.clone(),
                            contents: vec![
                                Line::raw(format!(
                                    "Rows: {}..{} ({})",
                                    segment.row_offset,
                                    segment.row_offset + segment.row_count,
                                    segment.row_count
                                )),
                                Line::raw(format!(
                                    "Bytes: {}..{} ({})",
                                    segment.spec.offset,
                                    segment.spec.offset + segment.spec.length as u64,
                                    segment.spec.length
                                )),
                                Line::raw(format!("Align: {}", segment.spec.alignment)),
                                Line::raw(format!("Byte gap: {}", segment.byte_gap)),
                            ],
                        },
                    );
                    Ok(node_id)
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            let node_id = tree.new_with_children(
                Style {
                    min_size: Size {
                        width: Dimension::length(40.0),
                        height: Dimension::percent(1.0),
                    },
                    padding: taffy::Rect {
                        left: LengthPercentage::length(1.0),
                        right: LengthPercentage::length(1.0),
                        top: LengthPercentage::length(1.0),
                        bottom: LengthPercentage::length(1.0),
                    },
                    flex_direction: FlexDirection::Column,
                    ..Default::default()
                },
                &leaves,
            )?;
            node_contents.insert(
                node_id,
                NodeContents {
                    title: name.clone(),
                    contents: Vec::new(),
                },
            );
            Ok(node_id)
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let root = tree.new_with_children(
        Style {
            size: Size {
                width: Dimension::percent(1.0),
                height: Dimension::percent(1.0),
            },
            flex_direction: FlexDirection::Row,
            ..Default::default()
        },
        &children,
    )?;
    Ok((tree, root, node_contents))
}

fn collect_segment_tree(root_layout: &Layout, segments: &Arc<[SegmentSpec]>) -> SegmentTree {
    let mut tree = SegmentTree {
        segments: HashMap::new(),
        segment_ordering: Vec::new(),
    };
    segments_by_name_impl(root_layout, None, None, segments, &mut tree).vortex_unwrap();

    tree
}

struct SegmentTree {
    segments: HashMap<Arc<str>, Vec<SegmentDisplay>>,
    segment_ordering: Vec<Arc<str>>,
}

fn segments_by_name_impl(
    root: &Layout,
    group_name: Option<Arc<str>>,
    name: Option<Arc<str>>,
    segments: &Arc<[SegmentSpec]>,
    segment_tree: &mut SegmentTree,
) -> VortexResult<()> {
    let layout_id = root.id();

    if layout_id == StructLayout.id() {
        let dtype = root.dtype().as_struct().vortex_expect("");
        for child_idx in 0..dtype.fields().len() {
            let child_name = dtype.field_name(child_idx)?;
            let child_dtype = dtype.field_by_index(child_idx)?;
            let child_layout = root.child(child_idx, child_dtype, child_name)?;
            let group_name = group_name.as_ref().map_or(child_name.clone(), |n| {
                Arc::from(format!("{n}.{child_name}"))
            });
            segment_tree.segment_ordering.push(group_name.clone());
            segments_by_name_impl(
                &child_layout,
                Some(group_name),
                name.clone(),
                segments,
                segment_tree,
            )?;
        }
    } else if layout_id == ChunkedLayout.id() {
        for child_idx in 0..root.nchildren() {
            let child_name = Arc::from(format!("[{child_idx}]"));
            let child_layout = root.child(child_idx, root.dtype().clone(), &child_name)?;
            let display_name = name.as_ref().unwrap_or(&child_name);
            segments_by_name_impl(
                &child_layout,
                group_name.clone(),
                Some(display_name.clone()),
                segments,
                segment_tree,
            )?;
        }
    } else if layout_id == StatsLayout.id() {
        let data_layout = root.child(0, root.dtype().clone(), "data")?;
        segments_by_name_impl(
            &data_layout,
            group_name.clone(),
            name.clone(),
            segments,
            segment_tree,
        )?;

        // For the stats layout, we use the stats segment accumulator
        let stats_layout = root.child(1, root.dtype().clone(), "stats")?;
        segments_by_name_impl(
            &stats_layout,
            group_name,
            Some(
                name.as_ref()
                    .map_or_else(|| Arc::from("stats"), |n| Arc::from(format!("{n}.stats"))),
            ),
            segments,
            segment_tree,
        )?;
    } else if layout_id == FlatLayout.id() {
        let current_segments = segment_tree
            .segments
            .entry(group_name.unwrap_or_else(|| Arc::from("root")))
            .or_default();
        // HACK: Pass row offset explicitly
        let last_row_offset = if name
            .as_ref()
            .map(|cn| cn.ends_with("stats"))
            .unwrap_or(false)
        {
            0
        } else {
            current_segments
                .last()
                .map(|s| s.row_offset + s.row_count)
                .unwrap_or(0)
        };

        let segment_spec = segments[*root
            .segment_id(0)
            .vortex_expect("flat layout missing segment")
            as usize]
            .clone();
        let byte_gap = current_segments
            .last()
            .map(|s| segment_spec.offset - s.spec.offset - s.spec.length as u64)
            .unwrap_or(0);
        current_segments.push(SegmentDisplay {
            name: name.unwrap_or_else(|| Arc::from("flat")),
            spec: segment_spec,
            row_count: root.row_count(),
            row_offset: last_row_offset,
            byte_gap,
        })
    } else if layout_id == DictLayout.id() {
        let values_layout = root.child(0, root.dtype().clone(), "values")?;
        segments_by_name_impl(
            &values_layout,
            group_name.clone(),
            Some(
                name.as_ref()
                    .map_or_else(|| Arc::from("values"), |n| Arc::from(format!("{n}.values"))),
            ),
            segments,
            segment_tree,
        )?;

        let codes_layout = root.child(1, root.dtype().clone(), "codes")?;
        segments_by_name_impl(
            &codes_layout,
            group_name,
            Some(name.map_or_else(|| Arc::from("codes"), |n| Arc::from(format!("{n}.codes")))),
            segments,
            segment_tree,
        )?;
    } else {
        unreachable!()
    };

    Ok(())
}
