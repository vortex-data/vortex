// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use humansize::DECIMAL;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::prelude::Alignment;
use ratatui::prelude::Line;
use ratatui::prelude::Margin;
use ratatui::prelude::StatefulWidget;
use ratatui::prelude::Widget;
use ratatui::widgets::Block;
use ratatui::widgets::Borders;
use ratatui::widgets::Paragraph;
use ratatui::widgets::Scrollbar;
use ratatui::widgets::ScrollbarOrientation;
use ratatui::widgets::ScrollbarState;
use ratatui::widgets::Wrap;
use taffy::AvailableSpace;
use taffy::Dimension;
use taffy::FlexDirection;
use taffy::LengthPercentage;
use taffy::NodeId;
use taffy::PrintTree;
use taffy::Size;
use taffy::Style;
use taffy::TaffyTree;
use taffy::TraversePartialTree;
use vortex::dtype::FieldName;
use vortex::error::VortexExpect;
use vortex::error::vortex_err;
use vortex::utils::aliases::hash_map::HashMap;

use crate::browse::app::AppState;
use crate::segment_tree::SegmentTree;
use crate::segment_tree::collect_segment_tree;

/// State for the segment grid visualization.
///
/// This struct manages the layout tree and scroll state for displaying segments in a grid view.
/// The segment tree is lazily computed on first render and cached for subsequent frames.
#[derive(Debug, Clone, Default)]
pub struct SegmentGridState<'a> {
    /// The computed layout tree for the segment grid, or `None` if not yet computed.
    ///
    /// Contains the taffy layout tree, root node ID, and a map of node contents.
    pub segment_tree: Option<(TaffyTree<()>, NodeId, HashMap<NodeId, NodeContents<'a>>)>,

    /// State for the horizontal scrollbar widget.
    pub horizontal_scroll_state: ScrollbarState,

    /// State for the vertical scrollbar widget.
    pub vertical_scroll_state: ScrollbarState,

    /// Current vertical scroll position in pixels.
    pub vertical_scroll: usize,

    /// Current horizontal scroll position in pixels.
    pub horizontal_scroll: usize,

    /// Maximum horizontal scroll position.
    pub max_horizontal_scroll: usize,

    /// Maximum vertical scroll position.
    pub max_vertical_scroll: usize,
}

impl SegmentGridState<'_> {
    /// Scroll the viewport up by the given amount.
    pub fn scroll_up(&mut self, amount: usize) {
        self.vertical_scroll = self.vertical_scroll.saturating_sub(amount);
        self.vertical_scroll_state = self.vertical_scroll_state.position(self.vertical_scroll);
    }

    /// Scroll the viewport down by the given amount.
    pub fn scroll_down(&mut self, amount: usize) {
        self.vertical_scroll = self
            .vertical_scroll
            .saturating_add(amount)
            .min(self.max_vertical_scroll);
        self.vertical_scroll_state = self.vertical_scroll_state.position(self.vertical_scroll);
    }

    /// Scroll the viewport left by the given amount.
    pub fn scroll_left(&mut self, amount: usize) {
        self.horizontal_scroll = self.horizontal_scroll.saturating_sub(amount);
        self.horizontal_scroll_state = self
            .horizontal_scroll_state
            .position(self.horizontal_scroll);
    }

    /// Scroll the viewport right by the given amount.
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
    title: FieldName,
    contents: Vec<Line<'a>>,
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "UI coordinates are small enough"
)]
pub fn segments_ui(app_state: &mut AppState, area: Rect, buf: &mut Buffer) {
    if app_state.segment_grid_state.segment_tree.is_none() {
        let segment_tree = collect_segment_tree(
            app_state.vxf.footer().layout().as_ref(),
            app_state.vxf.footer().segment_map(),
        );
        app_state.segment_grid_state.segment_tree = Some(
            to_display_segment_tree(segment_tree)
                .map_err(|e| vortex_err!("Fail to compute segment tree {e}"))
                .vortex_expect("operation should succeed in TUI"),
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
            .vortex_expect("operation should succeed in TUI");
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

#[expect(
    clippy::cast_possible_truncation,
    reason = "UI coordinates are small enough"
)]
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
            .title(blk_content.title.as_ref())
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
    mut segment_tree: SegmentTree,
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
                .get_mut(&name)
                .vortex_expect("Must have segment for name");
            chunks.sort_by(|a, b| a.spec.offset.cmp(&b.spec.offset));

            // Build leaf nodes for each segment chunk.
            let mut leaves = Vec::with_capacity(chunks.len());
            let mut current_offset = 0u64;
            for segment in chunks.iter() {
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
                                humansize::format_size(segment.spec.length, DECIMAL),
                            )),
                            Line::raw(format!("Align: {}", segment.spec.alignment)),
                            Line::raw(format!(
                                "Byte gap: {}",
                                if current_offset == 0 {
                                    0
                                } else {
                                    segment.spec.offset - current_offset
                                }
                            )),
                        ],
                    },
                );

                current_offset = segment.spec.length as u64 + segment.spec.offset;
                leaves.push(node_id);
            }

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
                    title: name,
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
