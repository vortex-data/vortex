// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayRef;

/// A lightweight view of one node in an array encoding tree.
///
/// The view borrows the underlying array. It does not own or collect its descendants.
#[derive(Clone, Copy)]
pub struct ArrayTreeNode<'a> {
    array: &'a ArrayRef,
}

impl<'a> ArrayTreeNode<'a> {
    /// Create a view over one array node.
    pub fn new(array: &'a ArrayRef) -> Self {
        Self { array }
    }

    /// Return the underlying array for this node.
    pub fn array(self) -> &'a ArrayRef {
        self.array
    }

    /// Visit the named children without collecting or cloning them.
    pub fn children(self) -> impl Iterator<Item = (String, ArrayTreeNode<'a>)> {
        self.array
            .named_children_iter()
            .map(|(name, child)| (name, ArrayTreeNode::new(child)))
    }
}

/// One event in a streaming depth-first traversal of an array encoding tree.
pub enum ArrayTreeEvent<'a> {
    /// A node is being entered. Its children, if any, follow this event.
    Enter {
        /// The node's name within its parent, or `root` for the root node.
        name: &'a str,
        /// The lightweight node view.
        node: ArrayTreeNode<'a>,
        /// Depth in the tree, where the root has depth zero.
        depth: usize,
        /// Whether this is the first child of its parent.
        is_first: bool,
        /// Whether this is the final child of its parent.
        is_last: bool,
    },
    /// A node and all of its children have been visited.
    Exit {
        /// The lightweight node view.
        node: ArrayTreeNode<'a>,
        /// Depth in the tree, where the root has depth zero.
        depth: usize,
    },
}

/// Visit an array encoding tree once in depth-first order.
///
/// Nodes are projected and passed to `visitor` one at a time. No intermediate tree or child
/// collection is allocated; traversal state is proportional to tree depth.
pub fn walk_array_tree<E>(
    root: &ArrayRef,
    mut visitor: impl FnMut(ArrayTreeEvent<'_>) -> Result<(), E>,
) -> Result<(), E> {
    fn walk_node<E>(
        name: &str,
        node: ArrayTreeNode<'_>,
        depth: usize,
        is_first: bool,
        is_last: bool,
        visitor: &mut impl FnMut(ArrayTreeEvent<'_>) -> Result<(), E>,
    ) -> Result<(), E> {
        visitor(ArrayTreeEvent::Enter {
            name,
            node,
            depth,
            is_first,
            is_last,
        })?;

        let mut children = node.children().peekable();
        let mut is_first_child = true;
        while let Some((child_name, child)) = children.next() {
            let child_is_last = children.peek().is_none();
            walk_node(
                &child_name,
                child,
                depth + 1,
                is_first_child,
                child_is_last,
                visitor,
            )?;
            is_first_child = false;
        }

        visitor(ArrayTreeEvent::Exit { node, depth })
    }

    walk_node(
        "root",
        ArrayTreeNode::new(root),
        0,
        true,
        true,
        &mut visitor,
    )
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::ArrayTreeEvent;
    use super::walk_array_tree;
    use crate::IntoArray;
    use crate::arrays::StructArray;

    #[test]
    fn visits_each_array_node_once_in_stream_order() {
        let array = StructArray::from_fields(&[
            ("x", buffer![1_i32, 2].into_array()),
            ("y", buffer![3_i32, 4].into_array()),
        ])
        .unwrap()
        .into_array();
        let mut events = Vec::new();

        walk_array_tree(&array, |event| {
            match event {
                ArrayTreeEvent::Enter { name, depth, .. } => {
                    events.push(format!("enter:{depth}:{name}"));
                }
                ArrayTreeEvent::Exit { depth, .. } => events.push(format!("exit:{depth}")),
            }
            Ok::<_, std::convert::Infallible>(())
        })
        .unwrap();

        assert_eq!(
            events,
            [
                "enter:0:root",
                "enter:1:x",
                "exit:1",
                "enter:1:y",
                "exit:1",
                "exit:0",
            ]
        );
    }
}
