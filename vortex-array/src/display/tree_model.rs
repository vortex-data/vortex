// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A unified tree model for displaying hierarchical data structures.
//!
//! This module provides two approaches for tree display:
//!
//! 1. **Lazy (trait-based)**: Implement [`TreeDisplayable`] for your type, then use
//!    [`TreeDisplay`] wrapper to render it. Tree is walked lazily during rendering.
//!
//! 2. **Eager (struct-based)**: Build a [`DisplayTreeNode`] explicitly, which can
//!    then be rendered as text or JSON.
//!
//! # Example using the trait (lazy)
//!
//! ```ignore
//! use vortex_array::display::tree_model::{TreeDisplayable, TreeDisplay, Attr};
//!
//! struct MyNode { /* ... */ }
//!
//! impl TreeDisplayable for MyNode {
//!     fn name(&self) -> std::borrow::Cow<'_, str> { "my.node".into() }
//!     fn attrs(&self) -> Vec<Attr> { vec![] }
//!     fn nested_attrs(&self) -> Vec<Attr> { vec![] }
//!     fn children(&self) -> Vec<(std::borrow::Cow<'_, str>, &dyn TreeDisplayable)> { vec![] }
//! }
//!
//! let node = MyNode { /* ... */ };
//! println!("{}", TreeDisplay(&node));
//! ```
//!
//! # Example using DisplayTreeNode (eager)
//!
//! ```
//! use vortex_array::display::tree_model::{DisplayTreeNode, Attr, AttrValue};
//!
//! let node = DisplayTreeNode::new("vortex.struct")
//!     .with_attr("dtype", "{x=i64}")
//!     .with_attr("rows", 5u64);
//!
//! // Text output
//! println!("{}", node);
//!
//! // JSON output
//! println!("{}", serde_json::to_string_pretty(&node).unwrap());
//! ```

use std::borrow::Cow;
use std::fmt::{self, Display, Formatter};

use indexmap::IndexMap;
#[cfg(feature = "serde")]
use serde::Serialize;

/// Trait for types that can be displayed as a tree.
///
/// Implement this trait to enable lazy tree rendering. The tree is walked
/// during display/serialization rather than being built upfront.
pub trait TreeDisplayable {
    /// The node's primary name/label (e.g., "vortex.primitive").
    fn name(&self) -> Cow<'_, str>;

    /// Inline attributes shown on the same line as the name.
    fn attrs(&self) -> Vec<Attr>;

    /// Nested attributes displayed on their own indented lines.
    fn nested_attrs(&self) -> Vec<Attr> {
        Vec::new()
    }

    /// Named children of this node.
    ///
    /// Returns a vector of (name, child) pairs. The name is used as a label
    /// in tree display (e.g., "numbers:" or "[0]:").
    fn children(&self) -> Vec<(Cow<'_, str>, &dyn TreeDisplayable)> {
        Vec::new()
    }

    /// Convert this node to a [`DisplayTreeNode`] (eager/owned representation).
    ///
    /// This walks the entire tree and builds an owned copy suitable for
    /// JSON serialization or when you need to store the tree data.
    fn to_tree_node(&self) -> DisplayTreeNode {
        let mut node = DisplayTreeNode::new(self.name().into_owned());
        node.attrs = self.attrs();
        node.nested_attrs = self.nested_attrs();

        for (child_name, child) in self.children() {
            node.children
                .insert(child_name.into_owned(), child.to_tree_node());
        }

        node
    }
}

/// Wrapper for displaying a [`TreeDisplayable`] as text.
///
/// This renders the tree lazily during `Display::fmt`, avoiding the need
/// to build an intermediate data structure.
pub struct TreeDisplay<'a, T: TreeDisplayable + ?Sized>(pub &'a T);

impl<T: TreeDisplayable> Display for TreeDisplay<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write_tree(f, self.0, "")
    }
}

/// Write a tree node and its children with proper indentation.
fn write_tree(
    f: &mut Formatter<'_>,
    node: &dyn TreeDisplayable,
    prefix: &str,
) -> fmt::Result {
    // Build the node line: "name, attr1: val1, attr2: val2"
    let name = node.name();
    write!(f, "{}", name)?;
    for attr in node.attrs() {
        write!(f, ", {}", attr)?;
    }
    writeln!(f)?;

    // Get children and nested attrs
    let nested = node.nested_attrs();
    let children = node.children();

    // Write nested attributes
    for (i, attr) in nested.iter().enumerate() {
        let is_last_item = children.is_empty() && i == nested.len() - 1;
        let connector = if is_last_item { "└── " } else { "├── " };
        writeln!(f, "{}{}{}", prefix, connector, attr)?;
    }

    // Write children
    for (i, (child_name, child)) in children.iter().enumerate() {
        let is_last_child = i == children.len() - 1;
        let connector = if is_last_child { "└── " } else { "├── " };
        let child_prefix = if is_last_child {
            format!("{}    ", prefix)
        } else {
            format!("{}│   ", prefix)
        };

        write!(f, "{}{}{}: ", prefix, connector, child_name)?;
        write_tree(f, *child, &child_prefix)?;
    }

    Ok(())
}

/// A generic tree node for display and serialization (eager/owned).
///
/// This struct captures hierarchical data in a format-agnostic way, allowing
/// it to be rendered as either text (tree view) or JSON. Use this when you
/// need to store the tree data or serialize to JSON.
///
/// For lazy rendering during display, implement [`TreeDisplayable`] instead.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct DisplayTreeNode {
    /// Primary label for the node (e.g., "vortex.primitive", "vortex.struct").
    pub name: String,

    /// Inline attributes shown on the same line as the name.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Vec::is_empty"))]
    pub attrs: Vec<Attr>,

    /// Nested attributes rendered on their own indented lines.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Vec::is_empty"))]
    pub nested_attrs: Vec<Attr>,

    /// Named child nodes.
    ///
    /// Uses [`IndexMap`] to preserve insertion order for consistent display.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "IndexMap::is_empty"))]
    pub children: IndexMap<String, DisplayTreeNode>,
}

impl DisplayTreeNode {
    /// Create a new tree node with just a name.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            attrs: Vec::new(),
            nested_attrs: Vec::new(),
            children: IndexMap::new(),
        }
    }

    /// Add an inline attribute.
    #[must_use]
    pub fn with_attr(mut self, key: impl Into<String>, value: impl Into<AttrValue>) -> Self {
        self.attrs.push(Attr::new(key, value));
        self
    }

    /// Add a nested attribute (displayed on its own line).
    #[must_use]
    pub fn with_nested_attr(mut self, key: impl Into<String>, value: impl Into<AttrValue>) -> Self {
        self.nested_attrs.push(Attr::new(key, value));
        self
    }

    /// Add a child node.
    #[must_use]
    pub fn with_child(mut self, name: impl Into<String>, child: DisplayTreeNode) -> Self {
        self.children.insert(name.into(), child);
        self
    }
}

/// Implement TreeDisplayable for DisplayTreeNode so it can use the lazy renderer too.
impl TreeDisplayable for DisplayTreeNode {
    fn name(&self) -> Cow<'_, str> {
        Cow::Borrowed(&self.name)
    }

    fn attrs(&self) -> Vec<Attr> {
        self.attrs.clone()
    }

    fn nested_attrs(&self) -> Vec<Attr> {
        self.nested_attrs.clone()
    }

    fn children(&self) -> Vec<(Cow<'_, str>, &dyn TreeDisplayable)> {
        self.children
            .iter()
            .map(|(name, child)| (Cow::Borrowed(name.as_str()), child as &dyn TreeDisplayable))
            .collect()
    }
}

impl Display for DisplayTreeNode {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", TreeDisplay(self))
    }
}

/// A key-value attribute.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct Attr {
    /// The attribute key/name.
    pub key: String,
    /// The attribute value.
    pub value: AttrValue,
}

impl Attr {
    /// Create a new attribute.
    pub fn new(key: impl Into<String>, value: impl Into<AttrValue>) -> Self {
        Self {
            key: key.into(),
            value: value.into(),
        }
    }
}

impl Display for Attr {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.key, self.value)
    }
}

/// A dynamically-typed attribute value.
///
/// This enum supports common value types and serializes naturally to JSON
/// using `#[serde(untagged)]`.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize))]
#[cfg_attr(feature = "serde", serde(untagged))]
pub enum AttrValue {
    /// A string value.
    String(String),
    /// A signed integer value.
    Int(i64),
    /// An unsigned integer value.
    UInt(u64),
    /// A floating-point value.
    Float(f64),
    /// A boolean value.
    Bool(bool),
    /// A list of values.
    List(Vec<AttrValue>),
}

impl Display for AttrValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            AttrValue::String(s) => write!(f, "{}", s),
            AttrValue::Int(n) => write!(f, "{}", n),
            AttrValue::UInt(n) => write!(f, "{}", n),
            AttrValue::Float(n) => write!(f, "{}", n),
            AttrValue::Bool(b) => write!(f, "{}", b),
            AttrValue::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
        }
    }
}

// Implement From for common types to make building trees ergonomic.

impl From<String> for AttrValue {
    fn from(s: String) -> Self {
        AttrValue::String(s)
    }
}

impl From<&str> for AttrValue {
    fn from(s: &str) -> Self {
        AttrValue::String(s.to_string())
    }
}

impl From<i64> for AttrValue {
    fn from(n: i64) -> Self {
        AttrValue::Int(n)
    }
}

impl From<i32> for AttrValue {
    fn from(n: i32) -> Self {
        AttrValue::Int(n.into())
    }
}

impl From<u64> for AttrValue {
    fn from(n: u64) -> Self {
        AttrValue::UInt(n)
    }
}

impl From<u32> for AttrValue {
    fn from(n: u32) -> Self {
        AttrValue::UInt(n.into())
    }
}

impl From<usize> for AttrValue {
    fn from(n: usize) -> Self {
        AttrValue::UInt(n as u64)
    }
}

impl From<f64> for AttrValue {
    fn from(n: f64) -> Self {
        AttrValue::Float(n)
    }
}

impl From<bool> for AttrValue {
    fn from(b: bool) -> Self {
        AttrValue::Bool(b)
    }
}

impl<T: Into<AttrValue>> From<Vec<T>> for AttrValue {
    fn from(v: Vec<T>) -> Self {
        AttrValue::List(v.into_iter().map(Into::into).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_node_display() {
        let node = DisplayTreeNode::new("vortex.primitive")
            .with_attr("dtype", "i64")
            .with_attr("rows", 100u64);

        let output = node.to_string();
        assert!(output.contains("vortex.primitive"));
        assert!(output.contains("dtype: i64"));
        assert!(output.contains("rows: 100"));
    }

    #[test]
    fn test_nested_attrs_display() {
        let node = DisplayTreeNode::new("vortex.flat")
            .with_attr("dtype", "i64")
            .with_nested_attr("segment", 0u64)
            .with_nested_attr("buffers", vec![40usize, 1usize]);

        let output = node.to_string();
        assert!(output.contains("vortex.flat"));
        assert!(output.contains("segment: 0"));
        assert!(output.contains("buffers: [40, 1]"));
    }

    #[test]
    fn test_children_display() {
        let child = DisplayTreeNode::new("vortex.primitive").with_attr("dtype", "i64");

        let parent = DisplayTreeNode::new("vortex.struct")
            .with_attr("dtype", "{x=i64}")
            .with_child("x", child);

        let output = parent.to_string();
        assert!(output.contains("vortex.struct"));
        assert!(output.contains("x: vortex.primitive"));
    }

    #[test]
    fn test_json_serialization() {
        let child = DisplayTreeNode::new("vortex.primitive").with_attr("dtype", "i64");

        let parent = DisplayTreeNode::new("vortex.struct")
            .with_attr("dtype", "{x=i64}")
            .with_attr("rows", 5u64)
            .with_child("x", child);

        let json = serde_json::to_string_pretty(&parent).unwrap();

        // Verify JSON structure
        assert!(json.contains("\"name\": \"vortex.struct\""));
        assert!(json.contains("\"key\": \"dtype\""));
        assert!(json.contains("\"value\": \"{x=i64}\""));
        assert!(json.contains("\"x\":"));
    }

    #[test]
    fn test_attr_value_list() {
        let value: AttrValue = vec![40usize, 1usize].into();
        assert_eq!(value.to_string(), "[40, 1]");
    }

    #[test]
    fn test_tree_displayable_trait() {
        // Test the trait-based lazy approach
        struct SimpleNode {
            name: String,
            value: i32,
        }

        impl TreeDisplayable for SimpleNode {
            fn name(&self) -> Cow<'_, str> {
                Cow::Borrowed(&self.name)
            }

            fn attrs(&self) -> Vec<Attr> {
                vec![Attr::new("value", self.value)]
            }
        }

        let node = SimpleNode {
            name: "test.node".to_string(),
            value: 42,
        };

        let output = TreeDisplay(&node).to_string();
        assert!(output.contains("test.node"));
        assert!(output.contains("value: 42"));
    }

    #[test]
    fn test_to_tree_node_conversion() {
        struct ParentNode {
            child: ChildNode,
        }

        struct ChildNode {
            value: i32,
        }

        impl TreeDisplayable for ChildNode {
            fn name(&self) -> Cow<'_, str> {
                "child".into()
            }

            fn attrs(&self) -> Vec<Attr> {
                vec![Attr::new("value", self.value)]
            }
        }

        impl TreeDisplayable for ParentNode {
            fn name(&self) -> Cow<'_, str> {
                "parent".into()
            }

            fn attrs(&self) -> Vec<Attr> {
                vec![]
            }

            fn children(&self) -> Vec<(Cow<'_, str>, &dyn TreeDisplayable)> {
                vec![("my_child".into(), &self.child as &dyn TreeDisplayable)]
            }
        }

        let node = ParentNode {
            child: ChildNode { value: 123 },
        };

        // Test lazy display
        let display_output = TreeDisplay(&node).to_string();
        assert!(display_output.contains("parent"));
        assert!(display_output.contains("my_child: child"));
        assert!(display_output.contains("value: 123"));

        // Test conversion to owned tree node (for JSON)
        let tree_node = node.to_tree_node();
        assert_eq!(tree_node.name, "parent");
        assert!(tree_node.children.contains_key("my_child"));

        let json = serde_json::to_string(&tree_node).unwrap();
        assert!(json.contains("\"my_child\""));
    }
}
