// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt;

/// Internal intermediate representation of a single node in the display tree.
///
/// Built by collecting annotations from all extractors, then rendered to text.
pub(crate) struct DisplayNode {
    /// The name label for this node (e.g. "root", "x", "values").
    pub(crate) name: String,
    /// The encoding summary (e.g. "vortex.primitive(i16, len=5)").
    pub(crate) encoding_summary: String,
    /// Annotations appended to the header line, collected from extractors.
    pub(crate) header_annotations: Vec<String>,
    /// Detail lines shown below the header, collected from extractors.
    pub(crate) detail_lines: Vec<String>,
    /// Recursive children.
    pub(crate) children: Vec<DisplayNode>,
}

impl DisplayNode {
    /// Render this node and all descendants to the formatter with the given indent prefix.
    pub(crate) fn render(&self, f: &mut fmt::Formatter<'_>, indent: &str) -> fmt::Result {
        // Header line: "{indent}{name}: {encoding_summary} {annotations...}\n"
        write!(f, "{indent}{}: {}", self.name, self.encoding_summary)?;
        for ann in &self.header_annotations {
            write!(f, " {ann}")?;
        }
        writeln!(f)?;

        // Detail lines
        let child_indent = format!("{indent}  ");
        for line in &self.detail_lines {
            writeln!(f, "{child_indent}{line}")?;
        }

        // Children
        for child in &self.children {
            child.render(f, &child_indent)?;
        }

        Ok(())
    }
}
