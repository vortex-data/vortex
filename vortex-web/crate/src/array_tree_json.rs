// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::ArrayRef;
use vortex::array::display::ArrayTreeEvent;
use vortex::array::display::walk_array_tree;
use vortex::array::session::ArraySessionExt;
use vortex::session::VortexSession;

trait TraversalObserver {
    fn enter(&self, _array: &ArrayRef) {}
    fn exit(&self, _array: &ArrayRef) {}
}

impl TraversalObserver for () {}

/// Stream an array encoding tree directly to JSON in one depth-first traversal.
///
/// Each `ArrayTreeNode` is written before its children are visited. No recursive JSON model is
/// constructed, and the array's child iterator is consumed exactly once per node.
#[cfg(target_arch = "wasm32")]
pub(super) fn write_array_encoding_tree_json(
    array: &ArrayRef,
    session: &VortexSession,
) -> serde_json::Result<String> {
    write_array_encoding_tree_json_with_observer(array, session, &())
}

fn write_array_encoding_tree_json_with_observer(
    array: &ArrayRef,
    session: &VortexSession,
    observer: &impl TraversalObserver,
) -> serde_json::Result<String> {
    let mut output = Vec::new();

    walk_array_tree(array, |event| -> serde_json::Result<()> {
        match event {
            ArrayTreeEvent::Enter {
                name,
                node,
                depth,
                is_first,
                ..
            } => {
                let array = node.array();
                observer.enter(array);
                if depth > 0 && !is_first {
                    output.extend_from_slice(b",");
                }

                let name = if depth == 0 { "array" } else { name };
                let encoding = array.encoding_id().to_string();
                let dtype = array.dtype().to_string();
                let buffer_names = array.buffer_names();
                let buffer_handles = array.buffer_handles();
                let buffer_lengths: Vec<usize> =
                    buffer_handles.iter().map(|buffer| buffer.len()).collect();
                let metadata_bytes = session
                    .array_serialize(array)
                    .ok()
                    .flatten()
                    .map(|metadata| metadata.len())
                    .unwrap_or(0);

                output.extend_from_slice(b"{\"name\":");
                serde_json::to_writer(&mut output, name)?;
                output.extend_from_slice(b",\"encoding\":");
                serde_json::to_writer(&mut output, &encoding)?;
                output.extend_from_slice(b",\"dtype\":");
                serde_json::to_writer(&mut output, &dtype)?;
                output.extend_from_slice(b",\"metadataBytes\":");
                serde_json::to_writer(&mut output, &metadata_bytes)?;
                output.extend_from_slice(b",\"numBuffers\":");
                serde_json::to_writer(&mut output, &buffer_lengths.len())?;
                output.extend_from_slice(b",\"bufferLengths\":");
                serde_json::to_writer(&mut output, &buffer_lengths)?;
                output.extend_from_slice(b",\"bufferNames\":");
                serde_json::to_writer(&mut output, &buffer_names)?;
                output.extend_from_slice(b",\"children\":[");
            }
            ArrayTreeEvent::Exit { node, .. } => {
                observer.exit(node.array());
                output.extend_from_slice(b"]}");
            }
        }
        Ok(())
    })?;

    // Every byte written above is either fixed UTF-8 syntax or emitted by serde_json.
    String::from_utf8(output).map_err(<serde_json::Error as serde::ser::Error>::custom)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use serde_json::Value;
    use vortex::VortexSessionDefault;
    use vortex::array::IntoArray;
    use vortex::array::arrays::StructArray;
    use vortex::buffer::buffer;
    use vortex::session::VortexSession;

    use super::TraversalObserver;
    use super::write_array_encoding_tree_json_with_observer;

    #[derive(Default)]
    struct CountingObserver {
        enters: Cell<usize>,
        exits: Cell<usize>,
    }

    impl TraversalObserver for CountingObserver {
        fn enter(&self, _array: &vortex::array::ArrayRef) {
            self.enters.set(self.enters.get() + 1);
        }

        fn exit(&self, _array: &vortex::array::ArrayRef) {
            self.exits.set(self.exits.get() + 1);
        }
    }

    #[test]
    fn serializes_each_array_node_exactly_once() -> Result<(), Box<dyn std::error::Error>> {
        let array = StructArray::from_fields(&[
            ("x", buffer![1_i32, 2].into_array()),
            ("y", buffer![3_i32, 4].into_array()),
        ])?
        .into_array();
        let session = VortexSession::default();
        let observer = CountingObserver::default();

        let json = write_array_encoding_tree_json_with_observer(&array, &session, &observer)?;
        let tree: Value = serde_json::from_str(&json)?;

        assert_eq!(observer.enters.get(), 3);
        assert_eq!(observer.exits.get(), 3);
        assert_eq!(tree["name"], "array");
        assert_eq!(tree["children"][0]["name"], "x");
        assert_eq!(tree["children"][1]["name"], "y");
        Ok(())
    }
}
