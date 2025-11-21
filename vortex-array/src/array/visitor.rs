// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Formatter;
use std::sync::Arc;

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::arrays::ConstantArray;
use crate::patches::Patches;
use crate::validity::Validity;
use crate::{Array, ArrayRef};

pub trait ArrayVisitor {
    /// Returns the children of the array.
    fn children(&self) -> Vec<&dyn Array>;

    /// Visits each child array using the provided visitor, without allocating.
    ///
    /// This uses the vtable's visitor pattern directly and is the most efficient
    /// way to iterate over children.
    fn visit_children<'a>(&'a self, visitor: &mut dyn ArrayChildVisitor<'a>);

    /// Returns the number of children of the array.
    fn nchildren(&self) -> usize;

    /// Returns the names of the children of the array.
    fn children_names(&self) -> Vec<String>;

    /// Returns the array's children with their names.
    fn named_children(&self) -> Vec<(String, ArrayRef)>;

    /// Returns the buffers of the array.
    fn buffers(&self) -> Vec<ByteBuffer>;

    /// Returns the number of buffers of the array.
    fn nbuffers(&self) -> usize;

    /// Returns the serialized metadata of the array, or `None` if the array does not
    /// support serialization.
    fn metadata(&self) -> VortexResult<Option<Vec<u8>>>;

    /// Formats a human-readable metadata description.
    fn metadata_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result;
}

impl ArrayVisitor for Arc<dyn Array> {
    fn children(&self) -> Vec<&dyn Array> {
        self.as_ref().children()
    }

    fn visit_children<'a>(&'a self, visitor: &mut dyn ArrayChildVisitor<'a>) {
        // Delegate to the underlying Array trait implementation
        self.as_ref().visit_children(visitor)
    }

    fn nchildren(&self) -> usize {
        self.as_ref().nchildren()
    }

    fn children_names(&self) -> Vec<String> {
        self.as_ref().children_names()
    }

    fn named_children(&self) -> Vec<(String, ArrayRef)> {
        self.as_ref().named_children()
    }

    fn buffers(&self) -> Vec<ByteBuffer> {
        self.as_ref().buffers()
    }

    fn nbuffers(&self) -> usize {
        self.as_ref().nbuffers()
    }

    fn metadata(&self) -> VortexResult<Option<Vec<u8>>> {
        self.as_ref().metadata()
    }

    fn metadata_fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.as_ref().metadata_fmt(f)
    }
}

pub trait ArrayVisitorExt: Array {
    /// Count the number of buffers encoded by self and all child arrays.
    fn nbuffers_recursive(&self) -> usize {
        self.children()
            .iter()
            .map(|c| c.to_array().nbuffers_recursive())
            .sum::<usize>()
            + self.nbuffers()
    }

    /// Depth-first traversal of the array and its children.
    fn depth_first_traversal(&self) -> impl Iterator<Item = ArrayRef> {
        /// A depth-first pre-order iterator over an Array.
        struct ArrayChildrenIterator {
            stack: Vec<ArrayRef>,
        }

        impl Iterator for ArrayChildrenIterator {
            type Item = ArrayRef;

            fn next(&mut self) -> Option<Self::Item> {
                let next = self.stack.pop()?;
                for child in next.children().into_iter().rev() {
                    self.stack.push(child.to_array());
                }
                Some(next)
            }
        }

        ArrayChildrenIterator {
            stack: vec![self.to_array()],
        }
    }
}

impl<A: Array + ?Sized> ArrayVisitorExt for A {}

pub trait ArrayBufferVisitor {
    fn visit_buffer(&mut self, buffer: &ByteBuffer);
}

pub trait ArrayChildVisitor<'a> {
    /// Visit a child of this array.
    fn visit_child(&mut self, _name: &str, _array: &'a dyn Array);

    /// Utility for visiting Array validity.
    fn visit_validity(&mut self, validity: &'a Validity, len: usize) {
        if let Some(vlen) = validity.maybe_len() {
            assert_eq!(vlen, len, "Validity length mismatch");
        }

        match validity {
            Validity::NonNullable | Validity::AllValid => {}
            Validity::AllInvalid => {
                // To avoid storing metadata about validity, we store all invalid as a
                // constant array of false values.
                // This gives:
                //  * is_nullable & has_validity => Validity::Array (or Validity::AllInvalid)
                //  * is_nullable & !has_validity => Validity::AllValid
                //  * !is_nullable => Validity::NonNullable
                let constant = ConstantArray::new(false, len).to_array();
                // TODO: This leaks memory but fixes the dangling pointer bug that caused SIGSEGV
                let leaked: &'static ArrayRef = Box::leak(Box::new(constant));
                let constant_ref: &'a dyn Array = leaked.as_ref();
                self.visit_child("validity", constant_ref)
            }
            Validity::Array(array) => {
                self.visit_child("validity", array);
            }
        }
    }

    /// Utility for visiting Array patches.
    fn visit_patches(&mut self, patches: &'a Patches) {
        self.visit_child("patch_indices", patches.indices());
        self.visit_child("patch_values", patches.values());
        if let Some(chunk_offsets) = patches.chunk_offsets() {
            self.visit_child("patch_chunk_offsets", chunk_offsets);
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexResult;

    use crate::array::ArrayVisitor;
    use crate::arrays::StructArray;
    use crate::expr::traversal::{
        FoldDown, FoldUp, Node, NodeExt, NodeFolder, NodeVisitor, Transformed, TraversalOrder,
    };
    use crate::{Array, ArrayRef, IntoArray};

    /// Test helper: create a simple struct array with primitive children
    fn create_test_struct() -> ArrayRef {
        StructArray::from_fields(&[
            ("field1", buffer![1i32, 2, 3].into_array()),
            ("field2", buffer![10i64, 20, 30].into_array()),
        ])
        .unwrap()
        .into_array()
    }

    /// Test helper: create a nested struct array
    fn create_nested_struct() -> ArrayRef {
        let inner = create_test_struct();

        StructArray::from_fields(&[
            ("inner", inner),
            ("field3", buffer![100u32, 200, 300].into_array()),
        ])
        .unwrap()
        .into_array()
    }

    #[test]
    fn test_array_node_children_count() {
        let array = create_test_struct();
        assert_eq!(array.children_count(), 2);

        let nested = create_nested_struct();
        assert_eq!(nested.children_count(), 2);
    }

    #[test]
    fn test_array_node_iter_children() {
        let array = create_test_struct();

        let mut count = 0;
        array.iter_children(|children| {
            for child in children {
                count += 1;
                assert_eq!(child.len(), 3);
            }
        });

        assert_eq!(count, 2);
    }

    #[test]
    fn test_array_visitor_basic() {
        struct ArrayChildCounter {
            count: usize,
        }

        impl<'a> NodeVisitor<'a> for ArrayChildCounter {
            type NodeTy = ArrayRef;

            fn visit_down(&mut self, _node: &'a dyn Array) -> VortexResult<TraversalOrder> {
                self.count += 1;
                Ok(TraversalOrder::Continue)
            }
        }

        let array = create_test_struct();
        let mut visitor = ArrayChildCounter { count: 0 };
        array.accept(&mut visitor).unwrap();

        // Should visit: struct + 2 primitive children = 3
        assert_eq!(visitor.count, 3);
    }

    #[test]
    fn test_array_visitor_nested() {
        struct ArrayChildCounter {
            count: usize,
        }

        impl<'a> NodeVisitor<'a> for ArrayChildCounter {
            type NodeTy = ArrayRef;

            fn visit_down(&mut self, _node: &'a dyn Array) -> VortexResult<TraversalOrder> {
                self.count += 1;
                Ok(TraversalOrder::Continue)
            }
        }

        let array = create_nested_struct();
        let mut visitor = ArrayChildCounter { count: 0 };
        array.accept(&mut visitor).unwrap();

        // Should visit: outer struct + inner struct + 2 inner fields + 1 outer field = 5
        assert_eq!(visitor.count, 5);
    }

    #[test]
    fn test_array_visitor_complex_traversal_order() {
        // Create a complex nested structure to test Continue/Skip/Stop traversal
        //
        // Key insight: NodeVisitor visits NODES (arrays), not individual fields.
        // When we visit a struct node, we see ALL its field names, then decide
        // whether to Continue (visit children), Skip (don't visit children), or Stop (halt).
        //
        // Structure:
        //   root {                           // fields: "normal1", "normal2", "nested_skip", "nested_continue"
        //     "normal1" -> [1, 2, 3]         // field name has 'a' → count(1)
        //     "normal2" -> [4, 5, 6]         // field name has 'a' → count(2)
        //     "nested_skip" -> {             // field name has no skip trigger
        //       "has_skip" -> [7, 8, 9]      // when we visit this struct, we see "has_skip" with 's' → Skip children
        //       "child_a" -> {               // has 'a' but we'll Skip before visiting its children
        //         "deep_a" -> [10, 11, 12]   // would be counted but skipped
        //       }
        //     }
        //     "nested_continue" -> {         // field name has 'a' (continue) → eventually count(3)
        //       "inner_alpha" -> [13, 14, 15] // has 'a' → count(4)
        //       "inner_beta" -> [16, 17, 18]  // has 'a' → count(5)
        //       "nested_stop" -> {            // continue to visit
        //         "has_stop" -> [19, 20, 21]  // when we visit this struct, we see "has_stop" with 'x' → Stop
        //         "never_seen" -> [22, 23, 24] // never visited due to Stop
        //       }
        //     }
        //   }
        //
        // Traversal flow:
        // 1. Visit root struct → see fields: normal1(a), normal2(a), nested_skip, nested_continue
        //    → count normal1(1), normal2(2) → Continue
        // 2. Visit normal1 (primitive) → Continue
        // 3. Visit normal2 (primitive) → Continue
        // 4. Visit nested_skip struct → see fields: has_skip(a), child_a(a)
        //    → count has_skip(3), child_a(4) → Skip children (deep_a not visited)
        // 5. Visit nested_continue struct → see fields: inner_alpha(a), inner_beta(a), nested_stop
        //    → count inner_alpha(5), inner_beta(6) → Continue
        // 6. Visit inner_alpha (primitive) → Continue
        // 7. Visit inner_beta (primitive) → Continue
        // 8. Visit nested_stop struct → see fields: has_stop(a), never_seen
        //    → count has_stop(7) → Stop (halts further traversal)
        //
        // Expected count: 7 fields containing 'a':
        //   normal1, normal2, has_skip, child_a, inner_alpha, inner_beta, has_stop

        let normal1 = buffer![1i32, 2, 3].into_array();
        let normal2 = buffer![4i32, 5, 6].into_array();

        // Deep child that would be skipped
        let deep_a = buffer![10i32, 11, 12].into_array();
        let child_a = StructArray::from_fields(&[("deep_a", deep_a)])
            .unwrap()
            .into_array();

        let has_skip_arr = buffer![7i32, 8, 9].into_array();
        let nested_skip =
            StructArray::from_fields(&[("has_skip", has_skip_arr), ("child_a", child_a)])
                .unwrap()
                .into_array();

        let inner_alpha = buffer![13i32, 14, 15].into_array();
        let inner_beta = buffer![16i32, 17, 18].into_array();

        let has_stop_arr = buffer![19i32, 20, 21].into_array();
        let never_seen = buffer![22i32, 23, 24].into_array();
        let nested_stop =
            StructArray::from_fields(&[("has_stop", has_stop_arr), ("never_seen", never_seen)])
                .unwrap()
                .into_array();

        let nested_continue = StructArray::from_fields(&[
            ("inner_alpha", inner_alpha),
            ("inner_beta", inner_beta),
            ("nested_stop", nested_stop),
        ])
        .unwrap()
        .into_array();

        let root = StructArray::from_fields(&[
            ("normal1", normal1),
            ("normal2", normal2),
            ("nested_skip", nested_skip),
            ("nested_continue", nested_continue),
        ])
        .unwrap()
        .into_array();

        struct ComplexVisitor {
            count: usize,
            visited_names: Vec<String>,
        }

        impl<'a> NodeVisitor<'a> for ComplexVisitor {
            type NodeTy = ArrayRef;

            fn visit_down(&mut self, node: &'a dyn Array) -> VortexResult<TraversalOrder> {
                // If it's a struct, check field names
                if let Some(struct_arr) = node.as_opt::<crate::arrays::StructVTable>() {
                    let field_names = struct_arr.names();
                    let mut should_skip = false;
                    let mut should_stop = false;

                    // Process all field names first, counting and checking for triggers
                    for field_name in field_names.iter() {
                        let name = field_name.as_ref();
                        self.visited_names.push(name.to_string());

                        // Count fields with 'a' in the name
                        if name.contains('a') {
                            self.count += 1;
                        }

                        // Check for skip trigger using exact match
                        // Only the field "has_skip" triggers Skip
                        if name == "has_skip" {
                            should_skip = true;
                        }

                        // Check for stop trigger using exact match
                        // Only the field "has_stop" triggers Stop
                        if name == "has_stop" {
                            should_stop = true;
                        }
                    }

                    // After processing all fields, return the appropriate order
                    if should_stop {
                        return Ok(TraversalOrder::Stop);
                    }
                    if should_skip {
                        return Ok(TraversalOrder::Skip);
                    }
                }

                Ok(TraversalOrder::Continue)
            }
        }

        let mut visitor = ComplexVisitor {
            count: 0,
            visited_names: Vec::new(),
        };
        root.accept(&mut visitor).unwrap();

        // Verify count of field names containing 'a':
        // - root: normal1(1), normal2(2) = 2
        // - nested_skip: has_skip(3), child_a(4) = 4 → Skip (prevents visiting child_a's children)
        // - nested_continue: inner_alpha(5), inner_beta(6) = 6
        // - nested_stop: has_stop(7) = 7 → Stop (prevents further traversal after this struct)
        assert_eq!(visitor.count, 7);

        // Verify visited names (in traversal order)
        assert_eq!(
            visitor.visited_names,
            vec![
                // Root struct fields
                "normal1",
                "normal2",
                "nested_skip",
                "nested_continue",
                // nested_skip struct fields (then Skip, so child_a's children not visited)
                "has_skip",
                "child_a",
                // nested_continue struct fields
                "inner_alpha",
                "inner_beta",
                "nested_stop",
                // nested_stop struct fields (then Stop)
                "has_stop",
                "never_seen", // Processed as field name but Stop prevents further traversal
            ]
        );

        // Verify deep_a was never visited (due to Skip on nested_skip)
        assert!(!visitor.visited_names.contains(&"deep_a".to_string()));
    }

    #[test]
    fn test_array_transform_up() {
        let array = create_test_struct();

        // Count how many arrays were transformed
        let mut transform_count = 0;

        let result = array
            .transform_up(|arr: ArrayRef| {
                transform_count += 1;
                Ok(Transformed::no(arr))
            })
            .unwrap();

        // Should visit all 3 arrays (1 struct + 2 primitives)
        assert_eq!(transform_count, 3);
        assert!(!result.changed);
    }

    #[test]
    fn test_array_transform_down() {
        let array = create_test_struct();

        // Count how many arrays were visited during down pass
        let mut transform_count = 0;

        let result = array
            .transform_down(|arr: ArrayRef| {
                transform_count += 1;
                Ok(Transformed::no(arr))
            })
            .unwrap();

        // Should visit all 3 arrays
        assert_eq!(transform_count, 3);
        assert!(!result.changed);
    }

    #[test]
    fn test_array_transform_change_detection() {
        let array = create_test_struct();

        // Transform leaf arrays only
        let result = array
            .transform_up(|arr: ArrayRef| {
                if arr.nchildren() == 0 {
                    // Leaf node - slice it to create a change
                    Ok(Transformed::yes(arr.slice(0..arr.len())))
                } else {
                    Ok(Transformed::no(arr))
                }
            })
            .unwrap();

        // Should detect change because children changed
        assert!(result.changed);
    }

    #[test]
    fn test_array_fold_count() {
        struct ArrayCounter;

        impl NodeFolder for ArrayCounter {
            type NodeTy = ArrayRef;
            type Result = usize;

            fn visit_up(
                &mut self,
                _node: Self::NodeTy,
                children: Vec<Self::Result>,
            ) -> VortexResult<FoldUp<Self::Result>> {
                Ok(FoldUp::Continue(1 + children.iter().sum::<usize>()))
            }
        }

        let array = create_test_struct();
        let mut folder = ArrayCounter;
        let count = array.fold(&mut folder).unwrap().value();

        // 1 struct + 2 primitive arrays = 3
        assert_eq!(count, 3);
    }

    #[test]
    fn test_array_fold_nested_count() {
        struct ArrayCounter;

        impl NodeFolder for ArrayCounter {
            type NodeTy = ArrayRef;
            type Result = usize;

            fn visit_up(
                &mut self,
                _node: Self::NodeTy,
                children: Vec<Self::Result>,
            ) -> VortexResult<FoldUp<Self::Result>> {
                Ok(FoldUp::Continue(1 + children.iter().sum::<usize>()))
            }
        }

        let array = create_nested_struct();
        let mut folder = ArrayCounter;
        let count = array.fold(&mut folder).unwrap().value();

        // outer struct + inner struct + 2 inner primitives + 1 outer primitive = 5
        assert_eq!(count, 5);
    }

    #[test]
    fn test_array_fold_stop() {
        struct StopAtStruct;

        impl NodeFolder for StopAtStruct {
            type NodeTy = ArrayRef;
            type Result = usize;

            fn visit_down(&mut self, node: &Self::NodeTy) -> VortexResult<FoldDown<Self::Result>> {
                if node.is::<crate::arrays::StructVTable>() && node.nchildren() == 2 {
                    // Stop at first struct with 2 children
                    Ok(FoldDown::Stop(999))
                } else {
                    Ok(FoldDown::Continue)
                }
            }

            fn visit_up(
                &mut self,
                _node: Self::NodeTy,
                children: Vec<Self::Result>,
            ) -> VortexResult<FoldUp<Self::Result>> {
                Ok(FoldUp::Continue(1 + children.iter().sum::<usize>()))
            }
        }

        let array = create_nested_struct();
        let mut folder = StopAtStruct;
        let count = array.fold(&mut folder).unwrap().value();

        // Should stop at inner struct
        assert_eq!(count, 999);
    }

    #[test]
    fn test_array_visitor_skip() {
        struct SkipPrimitiveChildren {
            visited: Vec<String>,
        }

        impl<'a> NodeVisitor<'a> for SkipPrimitiveChildren {
            type NodeTy = ArrayRef;

            fn visit_down(&mut self, node: &'a dyn Array) -> VortexResult<TraversalOrder> {
                self.visited.push(format!("{}", node.encoding().id()));

                // Skip children of the first primitive array we encounter
                if node.is::<crate::arrays::PrimitiveVTable>()
                    && self
                        .visited
                        .iter()
                        .filter(|v| v.contains("primitive"))
                        .count()
                        == 1
                {
                    return Ok(TraversalOrder::Skip);
                }

                Ok(TraversalOrder::Continue)
            }
        }

        let array = create_test_struct();
        let mut visitor = SkipPrimitiveChildren {
            visited: Vec::new(),
        };
        array.accept(&mut visitor).unwrap();

        // Should visit: struct + 2 primitive arrays = 3 total
        // Skip doesn't prevent siblings from being visited, only children
        assert_eq!(visitor.visited.len(), 3);
    }

    #[test]
    fn test_array_visitor_stop() {
        struct StopAtSecondArray {
            count: usize,
        }

        impl<'a> NodeVisitor<'a> for StopAtSecondArray {
            type NodeTy = ArrayRef;

            fn visit_down(&mut self, _node: &'a dyn Array) -> VortexResult<TraversalOrder> {
                self.count += 1;
                if self.count >= 2 {
                    Ok(TraversalOrder::Stop)
                } else {
                    Ok(TraversalOrder::Continue)
                }
            }
        }

        let array = create_test_struct();
        let mut visitor = StopAtSecondArray { count: 0 };
        array.accept(&mut visitor).unwrap();

        // Should stop after 2nd visit
        assert_eq!(visitor.count, 2);
    }

    #[test]
    fn test_array_visit_order() {
        // Verify pre-order down, post-order up traversal
        let array = create_nested_struct();

        struct OrderTracker {
            order: Vec<String>,
        }

        impl<'a> NodeVisitor<'a> for OrderTracker {
            type NodeTy = ArrayRef;

            fn visit_down(&mut self, node: &'a dyn Array) -> VortexResult<TraversalOrder> {
                self.order.push(format!("down:{}", node.encoding().id()));
                Ok(TraversalOrder::Continue)
            }

            fn visit_up(&mut self, node: &'a dyn Array) -> VortexResult<TraversalOrder> {
                self.order.push(format!("up:{}", node.encoding().id()));
                Ok(TraversalOrder::Continue)
            }
        }

        let mut visitor = OrderTracker { order: Vec::new() };
        array.accept(&mut visitor).unwrap();

        // Verify we visited all 5 nodes (outer struct + inner struct + 2 inner primitives + 1 outer primitive)
        // Each node should have both down and up
        assert_eq!(visitor.order.len(), 10); // 5 nodes * 2 (down + up)

        // Verify first visit is down and last is up for root
        assert_eq!(visitor.order[0], "down:vortex.struct");
        assert_eq!(visitor.order[visitor.order.len() - 1], "up:vortex.struct");

        // Verify every down has a corresponding up (pre-order down, post-order up)
        let down_count = visitor
            .order
            .iter()
            .filter(|s| s.starts_with("down:"))
            .count();
        let up_count = visitor
            .order
            .iter()
            .filter(|s| s.starts_with("up:"))
            .count();
        assert_eq!(down_count, up_count);
    }

    #[test]
    fn test_array_both_down_and_up_called() {
        // Ensure visit_down and visit_up are both called for every node
        let array = create_test_struct();

        struct UpDownCounter {
            down_count: usize,
            up_count: usize,
        }

        impl<'a> NodeVisitor<'a> for UpDownCounter {
            type NodeTy = ArrayRef;

            fn visit_down(&mut self, _node: &'a dyn Array) -> VortexResult<TraversalOrder> {
                self.down_count += 1;
                Ok(TraversalOrder::Continue)
            }

            fn visit_up(&mut self, _node: &'a dyn Array) -> VortexResult<TraversalOrder> {
                self.up_count += 1;
                Ok(TraversalOrder::Continue)
            }
        }

        let mut visitor = UpDownCounter {
            down_count: 0,
            up_count: 0,
        };
        array.accept(&mut visitor).unwrap();

        // Every node should have both down and up called
        assert_eq!(visitor.down_count, visitor.up_count);
        assert_eq!(visitor.down_count, 3); // struct + 2 primitives
    }

    #[test]
    fn test_array_no_duplicate_visits() {
        // Ensure each node is visited exactly once
        use vortex_utils::aliases::hash_set::HashSet;

        let array = create_nested_struct();

        struct DuplicateDetector {
            visited: HashSet<*const dyn Array>,
        }

        impl<'a> NodeVisitor<'a> for DuplicateDetector {
            type NodeTy = ArrayRef;

            fn visit_down(&mut self, node: &'a dyn Array) -> VortexResult<TraversalOrder> {
                let ptr = node as *const dyn Array;
                assert!(!self.visited.contains(&ptr), "Node visited twice!");
                self.visited.insert(ptr);
                Ok(TraversalOrder::Continue)
            }
        }

        let mut visitor = DuplicateDetector {
            visited: HashSet::new(),
        };
        array.accept(&mut visitor).unwrap();

        // Should have visited 5 unique nodes
        assert_eq!(visitor.visited.len(), 5);
    }

    #[test]
    fn test_array_deep_recursion() {
        // Test deeply nested structure to ensure recursion works
        use crate::arrays::StructArray;

        // Create deeply nested structure (10 levels)
        let mut array = buffer![1i32, 2, 3].into_array();

        for i in 0..10 {
            array = StructArray::from_fields(&[(format!("field{}", i).as_str(), array)])
                .unwrap()
                .into_array();
        }

        struct DepthTracker {
            current_depth: usize,
            max_depth: usize,
        }

        impl<'a> NodeVisitor<'a> for DepthTracker {
            type NodeTy = ArrayRef;

            fn visit_down(&mut self, _node: &'a dyn Array) -> VortexResult<TraversalOrder> {
                self.current_depth += 1;
                self.max_depth = self.max_depth.max(self.current_depth);
                Ok(TraversalOrder::Continue)
            }

            fn visit_up(&mut self, _node: &'a dyn Array) -> VortexResult<TraversalOrder> {
                self.current_depth -= 1;
                Ok(TraversalOrder::Continue)
            }
        }

        let mut visitor = DepthTracker {
            current_depth: 0,
            max_depth: 0,
        };
        array.accept(&mut visitor).unwrap();

        assert_eq!(visitor.max_depth, 11); // 10 structs + 1 primitive
        assert_eq!(visitor.current_depth, 0); // Should be back to 0 after traversal
    }

    #[test]
    fn test_array_transform_with_reconstruction() {
        let array = create_test_struct();
        let original_len = array.len();
        let original_nchildren = array.nchildren();

        // Track that we actually transformed something
        let mut transformed_count = 0;

        // Identity transform on all children - still requires reconstruction
        let result = array
            .transform_up(|arr: ArrayRef| {
                if arr.nchildren() == 0 {
                    transformed_count += 1;
                    // Leaf node - return a sliced version (but full length to maintain validity)
                    Ok(Transformed::yes(arr.slice(0..arr.len())))
                } else {
                    Ok(Transformed::no(arr))
                }
            })
            .unwrap();

        // Should have transformed the leaf nodes
        assert_eq!(transformed_count, 2);
        assert!(result.changed);
        // Structure should be preserved
        assert_eq!(result.value.len(), original_len);
        assert_eq!(result.value.nchildren(), original_nchildren);
    }

    #[test]
    fn test_array_transform_add_to_primitives() {
        use vortex_dtype::PType;

        use crate::arrays::{PrimitiveArray, PrimitiveVTable};
        use crate::validity::Validity;

        let array = create_test_struct();
        // Original: field1: [1i32, 2, 3], field2: [10i64, 20, 30]

        println!("Original array:\n{}", array.display_tree());
        println!("Original values:\n{}", array.display_values());

        // Transform: +1 to i32 arrays, +100 to i64 arrays
        let result = array
            .transform_up(|arr: ArrayRef| {
                if arr.is::<PrimitiveVTable>() {
                    let prim = arr.as_::<PrimitiveVTable>();
                    match prim.ptype() {
                        PType::I32 => {
                            // Add 1 to all i32 values
                            let values: Vec<i32> =
                                prim.as_slice::<i32>().iter().map(|&v| v + 1).collect();
                            let buffer = vortex_buffer::Buffer::from(values);
                            Ok(Transformed::yes(
                                PrimitiveArray::new(buffer, Validity::NonNullable).into_array(),
                            ))
                        }
                        PType::I64 => {
                            // Add 100 to all i64 values
                            let values: Vec<i64> =
                                prim.as_slice::<i64>().iter().map(|&v| v + 100).collect();
                            let buffer = vortex_buffer::Buffer::from(values);
                            Ok(Transformed::yes(
                                PrimitiveArray::new(buffer, Validity::NonNullable).into_array(),
                            ))
                        }
                        _ => Ok(Transformed::no(arr)),
                    }
                } else {
                    Ok(Transformed::no(arr))
                }
            })
            .unwrap();

        assert!(result.changed);
        assert_eq!(result.value.len(), 3);

        println!("\nTransformed array:\n{}", result.value.display_tree());
        println!("Transformed values:\n{}", result.value.display_values());

        // Verify the transformations
        let children = result.value.children();

        // field1 should be [2, 3, 4] (original [1, 2, 3] + 1)
        let field1 = children[0].as_::<PrimitiveVTable>();
        assert_eq!(field1.as_slice::<i32>(), &[2, 3, 4]);

        // field2 should be [110, 120, 130] (original [10, 20, 30] + 100)
        let field2 = children[1].as_::<PrimitiveVTable>();
        assert_eq!(field2.as_slice::<i64>(), &[110, 120, 130]);
    }

    #[test]
    fn test_array_transform_canonicalize_all() {
        let array = create_test_struct();

        // Convert all arrays to canonical form
        let result = array
            .transform_up(|arr: ArrayRef| {
                if !arr.is_canonical() {
                    Ok(Transformed::yes(arr.to_canonical().into_array()))
                } else {
                    Ok(Transformed::no(arr))
                }
            })
            .unwrap();

        // Should have transformed something (struct at least)
        // All arrays in result should be canonical
        fn check_all_canonical(arr: &ArrayRef) -> bool {
            if !arr.is_canonical() {
                return false;
            }
            arr.children()
                .iter()
                .all(|c| check_all_canonical(&c.to_array()))
        }

        assert!(check_all_canonical(&result.value));
    }

    #[test]
    fn test_array_transform_count_primitives() {
        use crate::arrays::PrimitiveVTable;

        let array = create_nested_struct();

        // Count primitive arrays using transformation
        let primitive_count = array.fold(&mut PrimitiveCounter).unwrap().value();

        // Nested struct has: 2 primitives in inner struct + 1 primitive in outer = 3
        assert_eq!(primitive_count, 3);

        struct PrimitiveCounter;
        impl NodeFolder for PrimitiveCounter {
            type NodeTy = ArrayRef;
            type Result = usize;

            fn visit_up(
                &mut self,
                node: Self::NodeTy,
                children: Vec<Self::Result>,
            ) -> VortexResult<FoldUp<Self::Result>> {
                let is_primitive = if node.is::<PrimitiveVTable>() { 1 } else { 0 };
                Ok(FoldUp::Continue(
                    is_primitive + children.iter().sum::<usize>(),
                ))
            }
        }
    }

    #[test]
    fn test_array_transform_replace_encoding() {
        use crate::arrays::{ConstantArray, PrimitiveVTable};

        let array = create_test_struct();

        // Replace all primitive arrays with constant arrays
        let result = array
            .transform_up(|arr: ArrayRef| {
                if arr.is::<PrimitiveVTable>() && !arr.is_empty() {
                    // Replace with constant array of first value
                    let first_value = arr.scalar_at(0);
                    let constant = ConstantArray::new(first_value, arr.len());
                    Ok(Transformed::yes(constant.into_array()))
                } else {
                    Ok(Transformed::no(arr))
                }
            })
            .unwrap();

        assert!(result.changed);

        // Verify all leaf arrays are now constant
        fn check_no_primitives(arr: &ArrayRef) -> bool {
            if arr.is::<PrimitiveVTable>() {
                return false;
            }
            arr.children()
                .iter()
                .all(|c| check_no_primitives(&c.to_array()))
        }

        assert!(check_no_primitives(&result.value));
    }
}
