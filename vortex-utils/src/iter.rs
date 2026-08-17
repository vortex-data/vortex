// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Iterator extension traits.

/// An extension trait for iterators that provides balanced binary tree reduction.
///
/// Unlike [`Iterator::reduce`], which builds a left-leaning linear chain of depth N,
/// `reduce_balanced` builds a balanced binary tree of depth log(N). This avoids deep
/// nesting that can cause stack overflows on drop or suboptimal evaluation.
///
/// ```text
/// reduce:          reduce_balanced:
///     f                  f
///    / \                / \
///   f   d              f   f
///  / \                / \ / \
/// f   c              a  b c  d
/// |\
/// a b
/// ```
pub trait ReduceBalancedIterExt: Iterator {
    /// Like [`Iterator::reduce`], but builds a balanced binary tree instead of a linear chain.
    ///
    /// `[a, b, c, d]` becomes `combine(combine(a, b), combine(c, d))`.
    ///
    /// Returns `None` if the iterator is empty.
    fn reduce_balanced<F>(self, combine: F) -> Option<Self::Item>
    where
        F: Fn(Self::Item, Self::Item) -> Self::Item;

    /// Fallible version of [`reduce_balanced`](ReduceBalancedIterExt::reduce_balanced).
    ///
    /// Short-circuits on the first error.
    fn try_reduce_balanced<F, E>(self, combine: F) -> Result<Option<Self::Item>, E>
    where
        F: Fn(Self::Item, Self::Item) -> Result<Self::Item, E>;
}

impl<I: Iterator + Sized> ReduceBalancedIterExt for I {
    fn reduce_balanced<F>(self, combine: F) -> Option<Self::Item>
    where
        F: Fn(Self::Item, Self::Item) -> Self::Item,
    {
        let mut items: Vec<_> = self.collect();
        if items.is_empty() {
            return None;
        }
        if items.len() == 1 {
            return items.pop();
        }

        let mut next = Vec::with_capacity(items.len() / 2);
        while items.len() > 1 {
            next.clear();
            let mut iter = items.drain(..);
            while let Some(lhs) = iter.next() {
                if let Some(rhs) = iter.next() {
                    next.push(combine(lhs, rhs));
                } else {
                    // Merge the odd element into the last paired element so it stays inside the
                    // tree.
                    let Some(previous) = next.pop() else {
                        unreachable!("a reduction level with an odd tail has a preceding pair")
                    };
                    next.push(combine(previous, lhs));
                }
            }
            drop(iter);
            std::mem::swap(&mut items, &mut next);
        }

        assert_eq!(items.len(), 1);
        items.pop()
    }

    fn try_reduce_balanced<F, E>(self, combine: F) -> Result<Option<Self::Item>, E>
    where
        F: Fn(Self::Item, Self::Item) -> Result<Self::Item, E>,
    {
        let mut items: Vec<_> = self.collect();
        if items.is_empty() {
            return Ok(None);
        }
        if items.len() == 1 {
            return Ok(items.pop());
        }

        let mut next = Vec::with_capacity(items.len() / 2);
        while items.len() > 1 {
            next.clear();
            let mut iter = items.drain(..);
            while let Some(lhs) = iter.next() {
                if let Some(rhs) = iter.next() {
                    next.push(combine(lhs, rhs)?);
                } else {
                    let Some(previous) = next.pop() else {
                        unreachable!("a reduction level with an odd tail has a preceding pair")
                    };
                    next.push(combine(previous, lhs)?);
                }
            }
            drop(iter);
            std::mem::swap(&mut items, &mut next);
        }

        assert_eq!(items.len(), 1);
        Ok(items.pop())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        let result = std::iter::empty::<i32>().reduce_balanced(|a, b| a + b);
        assert_eq!(result, None);
    }

    #[test]
    fn test_single() {
        let result = [42].into_iter().reduce_balanced(|a, b| a + b);
        assert_eq!(result, Some(42));
    }

    #[test]
    fn test_two() {
        let result = [1, 2].into_iter().reduce_balanced(|a, b| a + b);
        assert_eq!(result, Some(3));
    }

    #[test]
    fn test_power_of_two() {
        let result = [1, 2, 3, 4].into_iter().reduce_balanced(|a, b| a + b);
        assert_eq!(result, Some(10));
    }

    #[test]
    fn test_odd_count() {
        let result = [1, 2, 3, 4, 5].into_iter().reduce_balanced(|a, b| a + b);
        assert_eq!(result, Some(15));
    }

    #[test]
    fn test_balanced_structure() {
        // Use string concatenation to verify the tree shape.
        // [a, b, c, d] should produce ((a+b)+(c+d)), not (((a+b)+c)+d).
        let result = ["a", "b", "c", "d"]
            .into_iter()
            .map(String::from)
            .reduce_balanced(|a, b| format!("({a}+{b})"));
        assert_eq!(result, Some("((a+b)+(c+d))".to_string()));
    }

    #[test]
    fn test_balanced_structure_odd() {
        // [a, b, c] should produce ((a+b)+c) — odd element merges into last pair.
        let result = ["a", "b", "c"]
            .into_iter()
            .map(String::from)
            .reduce_balanced(|a, b| format!("({a}+{b})"));
        assert_eq!(result, Some("((a+b)+c)".to_string()));
    }

    #[test]
    fn test_balanced_structure_five() {
        // [a, b, c, d, e] => ((a+b)+((c+d)+e))
        let result = ["a", "b", "c", "d", "e"]
            .into_iter()
            .map(String::from)
            .reduce_balanced(|a, b| format!("({a}+{b})"));
        assert_eq!(result, Some("((a+b)+((c+d)+e))".to_string()));
    }

    #[test]
    fn test_non_clone_items() {
        #[derive(Debug, PartialEq, Eq)]
        struct NonClone(String);

        let result = ["a", "b", "c"]
            .into_iter()
            .map(|value| NonClone(value.to_owned()))
            .reduce_balanced(|NonClone(lhs), NonClone(rhs)| NonClone(format!("({lhs}+{rhs})")));

        assert_eq!(result, Some(NonClone("((a+b)+c)".to_owned())));
    }

    #[test]
    fn test_try_reduce_balanced_ok() {
        let result: Result<_, &str> = [1, 2, 3, 4]
            .into_iter()
            .try_reduce_balanced(|a, b| Ok(a + b));
        assert_eq!(result, Ok(Some(10)));
    }

    #[test]
    fn test_try_reduce_balanced_err() {
        let result: Result<Option<i32>, &str> = [1, 2, 3, 4]
            .into_iter()
            .try_reduce_balanced(|a, b| if a + b > 4 { Err("too big") } else { Ok(a + b) });
        assert_eq!(result, Err("too big"));
    }

    #[test]
    fn test_try_reduce_balanced_empty() {
        let result: Result<_, &str> =
            std::iter::empty::<i32>().try_reduce_balanced(|a, b| Ok(a + b));
        assert_eq!(result, Ok(None));
    }

    #[test]
    fn test_try_reduce_balanced_single() {
        let result: Result<_, &str> = [42].into_iter().try_reduce_balanced(|a, b| Ok(a + b));
        assert_eq!(result, Ok(Some(42)));
    }
}
