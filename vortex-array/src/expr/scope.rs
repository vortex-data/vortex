// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::dtype::DType;
use crate::expr::variable::Variable;

/// A set of named bindings introduced by a single binder.
///
/// A lambda pushes exactly one frame, holding its parameters. Names within a frame must be unique;
/// a name in an inner frame shadows the same name in an outer one.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Frame(Arc<Vec<(Variable, DType)>>);

impl Frame {
    /// Create a frame from its bindings, in declaration order.
    ///
    /// Errors if a name is bound more than once, since a reference to it would be ambiguous.
    pub fn try_new(bindings: impl IntoIterator<Item = (Variable, DType)>) -> VortexResult<Self> {
        let bindings = Vec::from_iter(bindings);
        for (idx, (name, _)) in bindings.iter().enumerate() {
            if bindings[..idx].iter().any(|(seen, _)| seen == name) {
                vortex_bail!("duplicate binding '{name}' in a single frame");
            }
        }
        Ok(Self(Arc::new(bindings)))
    }

    /// The bindings in this frame, in declaration order.
    pub fn bindings(&self) -> &[(Variable, DType)] {
        &self.0
    }

    /// The dtype bound to `name` in this frame, if any.
    pub fn get(&self, name: &Variable) -> Option<&DType> {
        self.0
            .iter()
            .find_map(|(bound, dtype)| (bound == name).then_some(dtype))
    }
}

/// The context an [`Expression`](crate::expr::Expression) is bound against.
///
/// A scope is the dtype that [`root`](crate::expr::root) resolves to, plus a stack of [`Frame`]s
/// holding named bindings. The stack is what distinguishes "this binder's own name" from "a name
/// introduced by an enclosing binder", which is the information a capture check needs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scope {
    root: DType,
    /// Innermost frame last.
    frames: Vec<Frame>,
}

impl Scope {
    /// Create a scope in which `root` resolves to the given dtype, with no bindings.
    pub fn new(root: DType) -> Self {
        Self {
            root,
            frames: Vec::new(),
        }
    }

    /// The dtype that `root` resolves to.
    pub fn root(&self) -> &DType {
        &self.root
    }

    /// The number of enclosing binders.
    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    /// Return this scope extended with one more frame, which becomes the innermost.
    pub fn push_frame(&self, frame: Frame) -> Self {
        let mut frames = self.frames.clone();
        frames.push(frame);
        Self {
            root: self.root.clone(),
            frames,
        }
    }

    /// Resolve `name`, searching innermost-first so that inner bindings shadow outer ones.
    ///
    /// Returns the bound dtype and the index of the frame it was found in, counted from the
    /// outermost so the value stays stable as further frames are pushed. Comparing it against the
    /// [`depth`](Scope::depth) at a binder distinguishes a reference to that binder's own parameter
    /// from a capture of something further out.
    pub fn resolve(&self, name: &Variable) -> Option<(&DType, usize)> {
        self.frames
            .iter()
            .enumerate()
            .rev()
            .find_map(|(depth, frame)| frame.get(name).map(|dtype| (dtype, depth)))
    }
}

impl From<DType> for Scope {
    fn from(root: DType) -> Self {
        Self::new(root)
    }
}

impl From<&DType> for Scope {
    fn from(root: &DType) -> Self {
        Self::new(root.clone())
    }
}

impl From<&Scope> for Scope {
    fn from(scope: &Scope) -> Self {
        scope.clone()
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::dtype::Nullability;
    use crate::dtype::PType;

    fn i32_() -> DType {
        DType::Primitive(PType::I32, Nullability::NonNullable)
    }

    fn utf8() -> DType {
        DType::Utf8(Nullability::NonNullable)
    }

    #[test]
    fn root_round_trips() {
        let dtype = DType::Bool(Nullability::Nullable);
        assert_eq!(Scope::new(dtype.clone()).root(), &dtype);
        assert_eq!(Scope::from(dtype.clone()).root(), &dtype);
    }

    #[test]
    fn an_empty_scope_resolves_nothing() {
        let scope = Scope::new(i32_());
        assert_eq!(scope.depth(), 0);
        assert!(scope.resolve(&Variable::new("x")).is_none());
    }

    #[test]
    fn resolve_reports_the_frame_it_was_found_in() -> VortexResult<()> {
        let scope = Scope::new(i32_())
            .push_frame(Frame::try_new([(Variable::new("a"), i32_())])?)
            .push_frame(Frame::try_new([(Variable::new("b"), utf8())])?);

        assert_eq!(scope.resolve(&Variable::new("a")), Some((&i32_(), 0)));
        assert_eq!(scope.resolve(&Variable::new("b")), Some((&utf8(), 1)));
        assert_eq!(scope.depth(), 2);
        Ok(())
    }

    #[test]
    fn inner_frames_shadow_outer_ones() -> VortexResult<()> {
        let scope = Scope::new(i32_())
            .push_frame(Frame::try_new([(Variable::new("x"), i32_())])?)
            .push_frame(Frame::try_new([(Variable::new("x"), utf8())])?);

        assert_eq!(scope.resolve(&Variable::new("x")), Some((&utf8(), 1)));
        Ok(())
    }

    /// A name bound twice in one frame is ambiguous, so it is rejected rather than silently
    /// resolved by position.
    #[test]
    fn duplicate_names_in_one_frame_are_rejected() {
        assert!(
            Frame::try_new([(Variable::new("x"), i32_()), (Variable::new("x"), utf8())]).is_err()
        );
    }

    #[test]
    fn pushing_a_frame_does_not_mutate_the_original() -> VortexResult<()> {
        let outer = Scope::new(i32_());
        let inner = outer.push_frame(Frame::try_new([(Variable::new("x"), i32_())])?);

        assert!(outer.resolve(&Variable::new("x")).is_none());
        assert!(inner.resolve(&Variable::new("x")).is_some());
        Ok(())
    }
}
