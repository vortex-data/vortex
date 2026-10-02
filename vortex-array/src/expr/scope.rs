// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_utils::aliases::hash_set::HashSet;

use crate::dtype::DType;
use crate::expr::Variable;

/// A stable location for a variable resolved in a lexical [`Scope`].
///
/// Frames are indexed from the outermost binding and slots are indexed in declaration order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VariableRef {
    frame: usize,
    slot: usize,
}

impl VariableRef {
    fn new(frame: usize, slot: usize) -> Self {
        Self { frame, slot }
    }

    /// The lexical frame containing this binding, counted from the outermost frame.
    pub fn frame(&self) -> usize {
        self.frame
    }

    /// The binding's position in its frame, in declaration order.
    pub fn slot(&self) -> usize {
        self.slot
    }
}

/// A set of named bindings introduced by a single binder.
///
/// Names within a frame must be unique. A name in an inner frame shadows the same name in an
/// outer frame.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Frame(Arc<Vec<(Variable, DType)>>);

impl Frame {
    /// Create a frame from its bindings, in declaration order.
    pub fn try_new(bindings: impl IntoIterator<Item = (Variable, DType)>) -> VortexResult<Self> {
        let mut seen = HashSet::new();
        let mut frame = Vec::new();
        for (variable, dtype) in bindings {
            if !seen.insert(variable.clone()) {
                vortex_bail!("duplicate binding '{variable}' in a single frame");
            }
            frame.push((variable, dtype));
        }
        Ok(Self(Arc::new(frame)))
    }

    /// The bindings in this frame, in declaration order.
    pub fn bindings(&self) -> &[(Variable, DType)] {
        &self.0
    }

    fn get(&self, name: &Variable) -> Option<(&DType, usize)> {
        self.0
            .iter()
            .enumerate()
            .find_map(|(slot, (bound, dtype))| (bound == name).then_some((dtype, slot)))
    }
}

/// The context an [`Expression`](crate::expr::Expression) is bound against.
///
/// A scope is the dtype that [`root`](crate::expr::root) resolves to, plus a stack of [`Frame`]s
/// holding named bindings. Names resolve from the innermost frame outward.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scope {
    root: DType,
    frames: Vec<Frame>,
}

impl Scope {
    /// Create a scope in which `root` resolves to the given dtype.
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

    /// The number of lexical frames in this scope.
    pub fn depth(&self) -> usize {
        self.frames.len()
    }

    /// Return this scope extended with `frame` as its innermost frame.
    pub fn push_frame(&self, frame: Frame) -> Self {
        let mut frames = self.frames.clone();
        frames.push(frame);
        Self {
            root: self.root.clone(),
            frames,
        }
    }

    /// Return this scope extended with one frame containing `bindings`.
    pub fn with_bindings(
        self,
        bindings: impl IntoIterator<Item = (Variable, DType)>,
    ) -> VortexResult<Self> {
        Ok(self.push_frame(Frame::try_new(bindings)?))
    }

    /// Resolve `name`, searching innermost-first so inner bindings shadow outer ones.
    pub fn resolve(&self, name: &Variable) -> Option<(&DType, VariableRef)> {
        self.frames
            .iter()
            .enumerate()
            .rev()
            .find_map(|(frame, bindings)| {
                bindings
                    .get(name)
                    .map(|(dtype, slot)| (dtype, VariableRef::new(frame, slot)))
            })
    }
}

impl From<DType> for Scope {
    fn from(root: DType) -> Self {
        Self::new(root)
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
    fn resolve_tracks_frames_slots_and_shadowing() -> VortexResult<()> {
        let scope = Scope::new(i32_())
            .push_frame(Frame::try_new([(Variable::new("x"), i32_())])?)
            .push_frame(Frame::try_new([
                (Variable::new("x"), utf8()),
                (Variable::new("y"), i32_()),
            ])?);

        assert_eq!(scope.depth(), 2);
        assert_eq!(
            scope.resolve(&Variable::new("x")),
            Some((&utf8(), VariableRef::new(1, 0)))
        );
        assert_eq!(
            scope.resolve(&Variable::new("y")),
            Some((&i32_(), VariableRef::new(1, 1)))
        );
        assert!(scope.resolve(&Variable::new("missing")).is_none());
        Ok(())
    }

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

    #[test]
    fn pushing_an_inner_frame_preserves_outer_variable_refs() -> VortexResult<()> {
        let captured = Variable::new("captured");
        let outer = Scope::new(i32_()).push_frame(Frame::try_new([(captured.clone(), utf8())])?);
        let outer_ref = outer
            .resolve(&captured)
            .map(|(_, variable_ref)| variable_ref);

        let inner = outer.push_frame(Frame::try_new([(Variable::new("parameter"), i32_())])?);

        assert_eq!(outer_ref, Some(VariableRef::new(0, 0)));
        assert_eq!(
            inner
                .resolve(&captured)
                .map(|(_, variable_ref)| variable_ref),
            outer_ref
        );
        Ok(())
    }
}
