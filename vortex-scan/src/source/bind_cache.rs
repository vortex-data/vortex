// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bound request expressions, shared by the sources of one session.

use std::any::Any;
use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_session::SessionExt;
use vortex_session::SessionVar;
use vortex_session::VortexSession;

/// How many bound expressions a session keeps.
///
/// An engine scanning a table opens one source per file and scans each with the same request, so
/// the files of a table that share a dtype bind each of its expressions once. A handful of
/// entries covers the projection and filter of the few queries in flight at once.
const ENTRIES: usize = 32;

/// The expressions recently bound against a dtype and optimized, most recent last.
///
/// Requests stay unbound so that every source binds them against its own dtype. Binding and
/// optimizing a wide projection is real CPU, which an engine opening a source per file would
/// otherwise pay once per file rather than once per query.
///
/// Dynamic expressions compare by identity, so an entry is only found again for the same
/// dynamic value, never for an equal-looking one of another query.
#[derive(Clone, Default)]
pub(super) struct BindCache(Arc<Mutex<VecDeque<Entry>>>);

struct Entry {
    dtype: DType,
    expr: Expression,
    bound: BoundExpression,
}

impl BindCache {
    /// `expr` bound against `dtype` and optimized, from an earlier scan of the session where one
    /// bound it already.
    pub(super) fn bind(
        session: &VortexSession,
        expr: &Expression,
        dtype: &DType,
    ) -> VortexResult<BoundExpression> {
        let cache = BindCache::clone(&session.get::<BindCache>());
        if let Some(bound) = cache.find(expr, dtype) {
            return Ok(bound);
        }
        let bound = expr.bind(dtype)?.optimize_recursive()?;
        let mut entries = cache.0.lock();
        if entries.len() == ENTRIES {
            entries.pop_front();
        }
        entries.push_back(Entry {
            dtype: dtype.clone(),
            expr: expr.clone(),
            bound: bound.clone(),
        });
        Ok(bound)
    }

    fn find(&self, expr: &Expression, dtype: &DType) -> Option<BoundExpression> {
        let mut entries = self.0.lock();
        let index = entries
            .iter()
            .rposition(|entry| entry.expr == *expr && entry.dtype == *dtype)?;
        // Keep entries in use at the back, so the least recently used is evicted first.
        let entry = entries.remove(index)?;
        let bound = entry.bound.clone();
        entries.push_back(entry);
        Some(bound)
    }
}

impl fmt::Debug for BindCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BindCache")
            .field("entries", &self.0.lock().len())
            .finish()
    }
}

impl SessionVar for BindCache {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::Expression;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::lit;
    use vortex_array::expr::root;
    use vortex_error::VortexResult;
    use vortex_session::SessionExt;
    use vortex_session::VortexSession;

    use super::BindCache;

    fn ints() -> Expression {
        get_item("ints", root())
    }

    fn table(ints: PType) -> DType {
        DType::struct_(
            [("ints", DType::Primitive(ints, Nullability::NonNullable))],
            Nullability::NonNullable,
        )
    }

    fn entries(session: &VortexSession) -> usize {
        session.get::<BindCache>().0.lock().len()
    }

    #[test]
    fn files_sharing_a_dtype_bind_once() -> VortexResult<()> {
        let session = VortexSession::empty();
        let filter = gt(ints(), lit(5i32));
        let first = BindCache::bind(&session, &filter, &table(PType::I32))?;
        let second = BindCache::bind(&session, &filter, &table(PType::I32))?;
        assert_eq!(first, second);
        assert_eq!(
            first,
            filter.bind(&table(PType::I32))?.optimize_recursive()?
        );
        assert_eq!(entries(&session), 1);
        Ok(())
    }

    #[test]
    fn another_dtype_binds_again() -> VortexResult<()> {
        let session = VortexSession::empty();
        let narrow = BindCache::bind(&session, &ints(), &table(PType::I32))?;
        let wide = BindCache::bind(&session, &ints(), &table(PType::I64))?;
        assert_eq!(
            narrow.dtype(),
            &DType::Primitive(PType::I32, Nullability::NonNullable)
        );
        assert_eq!(
            wide.dtype(),
            &DType::Primitive(PType::I64, Nullability::NonNullable)
        );
        assert_eq!(entries(&session), 2);
        Ok(())
    }
}
