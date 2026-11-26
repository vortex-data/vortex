// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! One plan opens a standalone IVF index, searches it, and scans the hit rows of the data files
//! it covers, with every open and read going through one sources service.

#![expect(clippy::tests_outside_test_module)]
#![allow(
    clippy::cast_possible_truncation,
    reason = "test data sizes are small constants"
)]

use std::sync::Arc;

use futures::TryStreamExt;
use rstest::rstest;
use vortex::VortexSessionDefault;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::VortexSessionExecute;
use vortex_array::arrays::FixedSizeListArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::StructArray;
use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::dtype::PType;
use vortex_array::validity::Validity;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_expr::col;
use vortex_file::OpenOptionsSessionExt;
use vortex_file::WriteOptionsSessionExt;
use vortex_index::IndexFile;
use vortex_index::IndexFileBuilder;
use vortex_index::plan::IndexScan;
use vortex_index_ivf::IvfIndex;
use vortex_index_ivf::IvfOptions;
use vortex_index_ivf::VectorSearch;
use vortex_io::request::IoService;
use vortex_io::request::OpenTarget;
use vortex_io::request::ReadAtOpenable;
use vortex_io::request::SourcesIoService;
use vortex_io::runtime::tokio::TokioRuntime;
use vortex_scan::planning::drive::ParallelDriver;
use vortex_scan::planning::drive::PlanDriver;
use vortex_scan::planning::drive::PlanRoot;
use vortex_scan::planning::drive::SequentialDriver;
use vortex_scan::planning::planner::WorkScope;
use vortex_session::VortexSession;

const DIM: usize = 8;
const FILES: usize = 3;
const ROWS: usize = 1000;
const K: usize = 10;

fn session() -> VortexResult<VortexSession> {
    let session = VortexSession::default();
    vortex_index::initialize(&session)?;
    vortex_index_ivf::register(&session);
    Ok(session)
}

/// Vectors clustered around 4 directions, so IVF lists are meaningful.
fn vectors(file: usize) -> Vec<[f32; DIM]> {
    let mut state = 17 + file as u64;
    let mut unit = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) % 10_000) as f32 / 10_000.0
    };
    (0..ROWS)
        .map(|row| {
            let mut v = [0f32; DIM];
            v[row % 4] = 4.0;
            v.iter_mut().for_each(|x| *x += unit() - 0.5);
            v
        })
        .collect()
}

/// A file of `vectors` whose ids are unique across files.
fn array(file: usize, vectors: &[[f32; DIM]]) -> VortexResult<ArrayRef> {
    let ids: PrimitiveArray = (0..vectors.len() as i32)
        .map(|row| (file * ROWS) as i32 + row)
        .collect();
    let elements: PrimitiveArray = vectors.iter().flatten().copied().collect();
    let vectors = FixedSizeListArray::new(
        elements.into_array(),
        DIM as u32,
        Validity::NonNullable,
        vectors.len(),
    );
    Ok(
        StructArray::from_fields(&[("id", ids.into_array()), ("vec", vectors.into_array())])?
            .into_array(),
    )
}

/// Data files in memory and a standalone IVF index over their `vec` column.
struct Fixture {
    session: VortexSession,
    data: Vec<ByteBuffer>,
    index: ByteBuffer,
}

impl Fixture {
    async fn new() -> VortexResult<Self> {
        let session = session()?;
        let mut builder = IndexFileBuilder::new(
            IvfIndex,
            IvfOptions { lists: 16, dim: 0 },
            "vec",
            DType::FixedSizeList(
                Arc::new(DType::Primitive(PType::F32, Nullability::NonNullable)),
                DIM as u32,
                Nullability::NonNullable,
            ),
            &session,
        )?;
        let mut data = Vec::new();
        for file in 0..FILES {
            let mut bytes = ByteBufferMut::empty();
            session
                .write_options()
                .write(&mut bytes, array(file, &vectors(file))?.to_array_stream())
                .await?;
            let bytes = bytes.freeze();
            let opened = session.open_options().open_buffer(bytes.clone())?;
            builder
                .add_file(format!("mem://{file}.vortex"), &opened)
                .await?;
            data.push(bytes);
        }
        let mut index = ByteBufferMut::empty();
        builder.write(&mut index).await?;
        Ok(Self {
            session,
            data,
            index: index.freeze(),
        })
    }

    /// The ids of the nearest rows, found by searching the index and reading each file outside
    /// any plan.
    async fn expected(&self, query: &[f32], nprobe: usize) -> VortexResult<Vec<i32>> {
        let index = IndexFile::open(
            IvfIndex,
            &self
                .session
                .open_options()
                .include_metadata()
                .open_buffer(self.index.clone())?,
        )?;
        let hits = index.rows().search_vectors(query, K, nprobe).await?;
        let mut ids: Vec<i32> = index
            .hits_by_file(&hits)?
            .into_iter()
            .map(|hit| (hit.file * ROWS) as i32 + hit.row as i32)
            .collect();
        ids.sort_unstable();
        Ok(ids)
    }

    /// The ids one plan returns: open the index, search it, open and scan the hit files.
    async fn planned(
        &self,
        parallel: bool,
        query: Vec<f32>,
        k: usize,
        nprobe: usize,
    ) -> VortexResult<(Vec<i32>, usize)> {
        let sources = SourcesIoService::new();
        let io: Arc<dyn IoService> = Arc::new(sources.clone());
        let driver: Box<dyn PlanDriver> = if parallel {
            Box::new(ParallelDriver::new(TokioRuntime::current(), io).with_workers(3))
        } else {
            Box::new(SequentialDriver::new(io))
        };
        let scope = WorkScope {
            file_ordinal: 0,
            rows: 0..(FILES * ROWS) as u64,
        };
        let data = self.data.clone();
        let planner = IndexScan {
            sources: sources.id(),
            index: OpenTarget::new(ReadAtOpenable::new(
                "mem://index.vortex",
                Arc::new(self.index.clone()),
            )),
            vtable: IvfIndex,
            session: self.session.clone(),
            scope: scope.clone(),
        }
        .planner(
            Box::new(move |index: &IndexFile<IvfIndex>| {
                let selector = index.clone();
                index
                    .rows()
                    .search_vectors_steps(&query, k, nprobe)?
                    .map(move |hits| selector.hit_selections(&hits))
            }),
            Arc::new(move |covered| {
                let file: usize = covered
                    .uri
                    .trim_start_matches("mem://")
                    .trim_end_matches(".vortex")
                    .parse()
                    .map_err(|err| vortex_err!("Bad test uri {}: {err}", covered.uri))?;
                Ok(OpenTarget::new(ReadAtOpenable::new(
                    covered.uri.as_str(),
                    Arc::new(data[file].clone()),
                )))
            }),
            Arc::new(|builder| {
                let id = col("id").bind(builder.dtype())?;
                Ok(builder.with_projection(id))
            }),
        );
        let batches: Vec<_> = driver
            .execute(PlanRoot::new(scope, Box::new(planner)))
            .try_collect()
            .await?;
        let mut ctx = self.session.create_execution_ctx();
        let mut ids = Vec::new();
        for batch in batches {
            let array = batch.batch.execute::<PrimitiveArray>(&mut ctx)?;
            ids.extend_from_slice(array.as_slice::<i32>());
        }
        ids.sort_unstable();
        Ok((ids, sources.len()))
    }
}

#[rstest]
#[case::sequential(false)]
#[case::parallel(true)]
#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn one_plan_searches_the_index_and_reads_the_hit_rows(
    #[case] parallel: bool,
) -> VortexResult<()> {
    let fixture = Fixture::new().await?;
    let query = vectors(1)[321].to_vec();
    for nprobe in [2, 16] {
        let expected = fixture.expected(&query, nprobe).await?;
        assert_eq!(expected.len(), K);
        assert!(expected.contains(&(ROWS as i32 + 321)));
        let (ids, sources) = fixture.planned(parallel, query.clone(), K, nprobe).await?;
        assert_eq!(ids, expected, "nprobe {nprobe}");
        let files = expected
            .iter()
            .map(|id| *id as usize / ROWS)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            sources,
            1 + files.len(),
            "the index and only the hit files are opened"
        );
    }
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 3)]
async fn a_search_with_no_hits_opens_no_data_file() -> VortexResult<()> {
    let fixture = Fixture::new().await?;
    let (ids, sources) = fixture.planned(false, vec![1.0; DIM], 0, 16).await?;
    assert!(ids.is_empty());
    assert_eq!(sources, 1);
    Ok(())
}
