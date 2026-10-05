# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import asyncio
import importlib.util
import shutil
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def minimizer() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "minimize_fuzz_corpus.py"
    spec = importlib.util.spec_from_file_location("minimize_fuzz_corpus", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shards_partition_every_input_without_copying(minimizer, tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for index in range(19):
        (corpus / str(index)).write_bytes(bytes([index]) * (index + 1))
    files = minimizer.corpus_files(corpus)
    merges = minimizer.shard_corpus(files, tmp_path / "work", 4)
    sharded = [path for merge in merges for path in minimizer.corpus_files(merge.inputs[0])]
    assert sorted(path.read_bytes() for path in sharded) == sorted(path.read_bytes() for path in files)
    assert {path.stat().st_ino for path in sharded} == {path.stat().st_ino for path in files}
    assert max(merge.total for merge in merges) - min(merge.total for merge in merges) <= 1


def test_progress_waits_for_complete_records(minimizer, tmp_path):
    merge = minimizer.Merge("test", [], tmp_path, 2)
    merge.control.write_bytes(b"2\n0\ninput-0\ninput-1\nSTARTED 0 1\nFT 0 3\nCOV 0 3\nSTARTED 1 1\nCOV 1")
    merge.read_progress()
    assert merge.processed == 1
    with merge.control.open("ab") as output:
        output.write(b" 4\n")
    merge.read_progress()
    merge.read_progress()
    assert merge.processed == 2


@pytest.mark.parametrize("failure", ["shard", "final", "empty", None])
def test_publish_only_complete_coverage_union(minimizer, tmp_path, monkeypatch, failure):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    original = {str(index): data for index, data in enumerate([b"ab", b"a", b"bc", b"c", b"ab", b"d"])}
    for name, data in original.items():
        (corpus / name).write_bytes(data)
    work = tmp_path / "work"
    work.mkdir()
    (work / "stale-output").write_bytes(b"must not survive")
    phases = []

    async def fake_phase(merges, *_args: object) -> bool:
        phases.append([merge.name for merge in merges])
        assert not (work / "stale-output").exists()
        for merge in merges:
            merge.output.mkdir(parents=True)
            covered = set()
            for inputs in merge.inputs:
                for path in minimizer.corpus_files(inputs):
                    features = set(path.read_bytes())
                    if features - covered and failure != "empty":
                        shutil.copyfile(path, merge.output / path.name)
                        covered.update(features)
        return not (
            failure == "shard" and merges[0].name != "final" or failure == "final" and merges[0].name == "final"
        )

    monkeypatch.setattr(minimizer, "run_phase", fake_phase)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    success = asyncio.run(minimizer.minimize(Path("binary"), corpus, work, tmp_path / "artifacts", 2))
    if failure:
        assert not success
        assert {path.name: path.read_bytes() for path in corpus.iterdir()} == original
    else:
        assert success
        assert phases == [["shard-0", "shard-1"], ["final"]]
        assert set(b"".join(path.read_bytes() for path in corpus.iterdir())) == set(b"abcd")


def test_merge_rejects_skipped_inputs_even_with_success_exit(minimizer, tmp_path):
    # libFuzzer's crash-resistant merge may exit zero after skipping a crashing seed.
    binary = tmp_path / "fuzzer"
    binary.write_text("#!/bin/sh\nprintf '%s\\n' \"MERGE-INNER: input caused a failure\"\nexit 0\n")
    binary.chmod(0o755)
    merge = minimizer.Merge("test", [], tmp_path / "merge", 1)
    assert not asyncio.run(minimizer.run_phase([merge], binary, tmp_path / "artifacts", {}))
