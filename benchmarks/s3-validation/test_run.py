# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright the Vortex contributors

import copy
import json
import tempfile
import unittest
from pathlib import Path

import run


class ChunkTests(unittest.TestCase):
    def write(self, path: Path, header: dict, query: int):
        path.write_text(json.dumps(header) + "\n" + json.dumps({"kind": "sample", "query": query}) + "\n")

    def test_chunks_preserve_one_environment_and_all_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = {"kind": "environment", "instance_id": "i-1", "objects": [{"e_tag": "abc"}]}
            first, second, merged = root / "first.jsonl", root / "second.jsonl", root / "merged.jsonl"
            self.write(first, header, 1)
            self.write(second, header, 72)
            run.append_chunk(first, merged)
            run.append_chunk(second, merged)
            records = [json.loads(line) for line in merged.read_text().splitlines()]
            self.assertEqual(records, [header, {"kind": "sample", "query": 1}, {"kind": "sample", "query": 72}])
            self.assertFalse(first.exists())
            self.assertFalse(second.exists())

    def test_rejects_changed_host_or_objects_without_altering_previous_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            header = {"kind": "environment", "instance_id": "i-1", "objects": [{"e_tag": "abc"}]}
            chunk, merged = root / "chunk.jsonl", root / "merged.jsonl"
            self.write(chunk, header, 1)
            run.append_chunk(chunk, merged)
            original = merged.read_bytes()
            for field, value in (("instance_id", "i-2"), ("objects", [{"e_tag": "changed"}])):
                changed = copy.deepcopy(header)
                changed[field] = value
                self.write(chunk, changed, 72)
                with self.assertRaisesRegex(ValueError, "Query chunks differ"):
                    run.append_chunk(chunk, merged)
                self.assertEqual(merged.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
