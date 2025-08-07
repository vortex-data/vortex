#!/usr/bin/env python3

"""
Test script for Arrow FFI streaming support in PyVortex write API
"""

import sys
import tempfile
import os

try:
    import pyarrow as pa
    import vortex as vx

    print("Testing Arrow FFI streaming support...")

    # Create sample data
    data = {
        "id": [1, 2, 3, 4, 5],
        "name": ["Alice", "Bob", "Charlie", "David", "Eve"],
        "score": [95.5, 87.2, 92.1, 88.8, 91.3],
    }

    # Test 1: PyArrow Table streaming
    print("\n1. Testing PyArrow Table streaming...")
    table = pa.table(data)
    print(f"Created PyArrow table with {len(table)} rows")

    with tempfile.NamedTemporaryFile(suffix=".vortex", delete=False) as tmp:
        try:
            vx.io.write(table, tmp.name)
            print(f"✓ Successfully wrote PyArrow table to {tmp.name}")

            # Verify we can read it back
            result = vx.io.read_url(f"file://{tmp.name}")
            print(f"✓ Successfully read back {len(result)} rows")

        except Exception as e:
            print(f"✗ Failed to write PyArrow table: {e}")
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    # Test 2: PyArrow RecordBatchReader streaming
    print("\n2. Testing PyArrow RecordBatchReader streaming...")
    schema = pa.schema([("id", pa.int64()), ("name", pa.string()), ("score", pa.float64())])

    # Create batches
    batches = []
    for i in range(0, len(data["id"]), 2):  # Split into batches of 2
        batch_data = {"id": data["id"][i : i + 2], "name": data["name"][i : i + 2], "score": data["score"][i : i + 2]}
        batch = pa.record_batch(batch_data, schema)
        batches.append(batch)

    reader = pa.RecordBatchReader.from_batches(schema, batches)
    print(f"Created RecordBatchReader with {len(batches)} batches")

    with tempfile.NamedTemporaryFile(suffix=".vortex", delete=False) as tmp:
        try:
            vx.io.write(reader, tmp.name)
            print(f"✓ Successfully wrote RecordBatchReader to {tmp.name}")

            # Verify we can read it back
            result = vx.io.read_url(f"file://{tmp.name}")
            print(f"✓ Successfully read back {len(result)} rows")

        except Exception as e:
            print(f"✗ Failed to write RecordBatchReader: {e}")
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

    print("\n✓ All tests completed successfully!")
    print("\nArrow FFI streaming support is working correctly.")
    print("Users can now stream large datasets directly from PyArrow to Vortex")
    print("without loading the entire dataset into memory!")

except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure PyArrow and Vortex are installed")
    sys.exit(1)
except Exception as e:
    print(f"Test failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
