import vortex as vx
import pyarrow.parquet as pq
from time import time


print("reading parquet file to arrow...", end="", flush=True)
start = time()
table = pq.read_table("/Users/aduffy/Downloads/share_vortex/A0.small.50.parquet")
print("completed in ", time() - start)

print("reading arrow to Vortex...", end="", flush=True)
start = time()
vtable = vx.array(table)
print("completed in ", time() - start)

print("compressing vortex...", end="", flush=True)
start = time()
vtable = vx.compress(vtable)
print("completed in ", time() - start)

