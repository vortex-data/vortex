import pyarrow.parquet as pq
import vortex as vx

#table = pq.read_table("/Users/aduffy/Downloads/share_vortex/A0.small.parquet")
table = pq.read_table("/Users/aduffy/Downloads/share_vortex/A0.small.50.parquet")
vtable = vx.array(table)

vx.io.write_path(vtable, "50_avro_meta.vortex")
