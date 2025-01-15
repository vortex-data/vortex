import pyarrow.parquet as pq
import vortex as vx
import numpy as np
from time import time


# taken from OpenAI text-3-small
EMBED_DIM = 1536
N_EMBEDS = 1

def generate_users(row_count, col_count, wrapped=False):
    users = []
    for i in range(row_count):
        #username = f"user_{i}"
        # truncate the precision to 4 decimal places to trigger ALP
        if wrapped:
            user_dict = {f"embed_{i}": {"vals": np.round(np.random.rand(EMBED_DIM), 7) } for i in range(col_count) }
        else:
            user_dict = {f"embed_{i}": np.round(np.random.rand(EMBED_DIM), 7) for i in range(col_count) }
        #embedding = np.round(np.random.rand(EMBED_DIM), 4).tolist()
        #users.append({"username": username, "embedding": embedding})
        users.append(user_dict)
    return users

#
#print("generating test users...")
#start = time()
#test_users = generate_users(10_000)
#print("completed in ", time() - start)
#
## emit the users to Vortex
#print("moving array into vortex...")
#start = time()
#arr = vx.array(test_users)
#print("completed in ", time() - start)
##print(arr.tree_display())
#
#print("compressing vortex array")
#start = time()
#comp = vx.compress(arr)
#print("completed in ", time() - start)
#print(comp.tree_display())

def generate_list_cols(row_count, col_count, wrapped):
    users = generate_users(row_count, col_count, wrapped)
    arr = vx.array(users)
    # compression speed
    start = time()
    comp = vx.compress(arr)
    duration = time() - start
    print(f"wrapped={wrapped},rows={row_count},cols={col_count},time={duration}")
    #print(comp.tree_display())
    start = time()
    comp.to_arrow_table()
    print("   decompress time: ", time() - start)


for row_count in [1_000]:
    for col_count in [14000]:
        for wrapped in [False, True]:
    #for col_count in [1, 100, 200, 300, 400, 500, 700, 800, 900, 1_000]:
            generate_list_cols(row_count, col_count, wrapped)
