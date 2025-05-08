import ray
import pyarrow as pa
import lance

ray.init()
ds = ray.data.read_json("/test_different_vpc/json_100g")
df = ds.to_pandas()
table = pa.Table.from_pandas(df)
lance.write_dataset(table, "/test_different_vpc/lance_100g")
ds = ray.data.read_lance("/test_different_vpc/lance_100g")
print(ds.count())
#print(ds.take(1))
