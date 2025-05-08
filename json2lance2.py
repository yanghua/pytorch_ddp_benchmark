import ray
import pyarrow as pa
import lance
json_dir = "/test_different_vpc/json_100g"
lance_path = "/test_different_vpc/lance_100g.lance"
ray.init()
ds = ray.data.read_json(json_dir).write_lance(lance_path)
