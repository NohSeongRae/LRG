#please implement data generate code

from osmnx import get_original_csv
from data_preprocessing import preprocess_node
from data_preprocessing import preprocess_edge
from dgl_graph import make_dgl

def generate_dataset(city_name, make_original, pre_process, dgl):
    if make_original == True:
        # 1 osmnx
        get_original_csv(city_name)
    if pre_process == True:
        # 2 data_preprocessing
        preprocess_edge(city_name)
        preprocess_node(city_name)
    result = make_dgl(city_name)
    return result

city_name = 'atlanta'
result = generate_dataset(city_name, False, True, True)
print(result)