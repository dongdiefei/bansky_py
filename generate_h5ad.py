import scanpy as sc
import squidpy as sq
import pandas as pd
import json
import os

# 路径配置
data_path = "data/binned_outputs/square_016um"
h5_file = os.path.join(data_path, "filtered_feature_bc_matrix.h5")
position_file = os.path.join(data_path, "tissue_positions.parquet")
scale_file = os.path.join(data_path, "scalefactors_json.json")

# 1. 读表达数据
adata = sc.read_10x_h5(h5_file)
adata.var_names_make_unique()

# 2. 读空间位置信息
positions = pd.read_parquet(position_file)
positions.index = positions['barcode']
adata.obs = adata.obs.join(positions.set_index('barcode'))

# 3. 加入空间信息
with open(scale_file) as f:
    scale_dict = json.load(f)

adata.obsm['spatial'] = adata.obs[['pxl_row_in_fullres', 'pxl_col_in_fullres']].values
adata.uns["spatial"] = {
    "sample": {
        "scalefactors": scale_dict,
        "images": {}  # 可选：也可以加图像
    }
}

# 4. 保存为 .h5ad
adata.write("visium_hd_data_016.h5ad")
print("✅ 成功生成 visium_hd_data_016.h5ad")
