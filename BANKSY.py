import os
import pandas as pd
import scanpy as sc
import numpy as np
import matplotlib.pyplot as plt
from banksy.main import median_dist_to_nearest_neighbour
from banksy.initialize_banksy import initialize_banksy
from banksy.embed_banksy import generate_banksy_matrix
from banksy_utils.umap_pca import pca_umap
from banksy.cluster_methods import run_Leiden_partition


# ========================
# 1. 数据准备与格式转换
# ========================
def prepare_visium_hd_data(data_dir):
    """
    处理Visium HD数据格式：
    1. 将tissue_positions.parquet转换为旧版CSV格式
    2. 构建符合Scanpy要求的目录结构
    """
    # 空间坐标转换
    spatial_dir = os.path.join(data_dir, "spatial")
    tissue_pos = pd.read_parquet(os.path.join(spatial_dir, "tissue_positions.parquet"))
    tissue_pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_col_in_fullres", "pxl_row_in_fullres"]
    tissue_pos[["x_pixel", "y_pixel"]] = tissue_pos[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values
    tissue_pos.to_csv(os.path.join(spatial_dir, "tissue_positions_list.csv"), index=False, header=None)

    # 构建矩阵目录结构
    matrix_dir = os.path.join(data_dir, "filtered_feature_bc_matrix")
    os.makedirs(matrix_dir, exist_ok=True)
    # 此处假设已存在矩阵文件，实际可能需要从原始数据生成

    return data_dir


# 配置数据路径（需根据实际路径修改）
raw_data_path = "C:/Users/董蝶菲/Desktop/Banksy_py-main/data/binned_outputs/square_002um"
processed_data_path = prepare_visium_hd_data(raw_data_path)


# ========================
# 2. 数据读取与预处理
# ========================
def preprocess_data(data_path):
    # 读取数据
    adata = sc.read_visium(data_path, library_id="mouse_brain")

    # 添加空间坐标（转换后的CSV）
    spatial_df = pd.read_csv(
        os.path.join(data_path, "spatial", "tissue_positions_list.csv"),
        header=None,
        names=["barcode", "in_tissue", "array_row", "array_col", "x", "y"]
    )
    adata.obs["x_pixel"] = spatial_df["x"].values
    adata.obs["y_pixel"] = spatial_df["y"].values
    adata.obsm["spatial"] = adata.obs[["x_pixel", "y_pixel"]].values

    # 质控过滤
    sc.pp.filter_cells(adata, min_counts=50)
    adata = adata[adata.obs['pct_counts_mt'] < 30]  # 假设已有线粒体基因比例计算

    # 标准化
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 高变基因选择
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    adata = adata[:, adata.var.highly_variable]

    return adata


adata = preprocess_data(processed_data_path)


# ========================
# 3. 空间图构建
# ========================
def build_spatial_graph(adata, k_geom=15):
    # 计算中位数最近邻距离
    nbrs = median_dist_to_nearest_neighbour(adata, key="spatial")

    # 生成空间权重
    banksy_dict = initialize_banksy(
        adata,
        coord_keys=('x_pixel', 'y_pixel', 'spatial'),
        k_geom=k_geom,
        nbr_weight_decay="scaled_gaussian",
        max_m=1,
        plt_edge_hist=False,
        plt_nbr_weights=False
    )
    return banksy_dict


banksy_dict = build_spatial_graph(adata)

# ========================
# 4. Banksy矩阵生成
# ========================
banksy_matrix = generate_banksy_matrix(adata, banksy_dict)


# ========================
# 5. 降维聚类
# ========================
def run_banksy_clustering(banksy_matrix, lambda_=0.5):
    # PCA降维
    pca_result = pca_umap(banksy_matrix, n_components=50, method="pca")

    # Leiden聚类
    leiden_clusters = run_Leiden_partition(
        pca_result,
        resolution=1.0,
        lambda_=lambda_,
        random_state=42
    )
    return leiden_clusters


adata.obs["banksy_cluster"] = run_banksy_clustering(banksy_matrix)


# ========================
# 6. 可视化
# ========================
def visualize_results(adata, output_prefix="results"):
    # UMAP可视化
    sc.tl.umap(adata)
    sc.pl.umap(adata, color="banksy_cluster", frameon=False, save=f"{output_prefix}_umap.png")

    # 空间分布
    sc.pl.spatial(
        adata,
        color="banksy_cluster",
        library_id="mouse_brain",
        alpha_img=0.6,
        save=f"{output_prefix}_spatial.png"
    )


visualize_results(adata)

print("✅ 全部流程执行完成！结果已保存至当前目录")