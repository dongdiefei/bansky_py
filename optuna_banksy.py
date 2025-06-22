import os
import numpy as np
import pandas as pd
import scanpy as sc
from banksy.main import median_dist_to_nearest_neighbour
from banksy.initialize_banksy import initialize_banksy
from banksy.embed_banksy import generate_banksy_matrix
from banksy.main import concatenate_all
from banksy_utils.umap_pca import pca_umap
from banksy.cluster_methods import run_Leiden_partition
from banksy.plot_banksy import plot_results
import matplotlib.pyplot as plt
import warnings
import optuna
from libpysal.weights import WSP
from esda.moran import Moran

warnings.filterwarnings("ignore")
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

# === 读取数据 ===
adata = sc.read_h5ad('visium_hd_data_002.h5ad')
adata.var_names_make_unique()
adata.obs['sample'] = 'P1_CRC'
adata.obs['xcoord'] = adata.obsm['spatial'][:, 0]
adata.obs['ycoord'] = adata.obsm['spatial'][:, 1]

# === 质控 ===
adata.var['mt'] = adata.var_names.str.upper().str.startswith('MT-')
adata.var['ribo'] = adata.var_names.str.upper().str.startswith(("RPS", "RPL"))
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata.obs['log10GenesPerUMI'] = np.log10(adata.obs['n_genes_by_counts']) / np.log10(adata.obs['total_counts'])

# === 过滤细胞 ===
sc.pp.filter_cells(adata, min_counts=30)
sc.pp.filter_genes(adata, min_cells=10)
adata = adata[
    (adata.obs['log10GenesPerUMI'] > 0.8) &
    (adata.obs['log10GenesPerUMI'] < 0.99) &
    (adata.obs['pct_counts_mt'] < 20)
]

# === Normalize ===
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata)

# === HVG + 降采样优化 ===
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000)
adata = adata[:, adata.var['highly_variable']].copy()
sc.pp.subsample(adata, n_obs=10000)
adata.X = adata.X.astype(np.float32)

# === 定义空间连续性计算函数 ===
def calculate_spatial_continuity(labels, weights):
    """计算聚类结果的空间连续性（Moran's I均值）"""
    w = WSP(weights)
    moran_scores = []
    for label in np.unique(labels):
        y = (labels == label).astype(int)
        try:
            mi = Moran(y, w)
            moran_scores.append(mi.I)
        except:
            continue
    return np.mean(moran_scores) if moran_scores else -1

# === Optuna优化目标函数 ===
def objective(trial, adata, coord_keys):
    """定义优化目标函数"""
    try:
        # 参数搜索空间
        k_geom = trial.suggest_int('k_geom', 10, 50)  # 邻域大小
        lambda_val = trial.suggest_float('lambda', 0.1, 0.9)  # 空间权重系数
        resolution = trial.suggest_float('resolution', 0.1, 2.0)  # Leiden分辨率

        # 初始化BANKSY
        banksy_dict = initialize_banksy(
            adata,
            coord_keys,
            k_geom,
            nbr_weight_decay="scaled_gaussian",
            max_m=1,
            plt_edge_hist=False,
            plt_nbr_weights=False,
            plt_agf_angles=False,
            plt_theta=False
        )

        # 生成BANKSY矩阵
        banksy_dict, _ = generate_banksy_matrix(
            adata,
            banksy_dict,
            [lambda_val],
            max_m=1
        )

        # PCA降维
        pca_umap(
            banksy_dict,
            pca_dims=[20],
            add_umap=False,
            plt_remaining_var=False
        )

        # 执行聚类
        results_df, _ = run_Leiden_partition(
            banksy_dict,
            [resolution],
            num_nn=50,
            num_iterations=-1,
            partition_seed=12345,
            match_labels=True
        )

        # 获取聚类结果
        col_name = f'leiden_{str(resolution).replace(".", "p")}'
        if col_name not in results_df.columns:
            return -1.0

        labels = results_df[col_name].values
        weights = banksy_dict['scaled_gaussian']['weights'][0]

        # 计算空间连续性
        return calculate_spatial_continuity(labels, weights)

    except Exception as e:
        print(f"Trial failed: {str(e)}")
        return -1.0

# === 执行自动调参 ===
coord_keys = ('xcoord', 'ycoord', 'spatial')
study = optuna.create_study(direction='maximize')
study.optimize(lambda trial: objective(trial, adata, coord_keys), n_trials=50)

# === 输出最佳参数 ===
best_params = study.best_params
print(f"\nBest parameters found:")
print(f"k_geom: {best_params['k_geom']}")
print(f"lambda: {best_params['lambda']:.3f}")
print(f"resolution: {best_params['resolution']:.3f}")
with open("best_params_002.txt", "w") as f:
    f.write(f"k_geom: {best_params['k_geom']}\n")
    f.write(f"lambda: {best_params['lambda']:.3f}\n")
    f.write(f"resolution: {best_params['resolution']:.3f}\n")

# === 使用最佳参数重新运行完整流程 ===
# 重新初始化BANKSY
best_banksy_dict = initialize_banksy(
    adata,
    coord_keys,
    best_params['k_geom'],
    nbr_weight_decay="scaled_gaussian",
    max_m=1,
    plt_edge_hist=True,
    plt_nbr_weights=True,
    plt_agf_angles=False,
    plt_theta=True
)

# 生成BANKSY矩阵
best_banksy_dict, _ = generate_banksy_matrix(
    adata,
    best_banksy_dict,
    [best_params['lambda']],
    max_m=1
)

# PCA & UMAP可视化
pca_umap(
    best_banksy_dict,
    pca_dims=[20],
    add_umap=True,
    plt_remaining_var=False
)

# 最终聚类
best_results_df, max_num_labels = run_Leiden_partition(
    best_banksy_dict,
    [best_params['resolution']],
    num_nn=50,
    num_iterations=-1,
    partition_seed=12345,
    match_labels=True
)

# === 绘图 ===
c_map = 'tab20'
weights_graph = best_banksy_dict['scaled_gaussian']['weights'][0]
plot_results(
    best_results_df,
    weights_graph,
    c_map,
    match_labels=True,
    coord_keys=coord_keys,
    max_num_labels=max_num_labels,
    save_path='optimized_results_002',
    save_fig=True
)