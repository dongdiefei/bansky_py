import os
import numpy as np
import pandas as pd
import scanpy as sc
import harmonypy as hm
import matplotlib.pyplot as plt
import warnings

from banksy.main import median_dist_to_nearest_neighbour, concatenate_all
from banksy.initialize_banksy import initialize_banksy
from banksy.embed_banksy import generate_banksy_matrix
from banksy_utils.umap_pca import pca_umap
from banksy.cluster_methods import run_Leiden_partition
from banksy.plot_banksy import plot_results

warnings.filterwarnings("ignore")
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

# === Step 1. 读取多个 Visium HD 样本 ===
file_paths = [
    'visium_hd_data_002.h5ad',
    'visium_hd_data_008.h5ad',
    'visium_hd_data_016.h5ad'
]

sample_names = ['P002', 'P008', 'P016']
adatas = []

for file, sample in zip(file_paths, sample_names):
    ad = sc.read_h5ad(file)
    ad.var_names_make_unique()
    ad.obs['sample'] = sample
    ad.obs['xcoord'] = ad.obsm['spatial'][:, 0]
    ad.obs['ycoord'] = ad.obsm['spatial'][:, 1]

    # 质控
    ad.var['mt'] = ad.var_names.str.upper().str.startswith('MT-')
    sc.pp.calculate_qc_metrics(ad, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    ad.obs['log10GenesPerUMI'] = np.log10(ad.obs['n_genes_by_counts']) / np.log10(ad.obs['total_counts'])

    ad = ad[(ad.obs['log10GenesPerUMI'] > 0.8) &
            (ad.obs['log10GenesPerUMI'] < 0.99) &
            (ad.obs['pct_counts_mt'] < 20)]

    sc.pp.filter_cells(ad, min_counts=30)
    sc.pp.filter_genes(ad, min_cells=10)

    ad.layers["counts"] = ad.X.copy()
    sc.pp.normalize_total(ad)
    sc.pp.log1p(ad)

    adatas.append(ad)

# === Step 2. 合并多个样本 ===
adata = adatas[0].concatenate(
    adatas[1:],
    batch_key="sample",
    batch_categories=sample_names,
    index_unique=None
)

# === Step 3. 高变基因 + 降采样 ===
sc.pp.highly_variable_genes(adata, flavor='seurat', n_top_genes=2000, batch_key='sample')
adata = adata[:, adata.var['highly_variable']].copy()
sc.pp.subsample(adata, n_obs=10000)
adata.X = adata.X.astype(np.float32)

# === Step 4. Harmony 批效应校正 ===
sc.pp.scale(adata, max_value=10)
sc.tl.pca(adata, svd_solver='arpack', n_comps=30)
ho = hm.run_harmony(adata.obsm['X_pca'], adata.obs, 'sample')
adata.obsm['X_pca_harmony'] = ho.Z_corr.T

# === Step 5. BANKSY 初始化（使用 Harmony 校正）===
nbrs = median_dist_to_nearest_neighbour(adata, key="spatial")

banksy_dict = initialize_banksy(
    adata,
    coord_keys=('xcoord', 'ycoord', 'spatial'),
    nbr_weight_decay="scaled_gaussian",
    max_m=1,
    plt_edge_hist=False,
    plt_nbr_weights=False,
    plt_agf_angles=False,
    plt_theta=False
)

# 替换 PCA 为 Harmony 校正后的结果
adata.obsm['X_pca'] = adata.obsm['X_pca_harmony']

# === Step 6. BANKSY 嵌入矩阵生成 ===
resolutions = [0.5]
pca_dims = [20]
lambda_list = [0.4]

banksy_dict, banksy_matrix = generate_banksy_matrix(
    adata,
    banksy_dict,
    lambda_list,
    max_m=1
)

banksy_dict["nonspatial"] = {
    0.0: {"adata": adata.copy()}
}

# === Step 7. PCA + UMAP 可视化 ===
pca_umap(
    banksy_dict,
    pca_dims=pca_dims,
    add_umap=True,
    plt_remaining_var=False
)

# === Step 8. 聚类 ===
results_df, max_num_labels = run_Leiden_partition(
    banksy_dict,
    resolutions,
    num_nn=50,
    num_iterations=-1,
    partition_seed=12345,
    match_labels=True
)

# === Step 9. 可视化聚类结果 ===
plot_results(
    results_df,
    banksy_dict['scaled_gaussian']['weights'][0],
    c_map='tab20',
    match_labels=True,
    coord_keys=('xcoord', 'ycoord', 'spatial'),
    max_num_labels=max_num_labels,
    save_path='banksy_harmony',
    save_fig=True
)
