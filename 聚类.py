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

warnings.filterwarnings("ignore")
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'

adata = sc.read_h5ad('visium_hd_data_008.h5ad')
adata.var_names_make_unique()
adata.obs['sample'] = 'P1_CRC'

# 质控
adata.var['mt'] = adata.var_names.str.upper().str.startswith('MT-')
adata.var['ribo'] = adata.var_names.str.upper().str.startswith(("RPS", "RPL"))
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
adata.obs['log10GenesPerUMI'] = np.log10(adata.obs['n_genes_by_counts']) / np.log10(adata.obs['total_counts'])

# 细胞过滤
sc.pp.filter_cells(adata, min_counts=30)
sc.pp.filter_genes(adata, min_cells=10)
adata = adata[
    (adata.obs['log10GenesPerUMI'] > 0.8) & (adata.obs['log10GenesPerUMI'] < 0.99) & (adata.obs['pct_counts_mt'] < 20)]

# Normalize
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata)

# HVG
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(
    adata,
    flavor='seurat',
    n_top_genes=2000
)

## 构建空间最近邻图
# Find median distance to closest neighbours
nbrs = median_dist_to_nearest_neighbour(adata, key="spatial")

## 生成距离空间权重(Generate spatial weights from distance)
k_geom = 15  # number of spatial neighbours
max_m = 1  # use both mean and AFT
nbr_weight_decay = "scaled_gaussian"  # can also choose "reciprocal", "uniform" or "ranked"
coord_keys = ('xcoord', 'ycoord', 'spatial')
banksy_dict = initialize_banksy(
    adata,
    coord_keys,
    k_geom,
    nbr_weight_decay=nbr_weight_decay,
    max_m=max_m,
    plt_edge_hist=True,  # Visualize the edge histogram to show the histogram of distances between cells and the weights
    plt_nbr_weights=True,  # Visualize weights by plotting the connections.
    plt_agf_angles=False,  # Visualize weights with Azimuthal angles
    plt_theta=True,
    # Visualize angles around random cell. Plot points around a random index cell, annotated with angles from the index cell.
)

## 生成BANKSY矩阵
# BANKSY几个主要超参数
resolutions = [0.5]  # clustering resolution for UMAP
pca_dims = [20]  # Dimensionality in which PCA reduces to
lambda_list = [0.2]  # list of lambda parameters
banksy_dict, banksy_matrix = generate_banksy_matrix(
    adata,
    banksy_dict,
    lambda_list,
    max_m
)

## 将非空间结果附加到banksy_dict以进行比较（不必要）
banksy_dict["nonspatial"] = {
    # Here we simply append the nonspatial matrix (adata.X) to obtain the nonspatial clustering results
    0.0: {"adata": concatenate_all([adata.X], 0, adata=adata), }
}

## PCA降维, UMAP可视化
pca_umap(
    banksy_dict,
    pca_dims=pca_dims,  # 这里PCA降到20个维度
    add_umap=True,
    plt_remaining_var=False,
)

## 细胞聚类--默认情况下，建议使用基于分辨率的聚类（即leiden 或 louvain）
results_df, max_num_labels = run_Leiden_partition(
    banksy_dict,
    resolutions,
    num_nn=50,  # k_expr: number of neighbours in expression (BANKSY embedding or non-spatial) space
    num_iterations=-1,  # run to convergenece
    partition_seed=12345,
    match_labels=True,
)
from banksy.plot_banksy import plot_results
c_map =  'tab20' # specify color map
weights_graph =  banksy_dict['scaled_gaussian']['weights'][0]
plot_results(
    results_df,
    weights_graph,
    c_map,
    match_labels = True,
    coord_keys = coord_keys,
    max_num_labels  =  max_num_labels,
    save_path = 'tmp_png',
    save_fig = True
)