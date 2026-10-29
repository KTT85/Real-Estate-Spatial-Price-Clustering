"""California housing clustering. Run: python main.py --help."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
from pyproj import Transformer
from sklearn import __version__ as sklearn_version
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons
from sklearn.metrics import (
    adjusted_rand_score, calinski_harabasz_score,
    davies_bouldin_score, silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

SEED = 42
EARTH_KM = 6371.0088
DATA_URL = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
NUMERIC = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income", "median_house_value",
]
ATTRIBUTES = [
    "median_income", "rooms_per_household", "bedrooms_per_household",
    "population_per_household", "housing_median_age",
]


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_data(path: Path, download: bool) -> pd.DataFrame:
    if not path.exists():
        if not download:
            raise FileNotFoundError(
                f"Không tìm thấy {path}. Đặt housing.csv tại đây hoặc dùng --download."
            )
        print("Đang tải bản sao công khai của California Housing...", flush=True)
        with urlopen(DATA_URL, timeout=60) as response:
            payload = response.read()
        # Validate CSV before persisting a downloaded response.
        from io import BytesIO
        candidate = pd.read_csv(BytesIO(payload))
        if not set(NUMERIC).issubset(candidate.columns):
            raise ValueError("Nguồn tải không có cấu trúc California Housing mong đợi.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return pd.read_csv(path)


def clean_data(raw: pd.DataFrame):
    missing = sorted(set(NUMERIC) - set(raw.columns))
    if missing:
        raise ValueError(f"Thiếu các cột: {missing}")
    data = raw.copy()
    data.insert(0, "source_row", np.arange(len(data)))
    audit = {"input_rows": len(data), "missing_before": raw.isna().sum().to_dict()}
    for column in NUMERIC:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data[NUMERIC] = data[NUMERIC].replace([np.inf, -np.inf], np.nan)
    duplicate = data.duplicated(subset=list(raw.columns))
    audit["duplicate_rows_removed"] = int(duplicate.sum())
    data = data.loc[~duplicate].copy()
    # Deliberately broad bounding box: implausible rows are saved for inspection.
    valid = (
        data.latitude.between(32, 42.1) & data.longitude.between(-124.6, -114)
        & data.households.gt(0) & data.median_house_value.gt(0)
        & data.median_income.gt(0) & data.total_rooms.gt(0)
        & data.population.ge(0) & data.housing_median_age.ge(0)
        & (data.total_bedrooms.isna() | data.total_bedrooms.ge(0))
    )
    rejected = data.loc[~valid].copy()
    audit["invalid_rows_removed"] = len(rejected)
    data = data.loc[valid].copy()
    if len(data) < 30:
        raise ValueError("Cần ít nhất 30 dòng hợp lệ để chạy các thí nghiệm.")
    if data.total_bedrooms.notna().sum() == 0:
        raise ValueError("Toàn bộ total_bedrooms bị thiếu; không thể điền trung vị.")
    bedrooms_median = data.total_bedrooms.median()
    audit["bedrooms_imputed"] = int(data.total_bedrooms.isna().sum())
    audit["bedrooms_imputation_median"] = float(bedrooms_median)
    data["total_bedrooms"] = data.total_bedrooms.fillna(bedrooms_median)
    for source, target in [
        ("total_rooms", "rooms_per_household"),
        ("total_bedrooms", "bedrooms_per_household"),
        ("population", "population_per_household"),
    ]:
        data[target] = data[source] / data.households
    data["price_at_cap"] = data.median_house_value.ge(500001)
    audit["price_at_cap_rows"] = int(data.price_at_cap.sum())
    audit["clean_rows"] = len(data)
    if "ocean_proximity" in data:
        data["ocean_proximity"] = data.ocean_proximity.fillna("UNKNOWN").astype(str)
    return data.reset_index(drop=True), rejected, audit


def make_features(data, spatial_weight=0.5, include_price=True, scaler_name="standard"):
    # EPSG:3310 is California Albers, measured in metres.
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3310", always_xy=True)
    x, y = transformer.transform(data.longitude.to_numpy(), data.latitude.to_numpy())
    projected = np.column_stack([x, y]) / 1000.0
    columns = ATTRIBUTES + (["median_house_value"] if include_price else [])
    # After log1p, StandardScaler equalizes variance. RobustScaler is an optional
    # sensitivity experiment: narrow IQRs can strongly amplify legitimate tails.
    scaler_class = StandardScaler if scaler_name == "standard" else RobustScaler
    spatial_scaler = scaler_class()
    attribute_scaler = scaler_class()
    spatial = spatial_scaler.fit_transform(projected)
    attributes = attribute_scaler.fit_transform(np.log1p(data[columns]))
    # Divide by sqrt(dimension) so a larger feature block does not automatically dominate.
    combined = np.column_stack([
        spatial * np.sqrt(spatial_weight / spatial.shape[1]),
        attributes * np.sqrt((1 - spatial_weight) / attributes.shape[1]),
    ])
    metadata = {
        "projection": "EPSG:3310", "spatial_weight": spatial_weight,
        "attributes": columns, "attribute_transform": f"log1p then {scaler_class.__name__}",
        "scaler": scaler_class.__name__,
        "spatial_center": (spatial_scaler.mean_ if scaler_name == "standard" else spatial_scaler.center_).tolist(),
        "spatial_scale": spatial_scaler.scale_.tolist(),
        "attribute_center": (attribute_scaler.mean_ if scaler_name == "standard" else attribute_scaler.center_).tolist(),
        "attribute_scale": attribute_scaler.scale_.tolist(),
    }
    return projected, combined, metadata


def evaluate(X, labels, metric="euclidean", sample_size=1200):
    labels = np.asarray(labels)
    keep = labels != -1
    Y, groups = X[keep], labels[keep]
    sizes = pd.Series(groups).value_counts()
    result = {
        "n_clusters": int(len(sizes)), "noise_ratio": float(1 - keep.mean()),
        "coverage": float(keep.mean()), "clustered_rows": int(keep.sum()),
        "largest_cluster_share": float(sizes.max() / len(Y)) if len(Y) else np.nan,
        "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan,
        "silhouette_sample_rows": 0,
    }
    if not 2 <= len(sizes) < len(Y):
        return result
    # Bounded random sample keeps repeated parameter search tractable.
    rng = np.random.default_rng(SEED)
    indices = rng.choice(len(Y), min(sample_size, len(Y)), replace=False)
    sampled_groups = groups[indices]
    if 2 <= len(np.unique(sampled_groups)) < len(indices):
        result["silhouette"] = float(silhouette_score(Y[indices], sampled_groups, metric=metric))
        result["silhouette_sample_rows"] = len(indices)
    # Centroid scores are not reported on radians/Haversine geometry.
    if metric == "euclidean":
        result["davies_bouldin"] = float(davies_bouldin_score(Y, groups))
        result["calinski_harabasz"] = float(calinski_harabasz_score(Y, groups))
    return result


def fit_dbscan(X, eps, min_samples, metric):
    return DBSCAN(
        eps=eps, min_samples=min_samples, metric=metric,
        algorithm="ball_tree" if metric == "haversine" else "auto", n_jobs=-1,
    ).fit_predict(X)


def select_dbscan(X, metric, output, min_samples_values, eps_values=None, sample_size=1200):
    rows, best_labels, best_key, best_params = [], None, None, None
    fig, ax = plt.subplots(figsize=(9, 5))
    for minimum in min_samples_values:
        neighbors = NearestNeighbors(
            n_neighbors=minimum, metric=metric,
            algorithm="ball_tree" if metric == "haversine" else "auto", n_jobs=-1,
        ).fit(X)
        # Explicit X includes self; min_samples in DBSCAN also includes the point itself.
        distances = np.sort(neighbors.kneighbors(X)[0][:, -1])
        displayed = distances * EARTH_KM if metric == "haversine" else distances
        ax.plot(displayed, label=f"min_samples={minimum}")
        candidates = eps_values
        if candidates is None:
            candidates = np.unique(np.quantile(distances, [0.50, 0.65, 0.80, 0.90, 0.95]))
        for epsilon in candidates:
            if epsilon <= 0:
                continue
            labels = fit_dbscan(X, float(epsilon), minimum, metric)
            scores = evaluate(X, labels, metric, sample_size)
            row = {"eps": float(epsilon), "min_samples": minimum, **scores}
            if metric == "haversine":
                row["eps_km"] = float(epsilon * EARTH_KM)
            # Transparent heuristic: balance separation and coverage; not a universal optimum.
            score = scores["silhouette"]
            row["selection_score"] = (
                float((score + 1) / 2 * scores["coverage"]) if np.isfinite(score) else np.nan
            )
            rows.append(row)
            key = (np.isfinite(row["selection_score"]),
                   row["selection_score"] if np.isfinite(row["selection_score"]) else -np.inf,
                   scores["coverage"])
            if best_key is None or key > best_key:
                best_key, best_labels = key, labels
                best_params = {"eps": float(epsilon), "min_samples": minimum}
    ax.set(xlabel="Points sorted by neighbor distance", ylabel="km" if metric == "haversine" else "Feature distance",
           title="k-distance curves: inspect alongside parameter search")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "k_distance.png", dpi=160)
    plt.close(fig)
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("Không có eps dương. Kiểm tra dữ liệu trùng tọa độ/đặc trưng.")
    table.to_csv(output / "dbscan_search.csv", index=False)
    return best_labels, best_params, table


def select_kmeans(X, output, max_k, sample_size):
    rows, best_labels, best_score, best_k = [], None, -np.inf, None
    distinct = len(np.unique(X, axis=0))
    for k in range(2, min(max_k, len(X) - 1, distinct) + 1):
        model = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit(X)
        scores = evaluate(X, model.labels_, sample_size=sample_size)
        rows.append({"k": k, "inertia": float(model.inertia_), **scores})
        score = scores["silhouette"]
        if best_labels is None or (np.isfinite(score) and score > best_score):
            best_labels, best_k = model.labels_, k
            best_score = score if np.isfinite(score) else -np.inf
    if best_labels is None:
        raise ValueError("K-Means cần ít nhất hai điểm đặc trưng phân biệt.")
    table = pd.DataFrame(rows)
    table.to_csv(output / "kmeans_search.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(table.k, table.inertia, "o-")
    axes[0].set(title="Elbow", xlabel="K", ylabel="Inertia")
    axes[1].plot(table.k, table.silhouette, "o-")
    axes[1].set(title="Silhouette", xlabel="K", ylabel="Score")
    fig.tight_layout()
    fig.savefig(output / "kmeans_selection.png", dpi=160)
    plt.close(fig)
    return best_labels, best_k


def write_map(data, label_column, path, title):
    view = data.copy()
    view["cluster_display"] = view[label_column].map(lambda x: "Noise" if x == -1 else f"C{x}")
    categories = sorted(view.cluster_display.unique())
    colors = {name: px.colors.qualitative.Dark24[i % 24] for i, name in enumerate(categories)}
    colors["Noise"] = "#999999"
    fig = px.scatter_map(
        view, lat="latitude", lon="longitude", color="cluster_display",
        color_discrete_map=colors, zoom=5, opacity=0.7,
        hover_data=["source_row", "median_house_value", "median_income", "rooms_per_household"],
        title=title, height=760,
    )
    fig.update_layout(map_style="carto-positron", margin=dict(l=0, r=0, t=55, b=0))
    fig.write_html(str(path), include_plotlyjs=True)


def profile_clusters(data, label_column, path):
    summary = data.groupby(label_column).agg(
        count=("source_row", "size"),
        price_median=("median_house_value", "median"),
        price_q25=("median_house_value", lambda s: s.quantile(.25)),
        price_q75=("median_house_value", lambda s: s.quantile(.75)),
        income_median=("median_income", "median"),
        rooms_per_household_median=("rooms_per_household", "median"),
        age_median=("housing_median_age", "median"),
        price_at_cap_share=("price_at_cap", "mean"),
    )
    # Descriptive high-price flag, not a statistical hot-spot significance test.
    summary["high_price_descriptive"] = summary.price_median.ge(data.median_house_value.quantile(.75))
    summary.loc[summary.index == -1, "high_price_descriptive"] = False
    summary.to_csv(path)
    if "ocean_proximity" in data:
        pd.crosstab(data[label_column], data.ocean_proximity, normalize="index").to_csv(
            path.with_name(path.stem + "_ocean.csv")
        )


def experiment(data, X, name, output, args, geographic=False):
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    minimums = [v for v in [5, 10, 20] if v <= len(data)]
    # Geographic DBSCAN uses great-circle distance. K-Means requires Euclidean geometry.
    db_X = np.radians(data[["latitude", "longitude"]].to_numpy()) if geographic else X
    metric = "haversine" if geographic else "euclidean"
    eps_grid = np.array(args.eps_km) / EARTH_KM if geographic else None
    db_labels, params, search = select_dbscan(
        db_X, metric, folder, minimums, eps_grid, args.metric_sample,
    )
    km_labels, k = select_kmeans(X, folder, args.max_k, args.metric_sample)
    result = data.copy()
    result["dbscan_cluster"] = db_labels
    result["kmeans_cluster"] = km_labels
    result.to_csv(folder / "clustered_housing.csv", index=False)
    result.loc[result.dbscan_cluster != -1].to_csv(folder / "housing_without_noise.csv", index=False)
    result.loc[result.dbscan_cluster == -1].to_csv(folder / "dbscan_noise.csv", index=False)
    comparisons = []
    for label, labels in [("DBSCAN", db_labels), ("KMeans", km_labels)]:
        # A common projected/feature space makes these comparison scores comparable.
        comparisons.append({"algorithm": label, "scope": "assigned_points", "distance": "euclidean",
                            **evaluate(X, labels, sample_size=args.metric_sample)})
    keep = db_labels != -1
    if keep.any():
        for label, labels in [("DBSCAN", db_labels), ("KMeans", km_labels)]:
            comparisons.append({"algorithm": label, "scope": "same_DBSCAN_inliers", "distance": "euclidean",
                                **evaluate(X[keep], labels[keep], sample_size=args.metric_sample)})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(folder / "comparison.csv", index=False)
    # How much does a small epsilon perturbation change assignments?
    stability = []
    for factor in [.9, 1.1]:
        perturbed = fit_dbscan(db_X, params["eps"] * factor, params["min_samples"], metric)
        common = (db_labels != -1) & (perturbed != -1)
        stability.append({
            "eps_factor": factor,
            "ari_all_noise_as_one_label": float(adjusted_rand_score(db_labels, perturbed)),
            "ari_common_inliers": float(adjusted_rand_score(db_labels[common], perturbed[common]))
                if common.sum() >= 2 and len(np.unique(db_labels[common])) >= 2
                and len(np.unique(perturbed[common])) >= 2 else None,
            "common_inlier_fraction": float(common.mean()),
            "noise_ratio": float(np.mean(perturbed == -1)),
            "n_clusters": len(set(perturbed) - {-1}),
        })
    save_json(folder / "stability.json", stability)
    save_json(folder / "selected_parameters.json", {
        "dbscan": {**params, "metric": metric,
                   "eps_km": params["eps"] * EARTH_KM if geographic else None},
        "kmeans": {"k": k, "n_init": 10, "random_state": SEED},
        "dbscan_selection": "maximize ((sampled_silhouette + 1) / 2) * coverage; require >=2 clusters",
        "dbscan_valid_multicluster_candidate": bool(search.selection_score.notna().any()),
        "kmeans_selection": "maximum sampled silhouette across tested K",
    })
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))
    for ax, title, labels in zip(axes, ["DBSCAN", "K-Means"], [db_labels, km_labels]):
        ax.scatter(data.longitude, data.latitude, c=labels, s=3, cmap="tab20", rasterized=True)
        noise = labels == -1
        ax.scatter(data.longitude[noise], data.latitude[noise], c="gray", s=3)
        ax.set(title=title, xlabel="Longitude", ylabel="Latitude")
    fig.tight_layout()
    fig.savefig(folder / "clusters.png", dpi=180)
    plt.close(fig)
    for column in ["dbscan_cluster", "kmeans_cluster"]:
        profile_clusters(result, column, folder / f"{column}_profiles.csv")
        write_map(result, column, folder / f"{column}_map.html", f"{name} — {column}")
    return comparison


def shape_demo(output):
    X, truth = make_moons(n_samples=1200, noise=.06, random_state=SEED)
    labels = {
        "KMeans": KMeans(n_clusters=2, random_state=SEED, n_init=10).fit_predict(X),
        "DBSCAN": DBSCAN(eps=.16, min_samples=10).fit_predict(X),
    }
    rows = []
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, title, values in zip(axes, ["Ground truth", *labels], [truth, *labels.values()]):
        ax.scatter(*X.T, c=values, s=6, cmap="tab10")
        ax.set_title(title)
        ax.set_aspect("equal")
    for name, values in labels.items():
        rows.append({"algorithm": name, "ARI_against_synthetic_truth": adjusted_rand_score(truth, values),
                     **evaluate(X, values)})
    fig.tight_layout()
    fig.savefig(output / "complex_shapes.png", dpi=170)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(output / "complex_shapes_metrics.csv", index=False)


def eda(data, output):
    columns = ["median_house_value", "median_income", "rooms_per_household", "housing_median_age"]
    data[columns].hist(bins=45, figsize=(11, 8))
    plt.tight_layout()
    plt.savefig(output / "distributions.png", dpi=150)
    plt.close("all")
    correlation = data[NUMERIC + ["rooms_per_household", "bedrooms_per_household"]].corr()
    correlation.to_csv(output / "correlations.csv")
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(correlation, vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(correlation)), correlation.columns, rotation=90)
    ax.set_yticks(range(len(correlation)), correlation.columns)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "correlations.png", dpi=160)
    plt.close(fig)
    fig = px.scatter_map(data, lat="latitude", lon="longitude", color="median_house_value",
                            color_continuous_scale="Viridis", zoom=5, opacity=.65,
                            hover_data=["median_income", "rooms_per_household"],
                            title="Median house value (historical census data)", height=760)
    fig.update_layout(map_style="carto-positron")
    fig.write_html(str(output / "price_map.html"), include_plotlyjs=True)


def write_report(output, comparisons, audit, args):
    sections = [
        "# Báo cáo phân cụm California Housing\n",
        "## 1. Phạm vi và dữ liệu\n",
        "Mỗi dòng là một census block group, không phải một căn nhà. Dữ liệu lịch sử năm 1990; "
        "không có diện tích. Không suy luận giá thị trường hiện tại từ kết quả này.\n",
        f"Số dòng đầu vào: {audit['input_rows']}; sau làm sạch: {audit['clean_rows']}; "
        f"dòng sai bị loại: {audit['invalid_rows_removed']}; "
        f"phòng ngủ được điền trung vị: {audit['bedrooms_imputed']}.\n",
        "## 2. Phương pháp và lý do chọn\n",
        "DBSCAN nhận dạng vùng liên thông theo mật độ và đánh dấu nhiễu; K-Means tạo nhóm quanh tâm, "
        "dễ diễn giải nhưng cần K và có xu hướng chia cụm không lồi. "
        "Thí nghiệm spatial dùng Haversine cho DBSCAN, tọa độ California Albers (km) cho K-Means. "
        "Thí nghiệm combined dùng cùng ma trận Euclidean cho hai thuật toán: log1p các thuộc tính, "
        f"scaler={args.scaler} và trọng số theo nhóm đặc trưng. Không giả lập biến diện tích.\n",
        f"Trọng số không gian: {args.spatial_weight}; giá trong đầu vào combined: {not args.exclude_price}. "
        "Khi có giá trong đầu vào, chênh lệch giá giữa các cụm không phải bằng chứng độc lập về chất lượng.\n",
        "## 3. Các bước\n",
        "1. Kiểm tra schema; ép số; xử lý vô cực và giá trị thiếu.\n"
        "2. Loại dòng trùng và dòng vi phạm miền hợp lệ; lưu dòng sai để kiểm tra.\n"
        "3. Tính phòng/hộ, phòng ngủ/hộ, người/hộ; giữ ngoại lệ hợp lệ.\n"
        "4. Chọn khoảng cách, biến đổi và chuẩn hóa riêng cho từng thí nghiệm.\n"
        "5. Khảo sát k-distance và lưới eps/min_samples; chọn theo silhouette kết hợp coverage.\n"
        "6. Thử K, xem elbow và chọn silhouette cao nhất trong dải khảo sát.\n"
        "7. So sánh chỉ số trên không gian chung và cùng tập DBSCAN inliers; kiểm tra độ nhạy eps.\n"
        "8. Đọc bảng đặc trưng cụm và bản đồ tương tác; phân tích riêng nhiễu.\n",
        "## 4. Kết quả thực nghiệm\n",
    ]
    for name, comparison in comparisons.items():
        params = json.loads((output / name / "selected_parameters.json").read_text(encoding="utf-8"))
        sections.extend([f"### {name}\n", "```json\n" + json.dumps(params, ensure_ascii=False, indent=2) + "\n```\n"])
        sections.append("| Thuật toán | Phạm vi | Cụm | Nhiễu | Silhouette | DBI | CH |\n"
                        "|---|---|---:|---:|---:|---:|---:|\n")
        for row in comparison.itertuples():
            sections.append(f"| {row.algorithm} | {row.scope} | {row.n_clusters} | {row.noise_ratio:.1%} | "
                            f"{row.silhouette:.4f} | {row.davies_bouldin:.4f} | {row.calinski_harabasz:.2f} |\n")
        for algorithm in ["DBSCAN", "KMeans"]:
            row = comparison.loc[(comparison.algorithm == algorithm) & (comparison.scope == "assigned_points")].iloc[0]
            sections.append(f"{algorithm}: {int(row.n_clusters)} cụm; bao phủ {row.coverage:.1%} số điểm. "
                            f"Cụm lớn nhất chiếm {row.largest_cluster_share:.1%} số điểm được gán cụm.\n")
            if row.largest_cluster_share > .95:
                sections.append("**Cần thận trọng:** một cụm chiếm hơn 95% số điểm được gán. "
                                "Silhouette cao có thể phản ánh một nhóm nhỏ biệt lập; kết quả này "
                                "chưa tạo được nhiều phân khúc hữu ích. Cần đọc bảng tìm tham số "
                                "và khảo sát scaler/trọng số/bán kính trước khi gọi là các hot-spot.\n")
        sections.append(f"[Bản đồ DBSCAN](./{name}/dbscan_cluster_map.html) · "
                        f"[Bản đồ K-Means](./{name}/kmeans_cluster_map.html) · "
                        f"[Đặc trưng cụm](./{name}/dbscan_cluster_profiles.csv) · "
                        f"[Độ ổn định](./{name}/stability.json)\n")
    sections.extend([
        "## 5. Diễn giải và giới hạn đánh giá\n",
        "Silhouette/CH cao hơn, DBI thấp hơn thường thuận lợi, nhưng các chỉ số quanh tâm có thể ưu ái "
        "cụm gọn. Điểm số chỉ so sánh trong cùng thí nghiệm và cùng phạm vi điểm. nan nghĩa là "
        "không đủ điều kiện tính, không phải điểm 0. Silhouette dùng mẫu ngẫu nhiên cố định để giới hạn RAM.\n",
        "DBSCAN được chọn bằng heuristic cân bằng silhouette và coverage, không phải nghiệm tối ưu "
        "đã được chứng minh. Nếu không có ứng viên >=2 cụm với silhouette hợp lệ, tham số được lưu "
        "chỉ là phương án dự phòng; không tuyên bố phân cụm thành công. "
        "Các điểm nhiễu vẫn được giữ riêng, vì nhiễu mật độ không đồng nghĩa dữ liệu sai.\n",
        "ARI trong stability.json đo sự thay đổi phân hoạch khi eps đổi ±10%; ARI tất cả điểm "
        "coi nhiễu như một nhãn nên cần đọc cùng ARI trên inliers chung và tỷ lệ bao phủ. "
        "ARI của ví dụ hai vành trăng so với nhãn mô phỏng; không có nhãn chuẩn cho dữ liệu housing.\n",
        "Các cờ high_price_descriptive chỉ mô tả cụm có trung vị giá >= phân vị 75% toàn bộ dữ liệu, "
        "không phải kiểm định hot-spot. Mật độ điểm census còn chịu ảnh hưởng cách chia đơn vị điều tra. "
        "Giá ở mức trần làm giảm độ phân giải của nhóm giá cao.\n",
        "## 6. Cách đọc kết quả và kết luận\n",
        "Đọc clustered_housing.csv để tra từng dòng, profiles để so sánh giá/thu nhập/phòng, "
        "bản đồ để xác định vị trí. Nếu DBSCAN tạo một cụm rất lớn hoặc nhiều nhiễu, cần khảo sát "
        "lại bán kính và ảnh hưởng mật độ không đồng nhất. Nếu K-Means chia vùng liên tục thành "
        "các mảnh quanh tâm, đối chiếu với ví dụ complex_shapes.png. "
        "Không chọn thuật toán chỉ từ một chỉ số; kết luận theo hình dạng, coverage, độ ổn định "
        "và câu hỏi nghiên cứu.\n",
    ])
    (output / "REPORT_VI.md").write_text("\n".join(sections), encoding="utf-8")


def parse_args():
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=base / "data" / "housing.csv")
    parser.add_argument("--output", type=Path, default=base / "outputs")
    parser.add_argument("--download", action="store_true", help="Download public mirror only if local CSV is absent")
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--metric-sample", type=int, default=1200)
    parser.add_argument("--eps-km", type=float, nargs="+", default=[1, 2, 5, 10, 20])
    parser.add_argument("--spatial-weight", type=float, default=.5)
    parser.add_argument("--scaler", choices=["standard", "robust"], default="standard")
    parser.add_argument("--exclude-price", action="store_true", help="Keep price only for post-cluster analysis")
    args = parser.parse_args()
    if args.max_k < 2 or args.metric_sample < 30:
        parser.error("max-k >= 2 và metric-sample >= 30")
    if not 0 < args.spatial_weight < 1 or any(e <= 0 for e in args.eps_km):
        parser.error("0 < spatial-weight < 1 và mọi eps-km phải dương")
    return args


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    raw = load_data(args.data, args.download)
    data, rejected, audit = clean_data(raw)
    print(f"Dữ liệu sạch: {len(data):,} dòng", flush=True)
    data.to_csv(output / "cleaned_housing.csv", index=False)
    rejected.to_csv(output / "rejected_rows.csv", index=False)
    save_json(output / "cleaning_audit.json", audit)
    projected, combined, metadata = make_features(data, args.spatial_weight, not args.exclude_price, args.scaler)
    save_json(output / "preprocessing.json", metadata)
    save_json(output / "run_config.json", {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "python": platform.python_version(), "sklearn": sklearn_version,
        "pandas": pd.__version__, "numpy": np.__version__, "seed": SEED,
        "input_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "download_mirror": DATA_URL if args.download else None,
    })
    eda(data, output)
    comparisons = {}
    for name, X, geographic in [("spatial", projected, True), ("combined", combined, False)]:
        print(f"Đang chạy {name}: tìm tham số, đánh giá, tạo bản đồ...", flush=True)
        comparisons[name] = experiment(data, X, name, output, args, geographic)
    shape_demo(output)
    write_report(output, comparisons, audit, args)
    print(f"Hoàn tất. Xem {output / 'REPORT_VI.md'}", flush=True)


if __name__ == "__main__":
    main()
