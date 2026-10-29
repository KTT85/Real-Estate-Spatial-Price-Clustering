# Phân cụm không gian và giá nhà California

Pipeline Python đáp ứng: làm sạch, tạo đặc trưng, chuẩn hóa, DBSCAN, lọc nhiễu,
K-Means đối chứng, đánh giá, hình dạng không lồi và bản đồ Plotly tương tác.

## 1. Chạy trên Windows PowerShell

Mở terminal ở thư mục này. Python khuyến nghị 3.11–3.13.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe main.py --download
```

Nếu dùng môi trường ảo có sẵn ở thư mục cha của dự án:

```powershell
& '../.venv/Scripts/python.exe' main.py --download
```

`--download` tải bản sao CSV công khai của cùng dữ liệu từ kho handson-ml2,
không truy cập tài khoản Kaggle và không ghi đè CSV sẵn có.
Bạn cũng có thể tải `housing.csv` từ
[Kaggle California Housing](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
và chỉ định đường dẫn:

```powershell
python main.py --data 'D:/duong-dan/housing.csv'
```

Không dùng trực tiếp dữ liệu `fetch_california_housing` thay thế CSV này:
tên cột và ý nghĩa các biến đầu vào khác nhau.

## 2. Hai thí nghiệm

### A. Không gian (`spatial`)

- DBSCAN: Latitude/Longitude theo thứ tự đó, đổi sang radian, khoảng cách Haversine.
- Bán kính khảo sát mặc định: 1, 2, 5, 10, 20 km; min_samples: 5, 10, 20.
- eps theo radian = bán kính km / 6371.0088.
- K-Means: chiếu WGS84 sang California Albers EPSG:3310, đổi mét sang km.
- So sánh hai phương pháp trên cùng không gian tọa độ phẳng; chỉ số Haversine
  trong bảng tìm tham số DBSCAN có mục đích lựa chọn nội bộ.

### B. Không gian + giá + thuộc tính (`combined`)

- Vị trí phẳng được StandardScaler.
- Thu nhập, phòng/hộ, phòng ngủ/hộ, người/hộ, tuổi nhà và giá được log1p rồi StandardScaler.
- Mỗi nhóm được chia căn bậc hai số chiều và nhân căn bậc hai trọng số.
- Trọng số vị trí mặc định 0.5, nhóm thuộc tính 0.5.
- Hai thuật toán dùng cùng ma trận và khoảng cách Euclidean.
- eps được đề xuất từ các phân vị k-distance 50%, 65%, 80%, 90%, 95%.

Giá trị lớn hợp lệ được giữ. StandardScaler cân bằng phương sai sau log1p.
Có thể dùng `--scaler robust` để kiểm tra độ nhạy: RobustScaler dựa trên IQR,
nhưng biến có IQR rất hẹp (như phòng ngủ/hộ) có thể khuếch đại phần đuôi.
Không có lựa chọn scaler nào tự bảo đảm tạo ra cụm có ý nghĩa.
Các hệ số biến đổi được lưu trong [preprocessing.json](./outputs/preprocessing.json).

## 3. Chọn tham số và đánh giá

- DBSCAN: chọn ứng viên có >=2 cụm và silhouette hợp lệ, tối đa hóa
  `((silhouette + 1) / 2) * coverage`.
- Đây là heuristic minh bạch để tránh chọn rất ít điểm đẹp rồi loại phần còn lại;
  không phải tiêu chí tối ưu phổ quát. Luôn đọc toàn bộ bảng tìm kiếm và bản đồ.
- Nếu không ứng viên nào có silhouette hợp lệ, vẫn xuất kết quả dự phòng và đánh dấu
  `dbscan_valid_multicluster_candidate=false`; không coi là tìm được cấu trúc cụm tốt.
- K-Means: K từ 2 đến 10, chọn silhouette mẫu cao nhất; xuất thêm elbow.
- Silhouette tối đa 1.200 điểm lấy mẫu với seed cố định; số điểm thực tính được lưu.
  Cụm hiếm có thể vắng trong mẫu. Có thể tăng `--metric-sample` nếu RAM cho phép.
- DBI, Calinski-Harabasz tính trên toàn bộ điểm được gán cụm, trong không gian Euclidean.
- `assigned_points`: mỗi thuật toán trên các điểm nó gán cụm, vì vậy DBSCAN và K-Means
  có thể khác tập đánh giá. Đọc cùng coverage, không xếp hạng chỉ bằng silhouette.
- `same_DBSCAN_inliers`: đánh giá cả hai trên đúng cùng tập điểm DBSCAN giữ lại.
- Điểm không xác định được lưu NaN/ô trống, không gán 0.
- Kiểm tra độ nhạy DBSCAN khi eps thay đổi ±10%, báo cáo ARI và độ bao phủ.
- Ví dụ `make_moons` có nhãn thật mô phỏng, dùng ARI để minh họa hình dạng không lồi.
  Tham số demo đã chọn riêng cho dữ liệu demo; không suy rộng điểm ARI sang housing.

## 4. Tệp kết quả

Sau khi chạy, mở [báo cáo tiếng Việt](./outputs/REPORT_VI.md).

| Tệp/thư mục | Nội dung |
|---|---|
| `outputs/cleaning_audit.json` | Số dòng trước/sau, thiếu, trùng, trung vị điền |
| `outputs/rejected_rows.csv` | Dòng sai miền hợp lệ, giữ để kiểm tra |
| `outputs/cleaned_housing.csv` | Dữ liệu sạch và đặc trưng mới |
| `outputs/distributions.png`, `correlations.png` | Khám phá phân phối và tương quan |
| `outputs/price_map.html` | Bản đồ giá |
| `outputs/spatial/`, `outputs/combined/` | Hai thí nghiệm |
| `*/dbscan_search.csv`, `kmeans_search.csv` | Toàn bộ ứng viên và điểm đánh giá |
| `*/k_distance.png`, `kmeans_selection.png` | Đồ thị lựa chọn tham số |
| `*/comparison.csv` | Silhouette, DBI, CH, coverage và tỷ lệ nhiễu |
| `*/clustered_housing.csv` | Nhãn cả hai phương pháp cho từng dòng |
| `*/housing_without_noise.csv`, `dbscan_noise.csv` | Dữ liệu sau lọc nhiễu và nhiễu riêng |
| `*/*_profiles.csv` | Số điểm, giá Q25/Q50/Q75, thu nhập, phòng/hộ |
| `*/*_profiles_ocean.csv` | Tỷ lệ nhóm vị trí so với biển trong cụm |
| `*/*_map.html` | Bản đồ tương tác DBSCAN/K-Means |
| `*/clusters.png` | So sánh bản đồ tĩnh |
| `*/stability.json` | Độ nhạy khi thay đổi eps |
| `outputs/complex_shapes.png`, `complex_shapes_metrics.csv` | Demo hình dạng phức tạp |
| `outputs/run_config.json` | Tham số, phiên bản thư viện, checksum dữ liệu |

Mở HTML bằng trình duyệt; Plotly được nhúng trong HTML nhưng nền bản đồ vẫn cần Internet.
Nếu có hơn 24 cụm, màu sẽ lặp: đọc nhãn trong chú giải/tooltip.
Các bản đồ PNG và CSV vẫn dùng ngoại tuyến được.

## 5. Thử nghiệm bổ sung

Giữ giá để phân tích độc lập sau phân cụm:

```powershell
python main.py --exclude-price --output outputs_no_price
```

Khảo sát trọng số vị trí và bán kính khác, lưu ở thư mục riêng để đối chiếu:

```powershell
python main.py --spatial-weight 0.7 --eps-km 1 3 5 8 12 --max-k 12 --output outputs_weight07
```

Đọc `preprocessing.json` để kiểm tra trọng số và các biến thật sự được dùng.
Mỗi lần chạy ghi đè tệp cùng tên trong thư mục đầu ra đã chỉ định.

## 6. Giới hạn cần viết trong báo cáo

- Mỗi dòng là census block group năm 1990, không phải tin rao bán/căn nhà.
- Không có diện tích: không thay số phòng thành diện tích giả.
- Giá trung vị có mức trần; mã đánh dấu `price_at_cap` khi giá >= 500001.
- Nhiễu mật độ khác dữ liệu sai; không tự động xóa trước khi phân tích.
- DBSCAN có thể gặp khó khăn khi mật độ đô thị/nông thôn khác nhau.
- Cụm không gian không tự động là hot-spot giá có ý nghĩa thống kê.
- `high_price_descriptive` chỉ là trung vị giá cụm >= Q75 giá toàn tập, không là kiểm định.
- Giá nằm trong đầu vào combined: không dùng chênh lệch giá làm bằng chứng độc lập.
- Không so trực tiếp điểm số giữa spatial và combined vì không gian đặc trưng khác nhau.
- Bài toán không dự đoán giá: không dùng accuracy, RMSE hoặc R² để chấm cụm.

## 7. Kiểm thử

```powershell
python -m unittest -v test_pipeline.py
```

Kiểm tra xử lý dữ liệu sai/thiếu, mẫu số 0, đặc trưng bật/tắt giá,
đơn vị Haversine và chỉ số khi không có đủ cụm.

## 8. Kết quả đã chạy kiểm chứng

Đã chạy đủ 20.640 dòng, điền 207 giá trị phòng ngủ thiếu và qua 5 kiểm thử.

| Thí nghiệm | DBSCAN | K-Means |
|---|---|---|
| Không gian | 34 cụm, nhiễu 6,44%, eps 10 km, min_samples 20 | K=2 theo silhouette |
| Kết hợp, StandardScaler | 4 cụm, nhiễu 3,15%; cụm lớn nhất chiếm 99,80% inliers | K=2, cụm lớn nhất 57,58% |
| Hai vành trăng mô phỏng | ARI = 1,000 | ARI ≈ 0,246 |

Kết quả combined của DBSCAN chưa tạo được nhiều phân khúc hữu ích mặc dù silhouette
khá cao. Báo cáo tự đánh dấu trường hợp cụm lớn nhất vượt 95% để tránh kết luận chỉ từ điểm số.
Thí nghiệm minh họa cũng cho thấy silhouette của K-Means có thể cao hơn dù ARI thấp hơn:
chỉ số nội tại không thay thế hoàn toàn đánh giá hình dạng và mục tiêu nghiên cứu.

Các phiên bản đã dùng được lưu tại [requirements-lock.txt](./requirements-lock.txt).
Để tái lập trong môi trường sạch, có thể cài `pip install -r requirements-lock.txt`.
Kết quả có thể thay đổi nếu thay dữ liệu, tham số hoặc phiên bản thư viện.
