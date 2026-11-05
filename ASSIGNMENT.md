# Bài tập 2 — Từ mạng tích chập đến mô hình chuỗi


## Mục tiêu

Assignment 2 nối hai nhóm kiến thức lớn đã học:

$$
\text{CNN}
\longrightarrow
\text{chuẩn hóa và regularization}
\longrightarrow
\text{huấn luyện bằng PyTorch}
\longrightarrow
\text{RNN}
\longrightarrow
\text{sinh chuỗi có điều kiện}.
$$

Sau khi hoàn thành, bạn cần có khả năng:

- giải thích và cài đặt phép tích chập (convolution) và gộp cực đại (max pooling);
- thực hiện truyền xuôi và truyền ngược qua các lớp CNN cơ bản;
- giải thích vai trò của chuẩn hóa theo lô (Batch Normalization) và Dropout;
- xây dựng, huấn luyện và lựa chọn một CNN bằng PyTorch;
- giải thích trạng thái ẩn (hidden state) và cách RNN xử lý chuỗi;
- cài đặt một bước RNN và truyền xuôi qua cả chuỗi;
- làm việc với biểu diễn từ (word embedding) và hàm mất mát theo thời gian;
- xây dựng một mô hình RNN sinh chuỗi có điều kiện từ đặc trưng ảnh.

## Cấu trúc

Hãy làm theo đúng thứ tự:

| Notebook | Nội dung |
|---|---|
| `Q1_CNN_from_scratch.ipynb` | Convolution, pooling, truyền ngược và kiểm tra đạo hàm |
| `Q2_Normalization_Dropout.ipynb` | Batch Normalization, Layer Normalization và Dropout |
| `Q3_PyTorch_CNN.ipynb` | Huấn luyện CNN trên CIFAR-10 bằng PyTorch |
| `Q4_RNN_fundamentals.ipynb` | Vanilla RNN, embedding và hàm mất mát theo thời gian |
| `Q5_RNN_captioning.ipynb` | Mô hình sinh mô tả ảnh đơn giản bằng CNN + RNN |

## Cách sử dụng notebook

Mỗi notebook được thiết kế theo dạng **vừa học vừa thực hành**:

1. nhắc lại lý thuyết;
2. ví dụ minh họa;
3. phần cài đặt;
4. kiểm tra tính hợp lý hoặc so sánh với PyTorch;
5. câu hỏi ngắn;
6. thí nghiệm và phân tích kết quả.

Không nên chỉ tìm các ô code cần hoàn thiện. Hãy đọc phần giải thích ngay trước mỗi bài thực hành.

## Phần code cần hoàn thiện

Bạn **chỉ chỉnh sửa 5 notebook**.

Các file trong `dl2/` là tiện ích được cung cấp sẵn và **không có phần bài tập cần hoàn thiện**.

### Q1–Q2
Ưu tiên NumPy để hiểu cơ chế bên trong. Không dùng lớp PyTorch tương ứng để thay cho phần được yêu cầu tự cài đặt.

### Q3–Q5
Được phép dùng PyTorch đầy đủ và GPU.

## Câu trả lời

Không có báo cáo Word/PDF riêng.

Các câu hỏi cần trả lời có ô:

```text
Câu trả lời của bạn
```

Hãy trả lời trực tiếp trong notebook và giữ lại biểu đồ, bảng và đầu ra cần thiết.

## Hai điểm lưu mô hình

- Q3 phải tạo `best_cnn.pt`.
- Q5 phải tạo `caption_rnn.pt`.

Các file này được dùng để kiểm tra cấu trúc bài nộp.

## Trước khi nộp

Chạy:

```bash
python check_submission.py
python collect_submission.py <MSSV>
```

`check_submission.py` chỉ kiểm tra **cấu trúc** bài nộp. Việc kiểm tra chạy thành công
không đồng nghĩa toàn bộ code chắc chắn đúng.
