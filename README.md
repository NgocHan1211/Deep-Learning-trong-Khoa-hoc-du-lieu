<div align="center">
  <img src="https://www.uit.edu.vn/media/Logo_UIT_Web_6d18902c3d.png" width="160"/>
</div>

---

# 🧠 Deep Learning trong Khoa Học Dữ Liệu – Báo Cáo Thực Hành

**Sinh viên:** Trần Ngọc Hân – `23520437`  
**Giảng viên:** Nguyễn Hiếu Nghĩa  
**Môn học:** Deep Learning trong Khoa Học Dữ Liệu

> 📁 Notebook và kết quả chi tiết: [Google Drive](https://drive.google.com/drive/folders/1KT7DKQEmHhYjRdcOUiB-U_vjt2fE-yW-?usp=sharing)

---

## 📋 Tổng Quan

| Lab | Tên bài | Mô hình | Dữ liệu |
|-----|---------|---------|---------|
| 1 | Mạng Neuron Đơn Giản | MLP (1-layer, 3-layer) | MNIST |
| 2 | Mạng Neural Tích Chập | GoogLeNet, ResNet-18, ResNet-50 | MNIST, VinaFood21 |
| 3 | RNN – Phân Loại Văn Bản & Gán Nhãn Chuỗi | LSTM, GRU, BiLSTM | UIT-VSFC, PhoNER |
| 4 | Mạng Neural Hồi Quy – Dịch Máy | Encoder-Decoder LSTM + Attention | PhoMT |
| 5 | Transformer Encoder | Transformer Encoder (3 lớp) | UIT-ViOCD, PhoNER |

---

## 🗂️ Lab 1 – Xây Dựng Mạng Neuron Đơn Giản

**Bộ dữ liệu:** [MNIST](http://yann.lecun.com/exdb/mnist/)  
**Độ đo đánh giá:** Accuracy, Precision, Recall, F1-macro

### Bài 1 – 1-layer MLP + Softmax

- Kiến trúc: 1 lớp fully connected, activation function: **Softmax**
- Optimizer: **SGD**
- Đánh giá kết quả trên từng chữ số (0–9)

<!-- TODO: thêm ảnh kết quả bài 1 vào đây -->

### Bài 2 – 3-layer MLP + ReLU + Softmax

- Kiến trúc: 3 lớp fully connected
  - 2 lớp đầu: activation **ReLU**
  - Lớp cuối: activation **Softmax**
- Optimizer: **SGD**
- Đánh giá kết quả trên từng chữ số (0–9)

<!-- TODO: thêm ảnh kết quả bài 2 vào đây -->

---

## 🗂️ Lab 2 – Mạng Neural Tích Chập (CNN)

**Độ đo đánh giá:** Precision, Recall, F1

### Bài 2 – GoogLeNet

- Xây dựng mô hình **GoogLeNet** (Multi-branch Network với Inception Blocks)
- Optimizer: **Adam**
- Lưu ý kiến trúc:
  - Lớp Conv đầu tiên: `padding=3`
  - Các lớp MaxPooling: `ceil_mode=True`

<!-- TODO: thêm ảnh Inception Block vào đây -->
<!-- TODO: thêm ảnh GoogLeNet architecture vào đây -->
<!-- TODO: thêm ảnh GoogLeNet parameters vào đây -->

### Bài 3 – ResNet-18 trên VinaFood21

- Xây dựng mô hình **ResNet-18** từ đầu
- Bộ dữ liệu: **VinaFood21**
- Optimizer: **Adam**
- Lưu ý kiến trúc:
  - Giữa các Residual Block có MaxPooling: `kernel=3, stride=2, padding=0`

<!-- TODO: thêm ảnh ResNet Block vào đây -->
<!-- TODO: thêm ảnh ResNet architecture vào đây -->
<!-- TODO: thêm ảnh ResNet-18 parameters vào đây -->

### Bài 4 – Fine-tune ResNet-50 (HuggingFace) trên VinaFood21

- Sử dụng pretrained **ResNet-50** từ HuggingFace
- Fine-tune trên bộ dữ liệu **VinaFood21**

---

## 🗂️ Lab 3 – RNN cho Phân Loại Văn Bản & Gán Nhãn Chuỗi

**Bộ dữ liệu:**
- [UIT-VSFC](https://drive.google.com/drive/folders/1rdcXNGt_3-QUvV8EtSvVsLMVeHmk6Yqk?usp=drive_link) – Vietnamese Student Feedback Corpus
- [PhoNER COVID-19](https://github.com/VinAIResearch/PhoNER_COVID19)

**Optimizer:** Adam | **Độ đo:** F1

### Bài 1 – LSTM 5 lớp (Phân loại văn bản)

- Kiến trúc: **5-layer LSTM**, hidden size = **256**
- Bộ dữ liệu: UIT-VSFC

### Bài 2 – GRU 5 lớp (Phân loại văn bản)

- Kiến trúc: **5-layer GRU**, hidden size = **256**
- Bộ dữ liệu: UIT-VSFC

### Bài 3 – BiLSTM Encoder (NER)

- Kiến trúc: Encoder với **5-layer BiLSTM**, hidden size = **256**
- Bài toán: **Named Entity Recognition**
- Bộ dữ liệu: PhoNER COVID-19

---

## 🗂️ Lab 4 – Encoder-Decoder cho Dịch Máy

**Bộ dữ liệu:** [PhoMT](https://drive.google.com/drive/folders/186OAOuSEYEDVcry7WP5UBdqECXo26QAb?usp=drive_link) – Dịch Anh → Việt  
**Optimizer:** Adam | **Độ đo:** ROUGE-L

### Bài 1 – Encoder-Decoder LSTM thuần

- Encoder: **3-layer LSTM**, hidden size = **256**
- Decoder: **3-layer LSTM**, hidden size = **256**

### Bài 2 – Encoder-Decoder + Bahdanau Attention

- Kiến trúc giống Bài 1, Decoder trang bị **Attention (Bahdanau)**
- Tham khảo: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

### Bài 3 – Encoder-Decoder + Luong Attention

- Kiến trúc giống Bài 1, Decoder trang bị **Attention (Luong)**
- Tham khảo: [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)

---

## 🗂️ Lab 5 – Transformer Encoder

**Độ đo:** F1

### Bài 1 – Transformer Encoder cho Phân loại Domain

- Kiến trúc: **3-layer Transformer Encoder** theo [Attention is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- Bài toán: Phân loại domain câu bình luận
- Bộ dữ liệu: [UIT-ViOCD](https://drive.google.com/drive/folders/1Lu9axyLkw7dMx80uLRgvCnZsmNzhJWAa?usp=sharing)

### Bài 2 – Transformer Encoder cho NER

- Kiến trúc: **3-layer Transformer Encoder** theo [Attention is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- Bài toán: Gán nhãn chuỗi (Sequence Labeling)
- Bộ dữ liệu: [PhoNER COVID-19](https://github.com/VinAIResearch/PhoNER_COVID19)

---

> 📁 Xem chi tiết notebook và kết quả thực nghiệm tại: [Google Drive](https://drive.google.com/drive/folders/1KT7DKQEmHhYjRdcOUiB-U_vjt2fE-yW-?usp=sharing)

*Trần Ngọc Hân – 23520437 | GVHD: Nguyễn Hiếu Nghĩa*
