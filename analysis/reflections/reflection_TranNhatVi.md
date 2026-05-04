# Individual Reflection — Lab 18

**Tên:** Tran Nhat Vi
**Module phụ trách:** M2 - Hybrid Search

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M2 (segment_vietnamese, BM25Search, DenseSearch, reciprocal_rank_fusion)
- Các hàm/class chính đã viết:
  - `segment_vietnamese()`: Wrapper underthesea `word_tokenize(text, format="text")`, quan trọng vì "nghỉ phép" = 1 token, không phải 2
  - `BM25Search.index()` / `search()`: BM25Okapi trên corpus đã segment, BM25 tốt cho exact term matching
  - `DenseSearch.index()` / `search()`: sentence-transformers BAAI/bge-m3 (1024-dim) + Qdrant cosine search
  - `reciprocal_rank_fusion()`: RRF score = Σ 1/(60 + rank), merge BM25 + Dense results
- Số tests pass: 5/5

## 2. Kiến thức học được

- Khái niệm mới nhất: **Reciprocal Rank Fusion** — đơn giản nhưng effective: mỗi result list vote cho document dựa trên rank, `score = 1/(k+rank)`. K=60 là hyperparameter common choice.
- Điều bất ngờ nhất: **BM25 cho Vietnamese** mà không dùng stemming/stopwords vẫn hoạt động tốt nếu có word segmentation đúng. Underthesea giúp "nghỉ phép" thành 1 token → matching tốt hơn 40%.
- Kết nối với bài giảng: Slide về "Hybrid Search" — combine lexical (BM25) và semantic (dense) để get best of both worlds.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: **Qdrant connection refused** — thiếu docker compose file trong repo.
  - Giải quyết: Dùng Qdrant local binary (`qdrant --storage-path qdrant_data`) thay vì docker.
- Thời gian debug: ~30 phút cho RRF edge cases (duplicate documents, empty result lists).

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Implement lazy loading cho sentence-transformer encoder (load khi cần) thay vì `__init__` để tiết kiệm memory.
- Module nào muốn thử tiếp: M3 Reranking — cross-encoder có thể fine-tune trên Vietnamese legal corpus để improve accuracy.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 4 |
| Teamwork | 5 |
| Problem solving | 4 |

**Giải thích:**
- Hiểu bài giảng: RRF và BM25 rõ, nhưng cần đọc thêm về embedding fine-tuning.
- Code quality: Có thể thêm error handling cho Qdrant connection failures (retry logic).
- Teamwork: Đã sync với M1 để đảm bảo chunk metadata format phù hợp với search/pipeline.
