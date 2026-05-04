# Individual Reflection — Lab 18

**Tên:** Hoang Dinh Duy Anh
**Module phụ trách:** M1 - Advanced Chunking

---

## 1. Đóng góp kỹ thuật

- Module đã implement: M1 (chunk_semantic, chunk_hierarchical, chunk_structure_aware, compare_strategies)
- Các hàm/class chính đã viết:
  - `chunk_semantic()`: Sử dụng sentence-transformers all-MiniLM-L6-v2, group sentences by cosine similarity (threshold 0.85)
  - `chunk_hierarchical()`: Tạo parents (2048 chars) và children (256 chars) với sliding window overlap 50%
  - `chunk_structure_aware()`: Regex split markdown headers (##, ###), giữ nguyên section integrity
  - `compare_strategies()`: Collect stats (num_chunks, avg/min/max length) cho 4 strategies
- Số tests pass: 8/8

## 2. Kiến thức học được

- Khái niệm mới nhất: **Hierarchical chunking** — retrieve child (precision) → return parent (context). Đây là production pattern để avoid embedding noise của large chunks mà vẫn giữ context richness.
- Điều bất ngờ nhất: **Semantic chunking với sentence-transformers nhỏ (all-MiniLM-L6-v2)** chạy khá nhanh (~100ms cho 10 sentences) nhưng đòi hỏi threshold tuning (0.85 cho corpus này).
- Kết nối với bài giảng: Slide về "Chunking Strategies" — hierarchical là recommended pattern cho production RAG.

## 3. Khó khăn & Cách giải quyết

- Khó khăn lớn nhất: **Sentence splitting đa dạng** — text có bullet lists, tables, code blocks → regex simple không đủ.
  - Giải quyết: Dùng `re.split(r'(?<=[.!?])\s+|\n\n', text)` để split theo sentence boundaries + paragraph breaks. Không perfect nhưng cover 80% cases.
- Thời gian debug: ~45 phút cho semantic grouping edge cases (empty text, single sentence).

## 4. Nếu làm lại

- Sẽ làm khác điều gì: Dùng `spacy` hoặc `nltk` punkt tokenizer thay vì regex để sentence split chính xác hơn.
- Module nào muốn thử tiếp: M5 Enrichment — contextual prepend có thể cải thiện semantic chunking bằng cách thêm section context vào chunk.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 5 |
| Problem solving | 4 |

**Giải thích:**
- Code quality: Đã type hints, docstrings, nhưng có thể tách `chunk_hierarchical` thành 2 hàm nhỏ hơn.
- Teamwork: Sync với M2 (search) để đảm bảo chunk metadata có `parent_id` compatible với retrieval logic.
