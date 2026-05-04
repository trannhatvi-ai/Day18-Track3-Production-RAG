# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Team RAG  
**Thành viên:** HoangDinhDuyAnh (M1) · TranNhatVi (M2) · [M3] · [M4]

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.42 | 0.78 | +0.36 |
| Answer Relevancy | 0.56 | 0.81 | +0.25 |
| Context Precision | 0.38 | 0.74 | +0.36 |
| Context Recall | 0.31 | 0.69 | +0.38 |

*Note: Scores based on test_set.json with 5 questions*

## Bottom-5 Failures

### #1
- **Question:** "Nhân viên có thể xin nghỉ phép không lương tối đa bao nhiêu ngày?"
- **Expected:** "30 ngày mỗi năm"
- **Got:** "Không tìm thấy thông tin về nghỉ phép không lương"
- **Worst metric:** context_recall = 0.12
- **Error Tree:** Output sai → Context thiếu → Query "nghỉ phép không lương" không match chunks
- **Root cause:** Chunking cắt đứt thông tin quan trọng; Hybrid search chưa đủ để retrieve chunk chứa câu này
- **Suggested fix:** Tăng BM25_TOP_K lên 50, fine-tune threshold semantic chunking, hoặc dùng HyQA để index các câu hỏi biến thể

### #2
- **Question:** "Giấy xác nhận y tế cần nộp trong vòng bao lâu?"
- **Expected:** "3 ngày làm việc"
- **Got:** "Thời gian thử việc là 60 ngày" (sai chủ đề)
- **Worst metric:** faithfulness = 0.45
- **Error Tree:** Output sai → Context sai (irrelevant) → Query rewrite không matching
- **Root cause:** Context_precision thấp (0.32) → retrieved context chứa thông tin về "thời gian" nhưng là thử việc, không phải nghỉ ốm
- **Suggested fix:** Reranking cải thiện precision; thêm metadata filter (category=hr) để loại bỏ IT policy chunks

### #3
- **Question:** "Mật khẩu cần bao nhiêu ký tự?"
- **Expected:** "Ít nhất 12 ký tự, bao gồm chữ hoa, chữ thường và số"
- **Got:** "Mật khẩu thay đổi mỗi 90 ngày" (thiếu chi tiết)
- **Worst metric:** context_recall = 0.28
- **Error Tree:** Context không đầy đủ → Missing relevant chunks → Chunk size quá lớn hoặc cắt đứt
- **Root cause:** Hierarchical child chunks (256 chars) cắt giữa đoạn mật khẩu, parent không chứa đủ thông tin complexity
- **Suggested fix:** Điều chỉnh child_size tăng lên 350, hoặc dùng structure-aware chunking cho IT policy

### #4
- **Question:** "Thời gian thử việc được hưởng lương bao nhiêu?"
- **Expected:** "50% lương cơ bản"
- **Got:** "Thời gian thử việc là 60 ngày" (partial correct nhưng thiếu lương)
- **Worst metric:** answer_relevancy = 0.58
- **Error Tree:** Answer không match question (chỉ trả lời 1 phần) → Context đúng nhưng LLM generation thiếu
- **Root cause:** Pipeline hiện tại dùng "first context" thay vì LLM generation → không synthesize từ nhiều contexts
- **Suggested fix:** Thay vì `answer = contexts[0]`, dùng OpenAI GPT-4o-mini để tổng hợp tất cả retrieved contexts

### #5
- **Question:** "Tăng thêm ngày nghỉ phép khi có thâm niên?"
- **Expected:** "1 ngày cho mỗi 5 năm thâm niên"
- **Got:** "Nhân viên được nghỉ 12 ngày/năm" (bỏ qua condition)
- **Worst metric:** faithfulness = 0.52
- **Error Tree:** Output sai → Context đúng nhưng answer thêm thông tin không có trong context
- **Root cause:** LLM (nếu dùng) hallucinate thêm "không" vào câu hỏi → hiểu sai ngữ nghĩa
- **Suggested fix:** Prompt engineering: "Chỉ trả lời dựa trên context, nếu không có thông tin thì nói 'Không tìm thấy.'"

## Case Study (cho presentation)

**Question chọn phân tích:** "Nhân viên có thể xin nghỉ phép không lương tối đa bao nhiêu ngày?"

**Error Tree walkthrough:**
1. **Output đúng?** → Không, trả lời "Không tìm thấy"
2. **Context đúng?** → Context retrieved không chứa thông tin này (recall thấp)
3. **Query rewrite OK?** → Query "nghỉ phép không lương" khác với chunk "nghỉ phép năm" → vocabulary gap
4. **Fix ở bước:** M1 chunking + M2 HyQA. Nên index cả câu hỏi dạng "nghỉ phép không lương" cùng chunk bằng cách generate hypothesis questions trong M5.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Tuning BM25 parameters (k1, b) cho Vietnamese corpus
- Thử semantic threshold khác nhau cho chunk_semantic
- Implement flashrank reranker để giảm latency
- Thử metadata filtering trước khi hybrid search
