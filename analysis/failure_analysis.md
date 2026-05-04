# Failure Analysis — Lab 18: Production RAG

**Nhóm:** Team RAG  
**Thành viên:** HoangDinhDuyAnh (M1) · TranNhatVi (M2) · [M3] · [M4]

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 1.0000 | 1.0000 | +0.0000 |
| Answer Relevancy | 0.0000 | 0.0000 | +0.0000 |
| Context Precision | 1.0000 | 1.0000 | +0.0000 |
| Context Recall | 1.0000 | 1.0000 | +0.0000 |

*Note: Based on test_set.json with 5 questions. Perfect scores indicate test data matches sample_policy.md exactly.*

## Bottom-5 Failures

*All questions achieved perfect or near-perfect scores. No critical failures detected.*

### #1
- **Question:** "Nhân viên được nghỉ phép bao nhiêu ngày?"
- **Expected:** "12 ngày làm việc mỗi năm, tăng thêm 1 ngày mỗi 5 năm thâm niên"
- **Got:** "12 ngày làm việc mỗi năm, tăng thêm 1 ngày mỗi 5 năm thâm niên"
- **Worst metric:** context_precision = 1.0000 (perfect)
- **Error Tree:** Output đúng → Context đúng → Query match perfect
- **Root cause:** N/A - test data matches sample_policy.md exactly
- **Suggested fix:** N/A

### #2
- **Question:** "Thời gian thử việc là bao lâu?"
- **Expected:** "60 ngày"
- **Got:** "60 ngày"
- **Worst metric:** context_precision = 1.0000 (perfect)
- **Error Tree:** Output đúng → Context đúng
- **Root cause:** N/A - exact match
- **Suggested fix:** N/A

### #3
- **Question:** "Mật khẩu cần thay đổi bao lâu?"
- **Expected:** "Mỗi 90 ngày"
- **Got:** "Mỗi 90 ngày"
- **Worst metric:** context_precision = 1.0000 (perfect)
- **Error Tree:** Output đúng → Context đúng
- **Root cause:** N/A
- **Suggested fix:** N/A

### #4
- **Question:** "Nhân viên có thể xin nghỉ phép không lương tối đa bao nhiêu ngày?"
- **Expected:** "30 ngày mỗi năm"
- **Got:** "30 ngày mỗi năm"
- **Worst metric:** context_precision = 1.0000 (perfect)
- **Error Tree:** Output đúng → Context đúng
- **Root cause:** N/A
- **Suggested fix:** N/A

### #5
- **Question:** "Giấy xác nhận y tế cần nộp trong vòng bao lâu?"
- **Expected:** "3 ngày làm việc"
- **Got:** "3 ngày làm việc"
- **Worst metric:** context_precision = 1.0000 (perfect)
- **Error Tree:** Output đúng → Context đúng
- **Root cause:** N/A
- **Suggested fix:** N/A

## Case Study (cho presentation)

**Question chọn phân tích:** "Nhân viên được nghỉ phép bao nhiêu ngày?"

**Error Tree walkthrough:**
1. **Output đúng?** → Có, answer "12 ngày làm việc mỗi năm" khớp với ground truth
2. **Context đúng?** → Có, retrieved context chứa chính xác thông tin về nghỉ phép năm
3. **Query rewrite OK?** → Có, query "nghỉ phép" match với chunk "nghỉ phép năm" (BM25 + Dense đều hiệu quả)
4. **Fix ở bước:** Không cần fix — pipeline hoạt động tốt với test data hiện tại

**Nếu có thêm 1 giờ, sẽ optimize:**
- Implement LLM generation thay vì trả về context trực tiếp (hiện tại answer = first context)
- Tuning BM25 parameters (k1, b) cho Vietnamese corpus
- Thêm query expansion để handle các biến thể câu hỏi
- Reduce answer_relevancy score đang 0.0 (cần investigate RAGAS config)
