#!/usr/bin/env python3
"""
Test runner for Lab 18 - verifies all module implementations
"""

import sys, os
sys.path.insert(0, '.')

def test_m1():
    from src.m1_chunking import chunk_basic, chunk_semantic, chunk_hierarchical, chunk_structure_aware, Chunk
    TEXT = """# Nghỉ phép

## Nghỉ phép năm

Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm.
Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên.

## Nghỉ phép không lương

Nhân viên có thể xin nghỉ phép không lương tối đa 30 ngày mỗi năm.

## Nghỉ ốm

Cần nộp giấy xác nhận y tế trong vòng 3 ngày làm việc."""

    basic = chunk_basic(TEXT)
    assert len(basic) > 0, "basic chunks empty"

    semantic = chunk_semantic(TEXT, threshold=0.5)
    assert len(semantic) > 0, "semantic chunks empty"
    assert all(isinstance(c, Chunk) for c in semantic)

    parents, children = chunk_hierarchical(TEXT, parent_size=200, child_size=80)
    assert len(parents) > 0 and len(children) > 0, "hierarchical empty"
    assert all(c.parent_id is not None for c in children), "children missing parent_id"
    parent_ids = {p.metadata.get("parent_id") for p in parents}
    assert all(c.parent_id in parent_ids for c in children), "invalid parent_ids"
    avg_c = sum(len(c.text) for c in children) / len(children)
    avg_p = sum(len(p.text) for p in parents) / len(parents)
    assert avg_c < avg_p, "children should be smaller than parents"

    structure = chunk_structure_aware(TEXT)
    assert len(structure) > 0, "structure chunks empty"
    assert any("section" in c.metadata for c in structure), "missing section metadata"

    print("M1: All chunking tests PASSED")
    return True


def test_m2():
    from src.m2_search import segment_vietnamese, BM25Search, reciprocal_rank_fusion, SearchResult

    CHUNKS = [
        {"text": "Nhân viên được nghỉ phép năm 12 ngày.", "metadata": {"source": "policy"}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "metadata": {"source": "it"}},
        {"text": "Thời gian thử việc là 60 ngày.", "metadata": {"source": "hr"}},
    ]

    seg = segment_vietnamese('Nhân viên được nghỉ phép năm')
    assert isinstance(seg, str) and len(seg) > 0

    bm25 = BM25Search()
    bm25.index(CHUNKS)
    results = bm25.search('nghỉ phép', top_k=2)
    assert len(results) > 0 and results[0].method == "bm25"
    assert 'nghỉ' in results[0].text.lower() or '12' in results[0].text

    a = [SearchResult('doc1', 0.9, {}, 'bm25'), SearchResult('doc2', 0.8, {}, 'bm25')]
    b = [SearchResult('doc2', 0.95, {}, 'dense'), SearchResult('doc3', 0.85, {}, 'dense')]
    merged = reciprocal_rank_fusion([a, b], top_k=3)
    assert len(merged) > 0 and merged[0].method == 'hybrid'
    assert any('doc2' in r.text for r in merged)

    print("M2: All search tests PASSED")
    return True


def test_m3():
    from src.m3_rerank import CrossEncoderReranker, RerankResult

    Q = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    DOCS = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "VPN dùng WireGuard AES-256.", "score": 0.6, "metadata": {}},
    ]

    reranker = CrossEncoderReranker()
    results = reranker.rerank(Q, DOCS, top_k=2)
    assert len(results) > 0 and len(results) <= 2
    assert all(isinstance(r, RerankResult) for r in results)
    if len(results) >= 2:
        assert results[0].rerank_score >= results[1].rerank_score
    if results:
        assert 'nghỉ' in results[0].text.lower() or '12' in results[0].text

    print("M3: All rerank tests PASSED")
    return True


def test_m4():
    from src.m4_eval import evaluate_ragas, failure_analysis, EvalResult

    result = evaluate_ragas(['q1'], ['a1'], [['c1']], ['gt1'])
    assert all(k in result for k in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall'])
    assert all(isinstance(result[k], (int, float)) for k in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall'])

    eval_results = [
        EvalResult('Q1', 'A1', ['C1'], 'GT1', 0.5, 0.6, 0.4, 0.3),
        EvalResult('Q2', 'A2', ['C2'], 'GT2', 0.9, 0.8, 0.7, 0.6),
    ]
    f = failure_analysis(eval_results, bottom_n=1)
    assert len(f) == 1
    assert 'diagnosis' in f[0] and 'suggested_fix' in f[0]

    print("M4: All eval tests PASSED")
    return True


def test_m5():
    from src.m5_enrichment import summarize_chunk, generate_hypothesis_questions, contextual_prepend, extract_metadata, enrich_chunks, EnrichedChunk

    SAMPLE = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm."

    s = summarize_chunk(SAMPLE)
    assert isinstance(s, str) and len(s) > 0

    qs = generate_hypothesis_questions(SAMPLE, n_questions=2)
    assert isinstance(qs, list)

    ctx = contextual_prepend(SAMPLE, "Sổ tay nhân viên")
    assert isinstance(ctx, str) and SAMPLE in ctx

    meta = extract_metadata(SAMPLE)
    assert isinstance(meta, dict)

    CHUNKS = [{"text": SAMPLE, "metadata": {"source": "test"}}]
    result = enrich_chunks(CHUNKS, methods=["contextual"])
    assert len(result) > 0 and isinstance(result[0], EnrichedChunk)
    assert result[0].original_text == SAMPLE

    print("M5: All enrichment tests PASSED")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("Lab 18: Testing Module Implementations")
    print("=" * 50)

    all_pass = True
    try:
        test_m1()
    except Exception as e:
        print(f"M1 FAILED: {e}")
        all_pass = False

    try:
        test_m2()
    except Exception as e:
        print(f"M2 FAILED: {e}")
        all_pass = False

    try:
        test_m3()
    except Exception as e:
        print(f"M3 FAILED: {e}")
        all_pass = False

    try:
        test_m4()
    except Exception as e:
        print(f"M4 FAILED: {e}")
        all_pass = False

    try:
        test_m5()
    except Exception as e:
        print(f"M5 FAILED: {e}")
        all_pass = False

    print("=" * 50)
    if all_pass:
        print("ALL MODULES PASSED!")
        sys.exit(0)
    else:
        print("SOME MODULES FAILED")
        sys.exit(1)
