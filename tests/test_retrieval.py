from evidence_mdt.retrieval import MockDense, Retriever
from evidence_mdt.schema import Evidence


def test_exclusion_happens_before_top50():
    # 50 个同词片段占满旧实现的召回名额；排除后第 51 个必须能进来。
    corpus = [Evidence(f"old-{i}", f"doc-{i}", "alpha", "local") for i in range(50)]
    corpus.append(Evidence("fresh", "fresh-doc", "alpha", "local"))
    retriever = Retriever(corpus, MockDense())
    result = retriever.search("alpha", exclude={f"old-{i}" for i in range(50)})
    assert [e.chunk_id for e in result] == ["fresh"]


def test_source_whitelist_and_empty_search():
    corpus = [Evidence("paper", "PMID:1", "alpha", "https://pubmed.ncbi.nlm.nih.gov/1/"),
              Evidence("project", "readme", "alpha", "https://github.com/example/readme")]
    retriever = Retriever(corpus, MockDense())
    assert [e.chunk_id for e in retriever.search("alpha")] == ["paper"]
    assert retriever.search("alpha", exclude={"paper"}) == []
    assert Retriever([], MockDense()).search("anything") == []
