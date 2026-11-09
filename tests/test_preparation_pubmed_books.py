import io
import json

import pytest

from preparation.corpus import fetch_pubmed
from preparation.state import Cache, digest


def responses(monkeypatch, ids, xml):
    import preparation.corpus as corpus

    class Response(io.BytesIO):
        pass

    def open_url(url, timeout):
        raw = json.dumps({"esearchresult": {"idlist": ids}}) if "esearch.fcgi" in url else xml
        return Response(raw.encode())

    monkeypatch.setattr(corpus.urllib.request, "urlopen", open_url)
    monkeypatch.setattr(corpus.time, "sleep", lambda _: None)


def test_book_and_journal_abstracts_are_usable_with_distinct_provenance(tmp_path, monkeypatch):
    xml = """<PubmedArticleSet>
      <PubmedArticle><MedlineCitation><PMID>101</PMID><Article>
        <ArticleTitle>Journal title</ArticleTitle><Abstract><AbstractText>Journal abstract.</AbstractText></Abstract>
      </Article></MedlineCitation></PubmedArticle>
      <PubmedBookArticle><BookDocument><PMID>202</PMID><ArticleTitle>Book chapter</ArticleTitle>
        <Abstract><AbstractText>Book <b>abstract</b>.</AbstractText></Abstract>
        <Sections><Section>Full chapter text must not enter the abstract.</Section></Sections>
        <ReferenceList><Reference><PMID>999</PMID></Reference></ReferenceList>
      </BookDocument></PubmedBookArticle>
      <PubmedBookArticle><BookDocument><PMID>303</PMID><Book><BookTitle>No abstract</BookTitle></Book>
      </BookDocument></PubmedBookArticle>
    </PubmedArticleSet>"""
    responses(monkeypatch, ["101", "202", "303"], xml)
    result = fetch_pubmed("fixture", Cache(tmp_path, 0))
    assert result["status"] == "ok"
    assert result["record_types"] == {"101": "journal_article", "202": "book_article", "303": "book_article"}
    assert result["without_abstract"] == ["303"]
    journal, book = result["evidence"]
    assert journal["position"] == "abstract"
    assert book["position"] == "book_abstract"
    assert book["text"] == "Book abstract."
    assert book["title"] == "Book chapter"
    assert book["content_hash"] == digest(b"Book abstract.")
    assert book["source"] == "https://pubmed.ncbi.nlm.nih.gov/202/"
    assert book["document_id"] == "PMID:202"


def test_reference_pmid_does_not_hide_missing_requested_record(tmp_path, monkeypatch):
    xml = """<PubmedArticleSet><PubmedBookArticle><BookDocument><PMID>202</PMID>
      <ReferenceList><Reference><PMID>999</PMID></Reference></ReferenceList>
    </BookDocument></PubmedBookArticle></PubmedArticleSet>"""
    responses(monkeypatch, ["202", "999"], xml)
    with pytest.raises(ValueError, match="EFetch omitted requested PMIDs: 999"):
        fetch_pubmed("fixture", Cache(tmp_path, 0))
