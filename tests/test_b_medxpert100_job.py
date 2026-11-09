"""Offline safeguards for the explicitly authorized one-off 100-question B job."""
import importlib.util
from pathlib import Path
import sys

import pytest

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
spec=importlib.util.spec_from_file_location('b100_job',ROOT/'scripts/run_b_medxpert100_oct26.py')
job=importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)
from evidence_mdt.schema import Question,Evidence


def test_exactly_300_ordinary_assignments_and_6000_output_limit():
    questions=[Question(str(i),'MedXpertQA','stem',{'A':'a','B':'b'},'B') for i in range(100)]
    assignments=job.assignments(questions)
    assert len(assignments)==len({a['run_id'] for a in assignments})==300
    assert {a['group'] for a in assignments}=={'ordinary'}
    assert {a['variant'] for a in assignments}=={'B:base','B:literal3','B:paraphrase3'}
    assert all(a['repeat_id']==1 for a in assignments)
    assert job.MAX_OUTPUT_TOKENS==6000


def test_b_material_changes_exposure_but_not_original():
    original=[Evidence('e'+str(i),'d'+str(i),'original '+str(i)) for i in range(6)]
    conditions={c:job.b_evidence(original,c,'e0',['same proposition differently','another faithful wording']) for c in job.CONDITIONS}
    assert [len(conditions[c]) for c in job.CONDITIONS]==[6,8,8]
    assert len(original)==6 and original[0].text=='original 0'
    assert all(e.derived_from=='e0' for e in conditions['paraphrase3'][-2:])
    assert all(len({e.chunk_id for e in values})==len(values) for values in conditions.values())


@pytest.mark.parametrize('parts',[[],['one'],['same',' SAME '],['original','other'],['', 'other']])
def test_reject_empty_identical_or_literal_paraphrases(parts):
    with pytest.raises(ValueError):job.valid_paraphrases({'paraphrases':parts},'original')


def test_accidental_runtime_retrieval_is_rejected():
    with pytest.raises(RuntimeError):job.NoRetrieval([]).search('query')


def test_protocol_migration_reuses_original_three_queries_and_keeps_old_bytes(tmp_path,monkeypatch):
    import json
    import resume_b100_search_v3_oct26 as resume
    monkeypatch.setattr(resume.job,'DATA',tmp_path)
    monkeypatch.setattr(resume.job,'ROOT',ROOT)
    source=tmp_path/'broad';source.mkdir()
    original=['first query','second query','third query']
    manifest={'config':{},'records':[{'question_id':'q','status':'query_failed','model_calls':[
        {'raw_response':{'choices':[{'message':{'content':json.dumps({'queries':original})}}]}}]}]}
    path=source/'preparation_manifest.json';path.write_text(json.dumps(manifest),encoding='utf-8');before=path.read_bytes()
    (source/'evidence.raw').write_bytes(b'old raw body')
    resume.migrate()
    migrated=json.loads((tmp_path/'broad_search_v3/preparation_manifest.json').read_bytes())
    assert path.read_bytes()==before
    assert migrated['records'][0]['queries']==original
    assert len(migrated['records'][0]['snapshots'])==3
    assert migrated['protocol_migration']['regenerated_query_calls']==0
    assert (tmp_path/'broad_search_v3/evidence.raw').read_bytes()==b'old raw body'
    resume.migrate() # idempotent, no second copy or regeneration


def test_material_api_keeps_gold_out_and_saves_raw_response(monkeypatch,tmp_path):
    import json
    monkeypatch.setattr(job,'RUN',tmp_path)
    monkeypatch.setenv('DEEPSEEK_API_KEY','test-placeholder')
    captured=[]
    raw={'choices':[{'message':{'content':'{"paraphrases":["one","two"]}'}}], 'usage':{'total_tokens':17}}
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return json.dumps(raw).encode()
    def open_request(request,timeout):captured.append(json.loads(request.data));return Response()
    monkeypatch.setattr(job.urllib.request,'urlopen',open_request)
    result=job.material_call('b_paraphrases','q',job.GENERATOR,{'original':{'text':'original'}})
    assert result['paraphrases']==['one','two']
    assert captured[0]['max_tokens']==6000 and captured[0]['model']=='deepseek-flash'
    assert 'gold' not in json.dumps(captured[0])
    saved=job.read(tmp_path/'material_calls.jsonl')[0]
    assert saved['response']==raw and saved['usage']['total_tokens']==17
    assert 'test-placeholder' not in (tmp_path/'material_calls.jsonl').read_text()
