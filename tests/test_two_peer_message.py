import pytest

from evidence_mdt.interventions import c_message
from evidence_mdt.models import MockModel
from evidence_mdt.retrieval import MockDense, Retriever
from evidence_mdt.runner import Runner
from evidence_mdt.schema import Question


def test_two_peer_condition_preserves_paired_single_message_settings():
    for i in range(20):
        question = Question(str(i), "synthetic", "Choose", {"A": "one", "B": "two", "C": "three"}, "A")
        for wrong in (None, "C"):
            single = c_message(question, 1, "Choose {wrong_label}", False, wrong)
            double = c_message(question, 1, "Choose {wrong_label}", True, wrong)
            for field in ("speaker", "wrong_option", "round_id", "text"):
                assert single[field] == double[field]
            assert 'receiver' not in single and 'receiver' not in double
            assert len({double['speaker'],double['second_speaker']})==2
            assert double == c_message(question, 1, "Choose {wrong_label}", True, wrong)


@pytest.mark.parametrize("second", [None, -1, 1, 4, True])
def test_invalid_second_peer_rejected_before_any_model_call(tmp_path, second):
    runner = Runner(MockModel(), Retriever([], MockDense()), tmp_path)
    question = Question("q", "synthetic", "Choose", {"A": "one", "B": "two"}, "A")
    with pytest.raises(ValueError, match="two distinct speakers"):
        runner.run_group(question, 1, "ordinary", [], variant="C:repeated_message",
                         message_injection={"speaker": 1,
                                            "second_speaker": second, "repeat_message": True})
    assert not runner.call_log.exists()


def test_reject_legacy_directed_config_before_any_call(tmp_path):
    runner=Runner(MockModel(),Retriever([],MockDense()),tmp_path)
    q=Question('q','synthetic','Choose',{'A':'one','B':'two'},'A')
    with pytest.raises(ValueError,match='do not accept a receiver'):
        runner.run_group(q,1,'ordinary',[],message_injection={'receiver':0,'speaker':1})
    assert not runner.call_log.exists()


@pytest.mark.parametrize('group',['ordinary','semantic'])
@pytest.mark.parametrize('round_id',[1,3,4])
@pytest.mark.parametrize('repeat',[False,True])
def test_public_error_persists_for_every_later_expert_and_host(tmp_path,group,round_id,repeat):
    import json
    from evidence_mdt.runner import opaque_id
    class Citing(MockModel):
        def complete(self,payload):
            result=super().complete(payload)
            if payload.get('task') in {'answer','host'}:
                visible=[m for m in payload.get('messages',[]) if m.get('claim')=='Public erroneous assertion']
                expected=payload.get('task')=='host' or payload['round_id']>=round_id
                assert bool(visible)==expected # early rounds cannot see a future injection
                if visible:
                    result.content['citations']=[visible[0]['id']]
                    result.content['claims'][0]['citation_ids']=[visible[0]['id']]
            return result
    runner=Runner(Citing(),Retriever([],MockDense()),tmp_path)
    q=Question('q','synthetic','Choose',{'A':'one','B':'two'},'A')
    injected={'speaker':1,'wrong_option':'B','round_id':round_id,
              'text':'Public erroneous assertion','repeat_message':repeat}
    if repeat:injected['second_speaker']=2
    runner.run_group(q,1,group,[],variant='C:repeated_message' if repeat else 'C:message_error',message_injection=injected)
    events=[json.loads(x) for x in runner.event_log.read_text(encoding='utf-8').splitlines()]
    synthetic=[e for e in events if e['type']=='injected_message']
    assert len(synthetic)==(2 if repeat else 1)
    assert len({e['role'] for e in synthetic})==len(synthetic)
    turns=[e for e in events if e['type']=='turn'];assert len(turns)==21
    for turn in turns:
        payload=turn['visible_payload']
        visible=[m for m in payload['messages'] if m.get('claim')=='Public erroneous assertion']
        if turn['round_id']<round_id:assert not visible;continue
        assert len(visible)==(2 if repeat and group=='ordinary' else 1)
        attribution={a['role'] for m in visible for a in m['attributions']}
        assert attribution=={e['role'] for e in synthetic}
        assert all(opaque_id(e['id']) in {m['id'] for m in payload['messages']} for e in synthetic)
        serialized=json.dumps(payload)
        assert 'receiver' not in serialized and 'wrong_option' not in serialized
    first_injected=next(i for i,e in enumerate(events) if e['type']=='injected_message')
    assert all(e['round_id']<round_id for e in events[:first_injected] if e['type']=='turn')
    injection=next(e for e in events if e['type']=='injection')
    assert injection['visibility']=='public' and 'receiver' not in injection
    from scripts.audit_run import audit
    assert audit(tmp_path)['issue_count']==0
