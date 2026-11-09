from evidence_mdt.models import MockModel
from evidence_mdt.retrieval import MockDense, Retriever
from evidence_mdt.runner import Runner, opaque_id, public_view
from evidence_mdt.schema import AgentTurn, Claim


def make_turn(name, role, round_id):
    return AgentTurn(name, role, round_id, 0, [], [], {}, "A",
                     {"A": 0.8, "B": 0.2}, [],
                     [Claim(name + ":c", "same finding", "A")], "summary", {})


def test_attribution_survives_merge_and_links_to_visible_opinion(tmp_path):
    runner = Runner(MockModel(), Retriever([], MockDense()), tmp_path)
    turns = [make_turn("internal:ordinary:a", "Radiologist", 0),
             make_turn("internal:ordinary:b", "Pathologist", 1),
             make_turn("internal:ordinary:c", "Radiologist", 2)]
    for semantic, count in [(False, 3), (True, 1)]:
        view = public_view(runner.message_view("q:repeat-1:natural:ordinary", turns, semantic))
        claims = [m for m in view if m["kind"] == "claim"]
        opinions = {m["id"]: m for m in view if m["kind"] == "opinion"}
        assert len(claims) == count
        attrs = [a for c in claims for a in c["attributions"]]
        assert len(attrs) == 3
        for turn, attr in zip(turns, attrs):
            assert attr == {"role": turn.role, "round_id": turn.round_id}
            assert opinions[opaque_id(turn.turn_id)]["role"] == attr["role"]
            assert opinions[opaque_id(turn.turn_id)]["round_id"] == attr["round_id"]
        assert all("sources" not in m and "member_claim_ids" not in m for m in view)
        assert "internal:ordinary" not in str(view)


def test_injected_claim_has_public_speaker_and_round_without_injection_marker(tmp_path):
    runner = Runner(MockModel(), Retriever([], MockDense()), tmp_path)
    message = {"id": "peer-message:private", "role": "Neurologist", "round_id": 1,
               "answer": "A", "claims": [{"text": "given finding", "answer": "A"}]}
    view = public_view(runner.message_view("q:repeat-1:natural:semantic", [message], True))
    claim = next(m for m in view if m["kind"] == "claim")
    assert claim["attributions"] == [{"role": "Neurologist", "round_id": 1}]
    assert "peer-message:private" not in str(view)
