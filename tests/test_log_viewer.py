import importlib.util
import json
from pathlib import Path
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

spec = importlib.util.spec_from_file_location("log_viewer", Path(__file__).parents[1] / "scripts/log_viewer.py")
viewer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = viewer
spec.loader.exec_module(viewer)


def write(path, rows):
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")


def call(rid, task="answer", qid="803", repeat=1, **fields):
    return {"run_id":rid, "request":{"task":task, "question":{"question_id":qid,"stem":"题干 <script>坏内容</script>","options":{"A":"甲","B":"乙"}},"role":"expert","round_id":0},
            "wire_request":{"messages":[{"role":"system","content":"提示原文"}],"headers":{"Authorization":"credential"}},
            "response":{"choices":[{"message":{"content":'{"answer":"A"}'},"finish_reason":"stop"}]},
            "usage":{"total_tokens":12},"status":"ok",**fields}


@pytest.fixture
def store(tmp_path):
    folder=tmp_path / "batch"
    folder.mkdir()
    write(folder / "calls.jsonl", [
        call("803:repeat-1:recruit","recruit"),
        call("803:repeat-1:natural:ordinary"),
        call("803:repeat-1:natural:ordinary", "repair", attempt_id="retry"),
        call("803:repeat-2:recruit","recruit",repeat=2),
        call("803:repeat-2:natural:ordinary",repeat=2),
        call("8030:repeat-1:natural:ordinary",qid="8030"),
    ])
    write(folder / "results.jsonl",[{"run_id":"803:repeat-1:natural:ordinary","question_id":"803","answer":"A","gold":"B","status":"ok"}])
    return viewer.LogStore(tmp_path)


def test_exact_question_and_repeat_specific_shared_recruit(store):
    cases=store.search("803")["cases"]
    assert len(cases)==2
    case=store.case("batch","803:repeat-1:natural:ordinary")
    assert [c["task"] for c in case["calls"]]==["recruit","answer","repair"]
    assert case["meta"]["gold"]=="B"
    assert case["meta"]["status"]=="complete"
    assert "未记录attempt_id" in case["attempt_note"]
    assert case["calls"][-1]["attempt_id"]=="retry"


def test_full_run_id_filter_and_offsets(store):
    result=store.search("803:repeat-2:natural:ordinary")
    assert len(result["cases"])==1
    raw=store.call("batch",2)
    assert raw["request"]["question"]["stem"].startswith("题干")
    assert raw["wire_request"]["headers"]["Authorization"]=="[已隐藏]"
    assert raw["usage"]["total_tokens"]==12


def test_incomplete_append_and_refresh(store):
    path=store.root/"batch/calls.jsonl"
    with path.open("ab") as f:
        f.write(b'{"unfinished":')
    index=store.index("batch")
    assert len(index.entries)==6 and index.warnings
    with path.open("ab") as f:
        f.write(b'true}\n')
    assert len(store.index("batch").entries)==7


def test_direct_answer_results_without_calls(tmp_path):
    folder=tmp_path/"direct";folder.mkdir()
    write(folder/"results.jsonl",[{"question_id":"0","repeat_id":1,"request":{"messages":[{"role":"user","content":"题目与选项"}]},"response":{"choices":[{"message":{"content":"B"},"finish_reason":"stop"}]},"status":"ok","answer":"B","gold":"B"}])
    store=viewer.LogStore(tmp_path)
    assert store.batches()[0]["log_file"]=="results.jsonl"
    case=store.case("direct","0:repeat-1:direct")
    assert case["meta"]["status"]=="complete"
    assert store.call("direct",1)["response"]["choices"][0]["message"]["content"]=="B"


def test_results_update_refreshes_status_without_new_calls(store):
    assert store.case("batch","803:repeat-2:natural:ordinary")["meta"]["status"]=="logged"
    path=store.root/"batch/results.jsonl"
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps({'run_id':'803:repeat-2:natural:ordinary','question_id':'803','answer':'B','gold':'B','status':'ok'})+'\n')
    assert store.case("batch","803:repeat-2:natural:ordinary")["meta"]["status"]=="complete"


def test_recruit_only_attempt_is_visible(tmp_path):
    folder=tmp_path/'paused';folder.mkdir()
    write(folder/'calls.jsonl',[call('803:repeat-1:recruit','recruit')])
    store=viewer.LogStore(tmp_path)
    found=store.search('803')['cases']
    assert len(found)==1
    case=store.case('paused',found[0]['run_id'])
    assert len(case['calls'])==1 and case['calls'][0]['task']=='recruit'


@pytest.mark.parametrize("name",["..","../batch","batch/..","batch\\..","/tmp"])
def test_reject_arbitrary_paths(store,name):
    with pytest.raises(ValueError):
        store.index(name)


def test_http_routes_and_no_arbitrary_file_serving(store):
    server=ThreadingHTTPServer(("127.0.0.1",0),viewer.make_handler(store))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base+"/api/search?q=803") as r:
            assert json.load(r)["total"]==2
        with urllib.request.urlopen(base+"/") as r:
            assert "text/html" in r.headers["Content-Type"]
            assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]
            assert "模型对话" in r.read().decode()
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(base+"/.env")
        assert exc.value.code==404
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)


@pytest.fixture
def graph_store(tmp_path):
    folder = tmp_path / 'graph'; folder.mkdir()
    rid = '803:repeat-1:natural:ordinary'
    claim = {'claim_id':'c0','text':'evidence reason','answer':'A','citation_ids':['e1'],'polarity':'support','conditions':'condition'}
    initial = {'answer':'A','claims':[dict(claim)],'opinion':'initial opinion','probabilities':{'A':.7,'B':.3}}
    host_claim = {**claim,'claim_id':'hc','text':'host reason','citation_ids':['ref:x']}
    host = {**initial,'claims':[host_claim],'opinion':'host opinion'}
    events = [
        {'type':'attempt_start'},
        {'type':'evidence','chunk_id':'e1','text':'source text','title':'source title'},
        {'type':'turn','turn_id':'t0','role':'expert','round_id':0,'claims':[claim],**{k:v for k,v in initial.items() if k!='claims'},'raw_response':initial},
        {'type':'claim',**claim},
        {'type':'edge','source_id':'e1','target_id':'t0','relation':'exposed'},
        {'type':'edge','source_id':'e1','target_id':'c0','relation':'cited'},
        {'type':'cluster','id':'ref:x','claim':'evidence reason','member_claim_ids':['c0']},
        {'type':'edge','source_id':'c0','target_id':'ref:x','relation':'member_of'},
        {'type':'view_node','id':'ref:op','kind':'opinion','source_id':'t0'},
        {'type':'turn','turn_id':'host','role':'host','round_id':5,'claims':[host_claim],**{k:v for k,v in host.items() if k!='claims'},'raw_response':host},
        {'type':'edge','source_id':'ref:x','target_id':'hc','relation':'cited'},
    ]
    events = [{'run_id':rid,'attempt_id':'first',**e} for e in events]
    write(folder/'events.jsonl',events)
    write(folder/'calls.jsonl',[
        call(rid,attempt_id='first',response={'choices':[{'message':{'content':json.dumps(initial)}}]}),
        call(rid,'host',attempt_id='first',request={'task':'host','question':{'question_id':'803','stem':'stem','options':{'A':'alpha','B':'beta'}},'role':'host','round_id':5},response={'choices':[{'message':{'content':json.dumps(host)}}]}),
    ])
    write(folder/'results.jsonl',[{'run_id':rid,'attempt_id':'first','answer':'A','gold':'B','status':'ok'}])
    return viewer.LogStore(tmp_path),rid,events


def test_graph_preserves_claim_owner_and_explicit_citation_path(graph_store):
    store,rid,_=graph_store
    graph=store.graph('graph',rid)
    nodes={n['raw_id']:n for n in graph['nodes']}
    assert graph['counts']['claim']==2 # no duplication of embedded and standalone claim rows
    assert nodes['c0']['owner_id']==nodes['t0']['id']
    assert nodes['c0']['call_line']==1 and nodes['host']['call_line']==2
    assert nodes['c0']['data']['conditions']=='condition'
    edges={(e['source'],e['target'],e['relation']) for e in graph['edges']}
    for a,b,relation in [('e1','c0','cited'),('c0','ref:x','member_of'),('ref:x','hc','cited'),('hc','host','part_of'),('host','final','final_answer')]:
        assert (nodes[a]['id'],nodes[b]['id'],relation) in edges
    assert nodes['final']['answer']=='A' and graph['gold']=='B'
    assert not any(n['kind']=='gold' or n.get('label')=='标准答案' for n in graph['nodes'])
    assert nodes['host']['event_line']==10


def test_graph_attempt_isolation_including_same_turn_ids(graph_store):
    store,rid,events=graph_store
    folder=store.root/'graph'
    second=[{**e,'attempt_id':'second'} for e in events]
    second[1].update(chunk_id='e2',text='second evidence')
    write(folder/'events.jsonl',events+second)
    for attempt,expected,absent in [('first','e1','e2'),('second','e2',None)]:
        g=store.graph('graph',rid,attempt)
        assert g['attempt']==attempt and len(g['attempts'])==2
        assert any(n['raw_id']==expected and n['kind']=='evidence' for n in g['nodes'])
        if absent: assert not any(n['raw_id']==absent for n in g['nodes'])
    assert store.graph('graph',rid)['attempt']=='first' # successful result belongs to first
    assert store.graph('graph',rid,'second')['final_node'] is None
    with pytest.raises(ValueError): store.graph('graph',rid,'missing')


def test_cluster_versions_do_not_add_future_claims_to_earlier_citations(graph_store):
    store,rid,events=graph_store
    events += [
        {'run_id':rid,'attempt_id':'first','type':'claim','claim_id':'future','text':'later reason'},
        {'run_id':rid,'attempt_id':'first','type':'cluster','id':'ref:x','claim':'expanded later','member_claim_ids':['c0','future']},
    ]
    write(store.root/'graph/events.jsonl',events)
    g=store.graph('graph',rid)
    refs=[n for n in g['nodes'] if n['raw_id']=='ref:x']
    assert len(refs)==2
    hc=next(n for n in g['nodes'] if n['raw_id']=='hc')
    assert [e['source'] for e in g['edges'] if e['target']==hc['id'] and e['relation']=='cited']==[refs[0]['id']]
    assert 'future' not in refs[0]['data']['member_claim_ids']


def test_graph_updates_events_without_new_api_calls_and_skips_partial_line(graph_store):
    store,rid,events=graph_store
    graph=store.graph('graph',rid)
    path=store.root/'graph/events.jsonl'
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps({'run_id':rid,'attempt_id':'first','type':'edge','source_id':'unknown ref','target_id':'hc','relation':'cited'})+'\n')
        f.write('{"unfinished":')
    refreshed=store.graph('graph',rid)
    assert len(refreshed['edges'])==len(graph['edges'])+1
    assert refreshed['counts']['unresolved']==1
    assert any('无法解析' in w for w in refreshed['warnings'])


def test_fallback_has_no_rag_and_never_guesses_ref_author_by_matching_text(tmp_path):
    folder=tmp_path/'fallback';folder.mkdir();rid='32:repeat-1:natural:ordinary'
    claim={'text':'same words','answer':'A','citation_ids':['ref:unmapped']}
    payload={'task':'answer','role':'expert','round_id':0,'question':{'question_id':'32'},'evidence':[],
             'messages':[{'id':'ref:unmapped','kind':'claim','claim':'same words'}]}
    write(folder/'calls.jsonl',[call(rid,request=payload,response={'choices':[{'message':{'content':json.dumps({'answer':'A','claims':[claim]})}}]})])
    g=viewer.LogStore(tmp_path).graph('fallback',rid)
    assert g['source_mode']=='calls' and g['counts'].get('evidence',0)==0
    ref=next(n for n in g['nodes'] if n['raw_id']=='ref:unmapped')
    assert not any(e['target']==ref['id'] and e['relation'] in {'member_of','displayed_as'} for e in g['edges'])
    assert g['final_node'] is None and g['warnings']


def test_unknown_attempt_never_borrows_evidence_from_known_scope(graph_store):
    store,rid,events=graph_store
    events.append({'run_id':rid,'type':'evidence','chunk_id':'unknown_only','text':'unscoped'})
    write(store.root/'graph/events.jsonl',events)
    g=store.graph('graph',rid)
    assert g['attempt']=='first' and not any(n['raw_id']=='unknown_only' for n in g['nodes'])


def test_graph_rejects_arbitrary_batch_paths(store):
    with pytest.raises(ValueError): store.graph('../batch','803:repeat-1:natural:ordinary')


def test_graph_http_endpoint_and_fixed_assets(graph_store):
    store,rid,_=graph_store
    server=ThreadingHTTPServer(('127.0.0.1',0),viewer.make_handler(store))
    thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
    base=f'http://127.0.0.1:{server.server_port}'
    try:
        with urllib.request.urlopen(base+'/api/graph?batch=graph&run_id='+rid) as r:
            assert json.load(r)['counts']['turn']==2
        for asset in ['graph.js','graph.css']:
            with urllib.request.urlopen(base+'/'+asset) as r: assert r.status==200
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)


def test_graph_matches_numeric_string_compat_without_changing_raw_call(graph_store):
    store,rid,_=graph_store
    path=store.root/'graph/calls.jsonl'
    rows=viewer.small_jsonl(path)
    content=json.loads(rows[0]['response']['choices'][0]['message']['content'])
    content['probabilities']={'A':'0.7','B':'0.3'}
    rows[0]['response']['choices'][0]['message']['content']=json.dumps(content)
    write(path,rows)
    g=store.graph('graph',rid)
    assert next(n for n in g['nodes'] if n['raw_id']=='t0')['call_line']==1
    assert json.loads(store.call('graph',1)['response']['choices'][0]['message']['content'])['probabilities']['A']=='0.7'
    content['answer']='B';rows[0]['response']['choices'][0]['message']['content']=json.dumps(content);write(path,rows)
    assert next(n for n in store.graph('graph',rid)['nodes'] if n['raw_id']=='t0')['call_line'] is None


def test_direct_answer_graph_does_not_turn_gold_into_evidence(tmp_path):
    folder=tmp_path/'direct';folder.mkdir()
    write(folder/'results.jsonl',[{'question_id':'0','repeat_id':1,'request':{'messages':[{'role':'user','content':'question'}]},'response':{'choices':[{'message':{'content':'B'}}]},'status':'ok','answer':'B','gold':'A'}])
    graph=viewer.LogStore(tmp_path).graph('direct','0:repeat-1:direct')
    assert graph['source_mode']=='calls' and graph['counts'].get('evidence',0)==0
    assert graph['attempts'][0]['turns']==1
    final=next(n for n in graph['nodes'] if n['id']==graph['final_node'])
    assert final['answer']=='B' and graph['gold']=='A'
