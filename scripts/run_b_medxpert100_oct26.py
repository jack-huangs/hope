"""One-off ordinary-only B experiment. Preparation and answer-team costs stay separate.

Uses 100 newly selected MedXpertQA Text questions, frozen top-10 retrieval, one
random target per question, and three B conditions. Does not modify main prompts.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import hashlib
import io
import json
import os
import random
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/b_medxpert100_oct26"
RUN = ROOT / "runs/b_deepseek41flash_medxpert100_ordinary_oct26"
PREP = ROOT / "runs/b_medxpert100_preparation_oct26"
SNAPSHOT = RUN / "software_snapshot"
sys.path.insert(0, str(SNAPSHOT if (SNAPSHOT / "evidence_mdt").exists() else ROOT / "src"))
from evidence_mdt.data import load_questions
from evidence_mdt.env import load_env
from evidence_mdt.interventions import b_evidence
from evidence_mdt.models import create_model
from evidence_mdt.query_manifest import public_question_sha256
from evidence_mdt.retrieval import INITIAL_EVIDENCE_LIMIT, MedCPTDense, Retriever
from evidence_mdt.runner import Runner, append_jsonl
from evidence_mdt.schema import Budget, Evidence
import chunk_corpus
import prepare_relevance_corpus as base
import prepare_broad_corpus as broad

CONDITIONS = ("base", "literal3", "paraphrase3")
TARGET_SELECTION_SEED = 20261026
TARGET_SELECTION_POLICY = "uniform_random_frozen_top10_v1"
PARAPHRASE_POLICY = "generated_without_model_audit_v1"
MAX_OUTPUT_TOKENS = 6000
GENERATOR = """Write exactly two distinct English paraphrases of original.text. Return one JSON object with paraphrases: [string,string]."""
TARGET = """Annotate a evidence passage before any answer teams run. Using only the
supplied question and options, identify the option directly supported by
this passage. Do not pick a new passage. Do not infer missing support from medical knowledge.
Return JSON {"target_answer": option key or null, "reason": short string}.
Return null when no option has a sufficiently clear direct support relationship."""
QUERY_CHECK = """Check PubMed query faithfulness to the supplied stem; do not solve the question.
For each query, reject inferred diagnoses, symptoms, mechanisms, drugs or treatments absent from
the stem. Standard synonyms/abbreviations for explicitly stated facts are allowed.
Return JSON {"faithful":[one boolean per query],"reasons":[one short reason per query]}.
Do not use options, gold labels, or prior answers."""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    base.save_json(path, value)


def progress(stage, **fields):
    previous = json.loads((RUN / "summary.json").read_bytes()) if (RUN / "summary.json").exists() else {}
    if stage!='stopped_for_diagnosis':
        previous.pop('error',None);previous.pop('error_type',None)
    write(RUN / "summary.json", {**previous, "stage":stage,"updated_at":base.now(),**fields})
    print(json.dumps({"stage":stage,**fields},ensure_ascii=False),flush=True)


def curl_urlopen(request, timeout=180):
    """Isolated transport replacement for broken chunked urllib reads. No curl retries.

    Headers (including the API key) go over stdin, never command arguments or saved files.
    Payload bytes and response bytes are retained, and caller-side JSON/XML validation remains.
    """
    req=request if isinstance(request,urllib.request.Request) else urllib.request.Request(request)
    directory=RUN/'http';directory.mkdir(parents=True,exist_ok=True)
    name=uuid.uuid4().hex
    output=directory/(name+'.response.raw')
    config=['url = '+json.dumps(req.full_url),'request = '+json.dumps(req.get_method()),
            'header = "Connection: close"']
    for key,value in req.header_items():config.append('header = '+json.dumps(key+': '+value))
    if req.data is not None:
        body=directory/(name+'.request.body');body.write_bytes(req.data)
        config.append('data-binary = '+json.dumps('@'+body.as_posix(),ensure_ascii=False))
    result=subprocess.run([str(Path(os.environ.get('SystemRoot','C:/Windows'))/'System32/curl.exe'),
        '--config','-','--http1.1','--retry','0','--connect-timeout','20','--max-time',str(timeout),
        '--silent','--show-error','--output',str(output),'--write-out','%{http_code}'],
        input=('\n'.join(config)+'\n').encode('utf-8'),capture_output=True,timeout=timeout+30)
    raw=output.read_bytes() if output.exists() else b''
    status=result.stdout.decode('ascii',errors='replace').strip()
    append_jsonl(RUN/'transport.jsonl',{'at':base.now(),'transport':'curl-http1.1','returncode':result.returncode,
        'http_status':status,'response_file':str(output.relative_to(RUN)),'bytes':len(raw),
        'response_sha256':sha(raw),'method':req.get_method()})
    if result.returncode:
        raise urllib.error.URLError(f'curl transport exit={result.returncode}; raw response retained')
    if status!='200':
        raise urllib.error.HTTPError(req.full_url,int(status) if status.isdigit() else 502,
                                     'HTTP request failed; raw response retained',None,io.BytesIO(raw))
    return io.BytesIO(raw)


def assignments(questions):
    return [{"run_id":f"{q.question_id}:repeat-1:B:{condition}:ordinary", "question_id":q.question_id,
             "dataset":q.dataset,"repeat_id":1,"variant":"B:"+condition,"group":"ordinary"}
            for q in questions for condition in CONDITIONS]


def valid_paraphrases(value, original):
    parts = value.get("paraphrases")
    if not isinstance(parts,list) or len(parts)!=2 or any(not isinstance(x,str) or not x.strip() for x in parts):
        raise ValueError("Expected exactly two nonempty paraphrases")
    norm = lambda s:" ".join(s.casefold().split())
    if len({norm(x) for x in parts})!=2 or norm(original) in {norm(x) for x in parts}:
        raise ValueError("Paraphrases must differ from each other and from original")
    return parts


def material_call(stage, question_id, system, payload):
    # This is an independent preparation request; the expert/host prompts remain untouched.
    body={"model":"deepseek-flash","temperature":0,"max_tokens":MAX_OUTPUT_TOKENS,
          "thinking":{"type":"disabled"},"response_format":{"type":"json_object"},
          "messages":[{"role":"system","content":system},{"role":"user","content":json.dumps(payload,ensure_ascii=False)}]}
    for attempt in range(2):
        started=time.monotonic();raw=None
        record={"stage":stage,"question_id":question_id,"attempt":attempt+1,
                "wire_request":body,"started_at":time.time()}
        try:
            req=urllib.request.Request(os.environ.get("DEEPSEEK_BASE_URL","https://api.deepseek.com").rstrip('/')+'/chat/completions',
                data=json.dumps(body).encode(),headers={"Authorization":"Bearer "+os.environ['DEEPSEEK_API_KEY'],"Content-Type":"application/json"})
            with urllib.request.urlopen(req,timeout=180) as response:raw=json.load(response)
            record.update(response=raw,usage=raw.get('usage',{}))
            value=json.loads(raw['choices'][0]['message']['content'])
            if not isinstance(value,dict):raise ValueError("Expected JSON object")
            record['status']='ok'
            return value
        except Exception as exc:
            record.update(status='failed',error_type=type(exc).__name__,usage=(raw or {}).get('usage',{}))
            if attempt==1:raise
            time.sleep(1)
        finally:
            record['elapsed_seconds']=time.monotonic()-started
            append_jsonl(RUN/'material_calls.jsonl',record)


def wait_for_preparation():
    progress('waiting_initial_pubmed',answer_experiments_started=False)
    while True:
        path=PREP/'summary.json'
        state=json.loads(path.read_bytes()) if path.exists() else {}
        if state.get('stage')=='initial_materials_ready':return
        if state.get('stage')=='preparation_failed':raise RuntimeError('Initial preparation failed: '+str(state.get('error')))
        # A dead preparer must not leave this supervisor waiting forever.
        if path.exists() and time.time()-path.stat().st_mtime>1200:
            manifest=DATA/'relevance/preparation_manifest.json'
            if not manifest.exists() or time.time()-manifest.stat().st_mtime>1200:
                raise RuntimeError('Initial preparation has no checkpoint for 20 minutes; inspect process')
        time.sleep(10)


def freeze_short_queries(questions, tokenizer):
    destination=DATA/'initial_queries.jsonl'
    if destination.exists():return {r['question_id']:r['retrieval_query'] for r in read(destination)}
    manifest=json.loads((DATA/'queries/preparation_manifest.json').read_bytes())
    by_id={r['question_id']:r for r in manifest['records']}
    rows=[]
    audit_path=DATA/'query_faithfulness.jsonl'
    previous={r['question_id']:r for r in read(audit_path)} if audit_path.exists() else {}
    for q in questions:
        queries=by_id[q.question_id]['queries']
        if q.question_id not in previous:
            checked=material_call('query_faithfulness',q.question_id,QUERY_CHECK,
                {'stem':q.stem.split('\nAnswer Choices:',1)[0],'queries':queries})
            flags=checked.get('faithful')
            if not isinstance(flags,list) or len(flags)!=len(queries) or any(type(x) is not bool for x in flags):
                raise ValueError('Invalid query audit')
            # Keep complete original queries only. No truncation, inferred diagnosis, or quiet rewriting.
            retained=[term for term,flag in zip(queries,flags) if flag]
            combined=' ; '.join(retained)
            if combined and len(tokenizer(combined,truncation=False)['input_ids'])>64:
                retained=next(([term] for term in retained if len(tokenizer(term,truncation=False)['input_ids'])<=64),[])
                combined=' ; '.join(retained)
            repaired=None
            if not combined:
                repaired=material_call('query_repair',q.question_id,
                    'Return JSON {"queries":[one concise English PubMed query]}. Use only explicitly stated '
                    'stem findings and the information asked for. Do not infer a diagnosis, mechanism, '
                    'treatment or unstated symptom. Keep the query within 64 tokens.',
                    {'stem':q.stem.split('\nAnswer Choices:',1)[0]})
                corrected=repaired.get('queries')
                if not isinstance(corrected,list) or len(corrected)!=1 or not isinstance(corrected[0],str) or not corrected[0].strip():
                    raise ValueError('Invalid repaired query')
                verified=material_call('query_repair_audit',q.question_id,QUERY_CHECK,
                    {'stem':q.stem.split('\nAnswer Choices:',1)[0],'queries':corrected})
                if verified.get('faithful')!=[True] or type(verified['faithful'][0]) is not bool or len(tokenizer(corrected[0],truncation=False)['input_ids'])>64:
                    raise RuntimeError(f'No faithful query within 64 tokens after one repair: {q.question_id}')
                combined=corrected[0]
                repaired={'queries':corrected,'audit':verified,
                          'note':'Local retrieval input repaired once; rejected-query abstracts excluded; corpus coverage stage will search stem-only broad queries'}
            previous[q.question_id]={'question_id':q.question_id,'original_queries':queries,'audit':checked,
                                      'retained_queries':retained,'retrieval_query':combined,'repair':repaired}
            append_jsonl(audit_path,previous[q.question_id])
        rows.append({'question_id':q.question_id,'dataset':q.dataset,'public_question_sha256':public_question_sha256(q),
                     'retrieval_query':previous[q.question_id]['retrieval_query']})
        progress('checking_queries',checked=len(rows),total=100)
    base.write_jsonl(destination,rows)
    return {r['question_id']:r['retrieval_query'] for r in rows}


def prepare_retrieval(questions):
    from transformers import AutoTokenizer
    models=json.loads((ROOT/'indexes/model_download_manifest.json').read_bytes())['models']
    qt=AutoTokenizer.from_pretrained(models['ncbi/MedCPT-Query-Encoder']['snapshot_path'],local_files_only=True)
    lookup=freeze_short_queries(questions,qt)
    # Filter the initial snapshot to faithful query rows before building the shared corpus.
    src=DATA/'relevance';filtered=DATA/'faithful_relevance'
    if not (filtered/'preparation_manifest.json').exists():
        parent=json.loads((src/'preparation_manifest.json').read_bytes())
        keep={r['question_id']:set(r['retained_queries']) for r in read(DATA/'query_faithfulness.jsonl')}
        rows=[];union={};sources={}
        for r in parent['records']:
            retained=[(q,s) for q,s in zip(r['queries'],r['snapshots']) if q in keep[r['question_id']]]
            child={**r,'queries':[q for q,s in retained],'snapshots':[s for q,s in retained]}
            for q,s in retained:
                for name in ('evidence_file','snapshot_file','raw_xml_file'):
                    if s.get(name):
                        filtered.mkdir(parents=True,exist_ok=True)
                        shutil.copyfile(src/s[name],filtered/s[name])
                for e in read(src/s['evidence_file']):
                    union[e['chunk_id']]=e
                    sources.setdefault(e['chunk_id'],[]).append({'question_id':r['question_id'],'query':q})
            rows.append(child)
        filtered.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src/'questions.jsonl',filtered/'questions.jsonl')
        base.write_jsonl(filtered/'corpus.jsonl',list(union.values()))
        parent.update(records=rows,chunk_sources=sources,corpus_chunks=len(union),
                      corpus_sha256=sha((filtered/'corpus.jsonl').read_bytes()),
                      faithfulness_note='Model audit of queries; rejected queries and their unique abstracts excluded')
        write(filtered/'preparation_manifest.json',parent)
    progress('pre_answer_coverage_retrieval')
    corpus=broad.prepare(filtered,DATA/'broad_search_v3',max_model_calls=200,max_tokens=500000,retries=1)
    if corpus['status']!='complete':raise RuntimeError('Pre-answer coverage retrieval incomplete')
    at=AutoTokenizer.from_pretrained(models['ncbi/MedCPT-Article-Encoder']['snapshot_path'],local_files_only=True)
    progress('chunking_and_indexing')
    chunked=chunk_corpus.prepare(DATA/'broad_search_v3',DATA/'chunks',at,models['ncbi/MedCPT-Article-Encoder']['revision'],limit=399)
    dense=MedCPTDense(ROOT/'indexes/model_download_manifest.json')
    evidence=[Evidence(**r) for r in read(DATA/'chunks/corpus.jsonl')]
    vec=DATA/'vectors.pt'
    if vec.exists():dense.load_vectors(vec)
    retriever=Retriever(evidence,dense)
    if not vec.exists():dense.torch.save(dense.vectors,vec)
    snapshots=DATA/'retrieval_snapshots.jsonl'
    frozen={r['question_id']:r for r in read(snapshots)} if snapshots.exists() else {}
    for q in questions:
        if q.question_id in frozen:
            if frozen[q.question_id].get('initial_evidence_limit') != INITIAL_EVIDENCE_LIMIT:
                raise ValueError('Frozen initial evidence protocol differs; use a new preparation directory')
            continue
        top=retriever.search(lookup[q.question_id],limit=INITIAL_EVIDENCE_LIMIT)
        if not top:raise RuntimeError('Expected nonempty initial evidence for '+q.question_id)
        frozen[q.question_id]={'question_id':q.question_id,'initial_evidence_limit':INITIAL_EVIDENCE_LIMIT,'evidence':[asdict(e) for e in top],'trace':dict(retriever.last_trace)}
        append_jsonl(snapshots,frozen[q.question_id]);progress('freezing_top10',ready=len(frozen),total=100)
    return frozen


def select_target(question_id, snapshot):
    evidence = snapshot['evidence']
    if not evidence or len(evidence) > INITIAL_EVIDENCE_LIMIT:
        raise ValueError('Random target selection requires one to ten frozen evidence chunks')
    rng = random.Random(f'{TARGET_SELECTION_SEED}:{question_id}:B-target')
    return rng.choice(evidence)


def prepare_specifications(questions, snapshots):
    path=DATA/'b_specifications.jsonl'
    specs={r['question_id']:r for r in read(path)} if path.exists() else {}
    # Validate all saved stimuli before any paid call; never mix target protocols.
    for q in questions:
        if q.question_id not in specs:
            continue
        saved = specs[q.question_id]
        expected = select_target(q.question_id, snapshots[q.question_id])
        if (saved.get('target_policy') != TARGET_SELECTION_POLICY
                or saved.get('target_selection_seed') != TARGET_SELECTION_SEED
                or saved.get('b_target_id') != expected['chunk_id']):
            raise ValueError('Existing B materials use a different target policy; use new material and run directories')
        if saved.get('paraphrase_policy') != PARAPHRASE_POLICY:
            raise ValueError('Existing B materials use a different paraphrase policy; use new material and run directories')
    for q in questions:
        if q.question_id in specs:continue
        target=select_target(q.question_id, snapshots[q.question_id])
        value=material_call('b_paraphrases',q.question_id,GENERATOR,{'original':target})
        texts=valid_paraphrases(value,target['text'])
        annotation=material_call('b_target_annotation',q.question_id,TARGET,
            {'question':q.public(),'original':target})
        target_answer=annotation.get('target_answer')
        if target_answer is not None and target_answer not in q.options:
            raise ValueError('Invalid target option annotation')
        specs[q.question_id]={'question_id':q.question_id,'b_target_id':target['chunk_id'],
            'target_policy':TARGET_SELECTION_POLICY,'target_selection_seed':TARGET_SELECTION_SEED,
            'target_answer':target_answer,'target_annotation':annotation,'b_paraphrases':texts,
            'paraphrase_policy':PARAPHRASE_POLICY,
            'review_status':'model-generated; no semantic or medical verification'}
        append_jsonl(path,specs[q.question_id]);progress('preparing_b_paraphrases',ready=len(specs),total=100)
    return specs


class NoRetrieval:
    """Only frozen evidence is allowed during team runs; any accidental search fails loudly."""
    def __init__(self,evidence):self.evidence=evidence
    def search(self,*args,**kwargs):raise RuntimeError('Runtime retrieval forbidden in ordinary-only B experiment')


def run_teams(questions, snapshots, specs):
    config={'model':'deepseek-flash','documented_service_version':'DeepSeek-V4.1-Flash-1017',
        'groups':['ordinary'],'conditions':list(CONDITIONS),'repeats':1,'experts':4,'discussion_rounds':4,
        'temperature':0.3,'max_output_tokens':MAX_OUTPUT_TOKENS,'thinking':'disabled','runtime_retrieval':False,
        'question_sha256':sha((DATA/'questions.jsonl').read_bytes()),
        'snapshots_sha256':sha((DATA/'retrieval_snapshots.jsonl').read_bytes()),
        'specifications_sha256':sha((DATA/'b_specifications.jsonl').read_bytes()),
        'source_hashes':{p.name:sha(p.read_bytes()) for p in (SNAPSHOT/'evidence_mdt').glob('*.py')},
        'target_selection_policy':TARGET_SELECTION_POLICY,'target_selection_seed':TARGET_SELECTION_SEED,
        'paraphrase_policy':PARAPHRASE_POLICY,
        'b_target_answer_policy':'Pre-answer model annotation from public question and fixed randomly selected top10 passage only; null when direct mapping unclear; never gold or team outcomes',
        'max_calls':15000,'max_reported_tokens':100000000}
    fingerprint=sha(base.canonical(config))
    config_path=RUN/'config.json'
    if config_path.exists() and json.loads(config_path.read_bytes())!={**config,'fingerprint':fingerprint}:
        raise ValueError('Frozen answer configuration differs; refuse mixed-config resume')
    write(config_path,{**config,'fingerprint':fingerprint})
    planned=[{**a,'fingerprint':fingerprint} for a in assignments(questions)]
    base.write_jsonl(RUN/'assignments.jsonl',planned)
    completed={r['run_id'] for r in read(RUN/'results.jsonl') if r.get('status')=='ok'} if (RUN/'results.jsonl').exists() else set()
    failed={r['run_id'] for r in read(RUN/'failures.jsonl')} if (RUN/'failures.jsonl').exists() else set()
    previous=read(RUN/'calls.jsonl') if (RUN/'calls.jsonl').exists() else []
    budget=Budget(config['max_calls'],config['max_reported_tokens'],len(previous),sum(r.get('usage',{}).get('total_tokens',0) or 0 for r in previous))
    all_evidence={e['chunk_id']:Evidence(**e) for s in snapshots.values() for e in s['evidence']}
    model=create_model('deepseek','deepseek-flash');model.timeout=300
    runner=Runner(model,NoRetrieval(list(all_evidence.values())),RUN,budget);runner.fingerprint=fingerprint
    consecutive=0
    for q in questions:
        evidence=[Evidence(**e) for e in snapshots[q.question_id]['evidence']]
        spec=specs[q.question_id]
        runner.b_target_answer=spec.get('target_answer')
        for condition in CONDITIONS:
            rid=f'{q.question_id}:repeat-1:B:{condition}:ordinary'
            if rid in completed or rid in failed:continue
            try:
                exposed=b_evidence(evidence,condition,spec['b_target_id'],spec['b_paraphrases'])
                runner.run_group(q,1,'ordinary',exposed,'B:'+condition)
                completed.add(rid);consecutive=0
            except Exception as exc:
                append_jsonl(RUN/'failures.jsonl',{'run_id':rid,'question_id':q.question_id,'variant':'B:'+condition,
                    'group':'ordinary','status':'technical_failure','error':repr(exc),'fingerprint':fingerprint,
                    'calls':budget.calls,'tokens':budget.tokens,'attempt_id':runner.current_attempt_id})
                failed.add(rid);consecutive+=1
                if consecutive>=3:raise RuntimeError('Three consecutive technical failures; stopped for diagnosis') from exc
            results=read(RUN/'results.jsonl') if (RUN/'results.jsonl').exists() else []
            by_condition={c:{'completed':sum(r['variant']=='B:'+c for r in results),
                'correct':sum(r['variant']=='B:'+c and r['answer']==r['gold'] for r in results)} for c in CONDITIONS}
            progress('running_teams',answer_experiments_started=True,completed=len(completed),failed=len(failed),
                     planned=300,by_condition=by_condition,calls=budget.calls,known_tokens=budget.tokens)
    # Verify actual records, not merely process exit or the assignment count.
    events=read(RUN/'events.jsonl');results=read(RUN/'results.jsonl') if (RUN/'results.jsonl').exists() else []
    for r in results:
        turns=[e for e in events if e.get('type')=='turn' and e.get('run_id')==r['run_id'] and e.get('attempt_id')==r.get('attempt_id')]
        if len(turns)!=21:raise RuntimeError('Expected 21 saved turns per successful team')
    calls=read(RUN/'calls.jsonl')
    if any(r.get('request',{}).get('task') not in {'recruit','answer','host','repair'} for r in calls):
        raise RuntimeError('Unexpected semantic or retrieval model task')
    progress('complete' if len(completed)==300 else 'finished_with_technical_failures',
             completed=len(completed),failed=len(failed),audit_21_turns=True,zero_runtime_retrieval=True)


def main():
    RUN.mkdir(parents=True,exist_ok=True)
    # Exclusive lock avoids concurrent supervisors and duplicate paid team runs.
    lock=RUN/'active.lock'
    with lock.open('x',encoding='ascii') as f:f.write(str(os.getpid()))
    try:
        (RUN/'process.pid').write_text(str(os.getpid()),encoding='ascii')
        load_env(ROOT/'.env')
        os.environ.update(DEEPSEEK_MODEL='deepseek-flash',MDT_MAX_OUTPUT_TOKENS=str(MAX_OUTPUT_TOKENS),MDT_ANSWER_TEMPERATURE='0.3',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
        urllib.request.urlopen=curl_urlopen
        wait_for_preparation()
        questions=load_questions(DATA/'questions.jsonl')
        if len(questions)!=100 or len({q.question_id for q in questions})!=100 or any(q.dataset!='MedXpertQA' for q in questions):
            raise ValueError('Expected exactly 100 unique MedXpertQA questions')
        snapshots=prepare_retrieval(questions)
        specs=prepare_specifications(questions,snapshots)
        run_teams(questions,snapshots,specs)
    except Exception as exc:
        progress('stopped_for_diagnosis',error_type=type(exc).__name__,error=str(exc)[:600])
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
