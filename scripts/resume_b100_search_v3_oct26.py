"""Migrate saved broad-query responses to the user-approved 2/3-query, no-AND protocol.

Old snapshots are immutable. No successful query is regenerated. One bounded network
recovery pass fetches pending entries before the existing job continues to B teams.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import time
import run_b_medxpert100_oct26 as job


def migrate():
    source=job.DATA/'broad';target=job.DATA/'broad_search_v3'
    path=target/'preparation_manifest.json'
    if path.exists():return
    parent_raw=(source/'preparation_manifest.json').read_bytes()
    manifest=json.loads(parent_raw)
    accepted=[]
    for row in manifest['records']:
        if row['status']!='query_failed':continue
        # The failed format-only repair is preserved separately; use the original returned queries.
        call=row['model_calls'][-1]
        value=json.loads(call['raw_response']['choices'][0]['message']['content'])
        queries=job.broad.validate_queries(value.get('queries'))
        row['queries']=queries
        row['snapshots']=[{'query':q,'status':'fetch_pending','attempts':[]} for q in queries]
        row['status']='queried';row.pop('error',None)
        row['accepted_after_protocol_revision']={'original_status':'query_failed','query_count':len(queries),
            'reason':'user permits 2 or 3 queries and no explicit Boolean connector','at':job.base.now()}
        accepted.append(row['question_id'])
    # Copy original raw bodies and snapshots, never overwrite the failed old directory.
    shutil.copytree(source,target)
    config=manifest['config']
    config.update(script_sha256=job.sha(Path(job.broad.__file__).read_bytes()),
        retrieval_script_sha256=job.sha(Path(job.base.__file__).read_bytes()),
        models_code_sha256=job.sha((job.ROOT/'src/evidence_mdt/models.py').read_bytes()),
        query_protocol='stem-only-2-to-3-core-concepts-accept-2-or-3-no-required-boolean-v3')
    manifest.update(config=config,fingerprint=job.sha(job.base.canonical(config)),
                    status='incomplete',protocol_migration={'from':str(source),'source_manifest_sha256':job.sha(parent_raw),
                    'revalidated_original_responses':accepted,'regenerated_query_calls':0})
    job.write(path,manifest)
    job.write(target/'migration.json',manifest['protocol_migration'])


def main():
    if (job.RUN/'active.lock').exists():raise RuntimeError('Another B job may be active; verify before restart')
    migrate()
    job.append_jsonl(job.RUN/'restarts.jsonl',{'at':job.base.now(),'reason':'User-authorized no-AND and 2/3-query protocol; accept 19 original responses without regeneration, preserve old directory','job_script_sha256':job.sha(Path(job.__file__).read_bytes())})
    original=job.broad.prepare
    def recover(*args,**kwargs):
        result=original(*args,**kwargs)
        if result['status']=='complete':return result
        marker=job.RUN/'search_v3_network_recovery.json'
        if marker.exists():raise RuntimeError('One bounded v3 network recovery already used; inspect failures')
        failed=[r for r in result['records'] if r['status'] not in {'ready','not_triggered'}]
        # Never regenerate medically worded queries just to recover a download.
        if any(r['status']=='query_failed' for r in failed):raise RuntimeError('Unexpected query generation failure in v3')
        job.write(marker,{'at':job.base.now(),'question_ids':[r['question_id'] for r in failed],
                          'policy':'one recovery pass for saved queries only'})
        result=original(*args,**kwargs)
        return result
    job.broad.prepare=recover
    job.main()


if __name__=='__main__':main()
