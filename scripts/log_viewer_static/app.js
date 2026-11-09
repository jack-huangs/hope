/* 全部日志内容通过textContent呈现，不执行日志中的HTML或指令。 */
const $ = id => document.getElementById(id);
const state = {cases:[], current:null, call:null, calls:[], batches:[], searchSeq:0, caseSeq:0, callSeq:0, view:new URLSearchParams(location.search).get('view')==='graph'?'graph':'dialogue', graph:null};
const taskNames = {recruit:'专家招募',answer:'专家作答',host:'主持人汇总',repair:'格式修复',review:'补检索复核',queries:'生成检索词',novelty:'判断新信息',semantic:'语义关系判断',semantic_batch:'批量语义判断',direct:'单模型直接作答'};
const roleNames = {'obstetrician':'产科专家','neonatologist':'新生儿科专家','pediatric surgeon':'小儿外科专家','geneticist':'遗传学专家','medical geneticist':'医学遗传专家','neurologist':'神经科专家','neuropathologist':'神经病理专家','infectious disease specialist':'感染科专家','geriatrician':'老年科专家','host':'主持人','pharmacist':'药师','urologist':'泌尿外科专家','oncologist':'肿瘤科专家','nephrologist':'肾内科专家','endocrinologist':'内分泌科专家','cardiologist':'心脏科专家','psychiatrist':'精神科专家','clinical pharmacologist':'临床药理专家','internist':'内科专家'};
Object.assign(roleNames,{'emergency medicine physician':'急诊科专家','general surgeon':'普外科专家','gastroenterologist':'消化科专家','radiologist':'影像科专家','clinical pharmacist':'临床药师','primary care physician':'全科专家'});
const phase = e => e.task==='repair' ? '格式修复' : e.shared ? '专家招募' : e.task==='host' ? '主持人汇总' : e.task==='review' ? '补检索复核' : e.round===0 ? '独立作答 · R0' : typeof e.round==='number' ? `第 ${e.round} 轮讨论` : taskNames[e.task]||e.task;
const role = value => roleNames[String(value).toLowerCase()] || value || '';
const fmt = n => typeof n==='number' ? n.toLocaleString('zh-CN') : '—';
function node(tag,text,cls){const n=document.createElement(tag);if(text!==undefined&&text!==null)n.textContent=String(text);if(cls)n.className=cls;return n;}
function add(parent,tag,text,cls){const n=node(tag,text,cls);parent.appendChild(n);return n;}
function clear(parent){parent.replaceChildren();return parent;}
function jsonText(value){return typeof value==='string'?value:JSON.stringify(value,null,2);}
function parse(value){if(typeof value!=='string')return value;try{return JSON.parse(value)}catch{return value}}
function loading(parent,text){clear(parent);add(parent,'div',text,'loading');}
function details(parent,title,value,open=false){const d=add(parent,'details');d.open=open;add(d,'summary',title);add(d,'pre',jsonText(value));return d;}
function toast(message){$('toast').textContent=message;$('toast').classList.add('show');setTimeout(()=>$('toast').classList.remove('show'),2000);}
async function copy(value){try{await navigator.clipboard.writeText(value);toast('已复制')}catch{toast('浏览器未允许复制，请手动选择文字')}}
async function api(path,params={}){const r=await fetch(path+'?'+new URLSearchParams(params));const data=await r.json();if(!r.ok)throw Error(data.error||'读取失败');return data;}
function pill(parent,text,kind){return add(parent,'span',text,'pill '+kind);}
function status(meta){return meta.status==='complete'?['已完成','good']:meta.status==='failed'?['有失败记录','bad']:['已有调用','wait'];}
function setNotice(text,error=false){$('notice').textContent=text;$('notice').className=error?'error':'';}

async function search(chooseFirst=true){
  const seq=++state.searchSeq;
  state.caseSeq++;state.callSeq++;
  $('search-form').querySelector('button').disabled=true;
  setNotice($('batch').value==='all'?'正在搜索本地批次，首次读取稍需等待…':'正在读取本批次日志…');
  try{
    const data=await api('/api/search',{q:$('query').value,batch:$('batch').value});
    if(seq!==state.searchSeq)return;
    state.cases=data.cases; $('result-count').textContent=data.total;
    const box=clear($('case-list'));
    if(!data.cases.length){add(box,'div','没有匹配记录。可选择「全部批次」，或只输入题号。','placeholder');clear($('detail-panel'));add($('detail-panel'),'div','未找到模型调用记录','empty');}
    data.cases.forEach((c,i)=>{
      const b=add(box,'button',null,'case-card');b.type='button';b.dataset.case=i;
      const top=add(b,'div',null,'row');add(top,'strong',`题目 ${c.question_id}`);const [txt,kind]=status(c);pill(top,txt,kind);
      add(b,'span',c.batch,'batch-name');add(b,'span',c.run_id,'run-name');
      add(b,'div',`${c.calls} 次调用${c.shared_calls?` + ${c.shared_calls} 次招募`:''}${c.anomalies?` · ${c.anomalies} 次异常`:''}`,'mini-meta');
      b.addEventListener('click',()=>selectCase(c,i));
    });
    setNotice(`找到 ${data.total} 次实验${data.limited?'，先展示前 400 条，请输入题号缩小范围':''}。${data.warnings.length?' 部分日志行尚未写完或无法读取，已跳过。':''}`);
    if(chooseFirst&&data.cases.length)await selectCase(data.cases[0],0);
  }catch(e){setNotice(e.message,true)}finally{if(seq===state.searchSeq)$('search-form').querySelector('button').disabled=false}
}

async function selectCase(meta,index){
  const seq=++state.caseSeq;state.callSeq++;
  document.querySelectorAll('.case-card').forEach(b=>b.classList.toggle('active',Number(b.dataset.case)===index));
  loading($('detail-panel'),'读取本题的模型对话…');
  try{
    const data=await api('/api/case',{batch:meta.batch,run_id:meta.run_id});if(seq!==state.caseSeq)return;
    state.current=data;state.calls=data.calls;state.call=null;state.graph=null;renderCase(data);
    const url=new URL(location.href);url.searchParams.set('q',$('query').value||meta.question_id);url.searchParams.set('batch',$('batch').value);url.searchParams.set('run',meta.run_id);url.searchParams.set('source',meta.batch);history.replaceState(null,'',url);
    if(data.calls.length)await selectCall(data.calls[0]);
  }catch(e){if(seq===state.caseSeq){clear($('detail-panel'));add($('detail-panel'),'div',e.message,'error-box')}}
}

function renderCase(data){
  const panel=clear($('detail-panel')),m=data.meta;
  const header=add(panel,'div',null,'detail-header');add(header,'p',m.dataset||'模型调用记录','eyebrow');
  const heading=add(header,'div',null,'detail-heading');const left=add(heading,'div');add(left,'h2',`题目 ${m.question_id}`);
  const actions=add(heading,'div',null,'header-actions');const copyId=add(actions,'button','复制 run_id','quiet');copyId.onclick=()=>copy(m.run_id);
  const download=add(actions,'button','导出本题对话','quiet');download.onclick=exportCase;
  add(header,'div',m.run_id,'run-full');
  const stats=add(header,'div',null,'stats');
  [[data.totals.calls,'模型调用'],[data.totals.tokens,'累计 token'],[m.repeat_id||1,'重复编号']].forEach(([v,label])=>{const item=add(stats,'span');add(item,'b',fmt(Number(v)));item.appendChild(document.createTextNode(label))});
  const [st,kind]=status(m);pill(stats,st,kind);
  const qbox=add(panel,'div',null,'question-box'),qd=add(qbox,'details');qd.open=true;add(qd,'summary','题目与选项');
  if(data.question.stem){add(qd,'div',data.question.stem,'stem');const opts=add(qd,'div',null,'options');Object.entries(data.question.options||{}).forEach(([k,v])=>{const o=add(opts,'div',null,'option');add(o,'b',k);o.appendChild(document.createTextNode(v))});
    if(data.question.context){const d=add(qd,'details');add(d,'summary','题目自带背景');add(d,'div',data.question.context,'question-context')}
  }else{add(qd,'p','此脚本把题干和选项直接写在用户消息中，完整内容见下方「发送给模型」。','hint')}
  if(m.answer!==null||m.gold!==null){const comp=add(qbox,'div',null,'answer-compare');if(m.answer!==null)add(comp,'span',`最终答案：${String(m.answer).length<35?m.answer:'见原始返回'}`);if(m.gold!==null)add(comp,'span',`标准答案：${m.gold}`)}
  add(panel,'div',data.attempt_note+(data.totals.shared_calls?' 招募记录属于本题同一次repeat的共享准备调用。':''),'attempt-note');
  const tabs=add(panel,'div',null,'case-tabs');tabs.setAttribute('role','tablist');tabs.setAttribute('aria-label','查看方式');
  [['dialogue','模型对话'],['graph','证据图谱']].forEach(([v,t])=>{const b=add(tabs,'button',t);b.id='tab-'+v;b.setAttribute('role','tab');b.setAttribute('aria-controls','view-'+v);b.onclick=()=>setCaseView(v)});
  const graph=add(panel,'div');graph.id='view-graph';graph.setAttribute('role','tabpanel');graph.setAttribute('aria-labelledby','tab-graph');
  const dialogue=add(panel,'div');dialogue.id='view-dialogue';dialogue.setAttribute('role','tabpanel');dialogue.setAttribute('aria-labelledby','tab-dialogue');
  const area=add(dialogue,'div',null,'conversation-area');const timeline=add(area,'nav',null,'timeline');timeline.setAttribute('aria-label','模型调用时间线');
  const filterBox=add(timeline,'div',null,'timeline-filter'),filter=add(filterBox,'select');filter.id='task-filter';filter.setAttribute('aria-label','筛选调用类型');
  [['all','全部调用'],['answer','专家作答 / 讨论'],['host','主持人'],['repair','格式修复'],['anomaly','异常 / 截断'],['other','其他辅助调用']].forEach(([v,t])=>{const o=add(filter,'option',t);o.value=v});
  const list=add(timeline,'div');list.id='call-list';filter.onchange=renderTimeline;
  const body=add(area,'div',null,'call-body');body.id='call-body';renderTimeline();
  setCaseView(state.view);
}

function setCaseView(view){
  state.view=view;
  ['dialogue','graph'].forEach(v=>{$('view-'+v).hidden=v!==view;$('tab-'+v).setAttribute('aria-selected',String(v===view))});
  const url=new URL(location.href);url.searchParams.set('view',view);history.replaceState(null,'',url);
  if(view==='graph'&&!state.graph){
    const current=state.current;
    state.graph=new EvidenceGraph($('view-graph'),{batch:current.meta.batch,run_id:current.meta.run_id},line=>{
      if(state.current!==current)return;
      const entry=state.calls.find(c=>c.line===line);if(!entry)return;
      setCaseView('dialogue');$('task-filter').value='all';renderTimeline();selectCall(entry);$('call-body').scrollIntoView({block:'start',behavior:'smooth'});
    });
    state.graph.load();
  }
}

function renderTimeline(){
  const list=clear($('call-list')),filter=$('task-filter').value;
  state.calls.forEach((e,i)=>{
    if(filter==='answer'&&!['answer','review'].includes(e.task))return;
    if(filter==='host'&&e.task!=='host')return;
    if(filter==='repair'&&e.task!=='repair')return;
    if(filter==='anomaly'&&e.status==='ok'&&e.finish_reason!=='length')return;
    if(filter==='other'&&['answer','host','repair','review'].includes(e.task))return;
    const b=add(list,'button',null,'call-item');b.type='button';b.dataset.line=e.line;
    b.classList.toggle('active',state.call?.line===e.line);add(b,'span',String(i+1).padStart(2,'0'),'call-no');
    const info=add(b,'span',null,'call-info');add(info,'strong',phase(e));add(info,'small',role(e.role)||taskNames[e.task]||e.task);
    if(e.finish_reason==='length')add(info,'small','输出达到上限 · 截断','warn');else if(e.status!=='ok')add(info,'small','调用异常','warn');
    else add(info,'small',`${fmt(e.usage.total_tokens)} token${typeof e.elapsed_seconds==='number'?' · '+e.elapsed_seconds.toFixed(1)+'秒':''}`);
    b.onclick=()=>selectCall(e);
  });
  if(!list.childNodes.length)add(list,'div','没有此类调用','empty');
}

async function selectCall(entry){
  state.call=entry;const seq=++state.callSeq;const current=state.current;
  document.querySelectorAll('.call-item').forEach(b=>b.classList.toggle('active',Number(b.dataset.line)===entry.line));
  loading($('call-body'),'读取实际请求与原始返回…');
  try{const row=await api('/api/call',{batch:current.meta.batch,line:entry.line});if(seq!==state.callSeq)return;renderCall(entry,row)}
  catch(e){if(seq===state.callSeq){clear($('call-body'));add($('call-body'),'div',e.message,'error-box')}}
}

function bubble(parent,title,cls){const b=add(parent,'div',null,'bubble '+cls);add(b,'div',title,'bubble-header');return b;}
function field(parent,label,text){add(parent,'span',label,'field-label');add(parent,'p',text);}
function renderInput(parent,value){
  const obj=parse(value);
  if(!obj||typeof obj!=='object'||Array.isArray(obj)){add(parent,'div',typeof obj==='string'?obj:jsonText(obj));return;}
  if(!obj.task&&!obj.question){add(parent,'pre',jsonText(obj));return;}
  if(obj.role)field(parent,'本次角色',role(obj.role));
  if(obj.task)field(parent,'本次任务',taskNames[obj.task]||obj.task);
  if(obj.question){const q=obj.question;const d=add(parent,'details');add(d,'summary','题目、选项与背景（实际传入内容）');add(d,'pre',jsonText(q));}
  if(obj.evidence){const d=add(parent,'details');add(d,'summary',`文献片段 · ${obj.evidence.length} 条${!obj.evidence.length?'（本次无文献输入）':''}`);obj.evidence.forEach((e,i)=>{const n=add(d,'div',null,'nested');add(n,'b',`${i+1}. ${e.title||e.chunk_id||'证据'}`);add(n,'div',e.text||'');details(n,'来源及完整字段',e)});}
  if(obj.messages){const d=add(parent,'details');d.open=!!obj.messages.length;add(d,'summary',`本次可见的专家消息 · ${obj.messages.length} 条`);obj.messages.forEach((m,i)=>{const n=add(d,'div',null,'nested');add(n,'span',`${i+1}. ${m.kind==='claim'?'具体观点':'专家意见'}${m.role?' · '+role(m.role):''}${m.answer?' · 选 '+m.answer:''}`,'field-label');if(m.claim||m.opinion)add(n,'div',m.claim||m.opinion);if(m.conditions)field(n,'适用条件',m.conditions);if(m.polarity)field(n,'立场',({support:'支持',contradict:'反对',limit:'限制',uncertain:'不确定'})[m.polarity]||m.polarity);details(n,'消息完整字段',m)})}
  Object.entries(obj).filter(([k])=>!['role','task','question','evidence','messages','round_id'].includes(k)).forEach(([k,v])=>details(parent,({original_response:'待修复的原始回答',validation_error:'需要修复的格式问题',original_request:'原始请求',candidate:'候选文献',existing:'已有信息',pairs:'待比较的观点',latest_view:'专家最终观点',host:'主持人此前回答'})[k]||k,v));
}

function renderResponse(parent,content){
  const obj=parse(content);
  if(!obj||typeof obj!=='object'||Array.isArray(obj)){add(parent,'div',typeof obj==='string'?obj:jsonText(obj));return;}
  if(obj.answer!==undefined){const a=add(parent,'div',null,'answer-big');add(a,'b',obj.answer);add(a,'span','本次模型选择')}
  if(obj.probabilities){const probs=add(parent,'div',null,'probabilities');Object.entries(obj.probabilities).forEach(([k,v])=>add(probs,'span',`${k}：${Number.isFinite(Number(v))?(Number(v)*100).toFixed(1)+'%':v}`))}
  if(obj.opinion)field(parent,'模型总结意见',obj.opinion);
  if(obj.claims?.length){add(parent,'span',`具体依据 · ${obj.claims.length} 条`,'field-label');obj.claims.forEach(c=>{const x=add(parent,'div',null,'claim');add(x,'div',c.text);add(x,'div',`选项 ${c.answer||'—'} · ${({support:'支持',contradict:'反对',limit:'限制',uncertain:'不确定'})[c.polarity]||c.polarity||'未标注'}${c.conditions?' · 条件：'+c.conditions:''}`,'claim-tags');if(c.citation_ids?.length)add(x,'div','引用：'+c.citation_ids.join('、'),'claim-tags')})}
  Object.entries(obj).filter(([k])=>!['answer','probabilities','opinion','claims'].includes(k)).forEach(([k,v])=>{if(k==='roles'){field(parent,'招募的四位专家',v.map(role).join('、'))}else details(parent,({citations:'引用列表',unexplained_contradiction:'是否有未解释矛盾',missing_distinction:'是否缺少关键鉴别信息',queries:'生成的查询词',relation:'语义关系',relations:'语义关系判断列表',relevant:'是否相关',novel:'是否有新信息',new_claim_or_condition:'是否有新观点或条件'})[k]||k,v)});
}

function renderCall(entry,row){
  const body=clear($('call-body'));
  const heading=add(body,'div',null,'call-heading'),titles=add(heading,'div');add(titles,'h3',phase(entry)+(entry.role?' · '+role(entry.role):''));
  add(titles,'p',`第 ${state.calls.indexOf(entry)+1} 次调用 · ${fmt(entry.usage.prompt_tokens)} 输入 / ${fmt(entry.usage.completion_tokens)} 输出 token${entry.started_at?' · '+new Date(entry.started_at*1000).toLocaleString('zh-CN'):''}`);
  const cp=add(heading,'button','复制本次对话','quiet');cp.onclick=()=>copy(JSON.stringify({request:row.wire_request||row.request,response:row.response||{error:row.error}},null,2));
  if(entry.shared)add(body,'p','这是同题同次repeat的专家招募调用，可能被多个实验组共享，不是额外一次作答。','attempt-note');
  if(entry.finish_reason==='length')add(body,'div','模型输出达到上限，原文可能被截断。下方完整保留此次返回，不能把它当成完整有效答案。','error-box');
  if(row.error)add(body,'div','调用记录中的错误：'+row.error,'error-box');
  const wire=row.wire_request||{},messages=wire.messages||row.request?.messages;
  if(Array.isArray(messages)&&messages.length){
    messages.forEach(m=>{const label=m.role==='system'?'系统提示词':m.role==='user'?'发送给模型':`输入消息 · ${m.role}`;const b=bubble(body,label,m.role==='system'?'system':'request');
      if(m.role==='system'){const d=add(b,'details');add(d,'summary','展开完整系统提示词');add(d,'div',m.content,'bubble-content')}
      else{const c=add(b,'div',null,'bubble-content');renderInput(c,m.content)}
    });
  }else{const b=bubble(body,'程序提交的请求（此日志未保存实际API消息）','request');const c=add(b,'div',null,'bubble-content');renderInput(c,row.request||{})}
  const choices=row.response?.choices||[];
  if(choices.length){choices.forEach((choice,i)=>{const b=bubble(body,choices.length>1?`模型原始返回 · 候选 ${i+1}`:'模型原始返回','response');const c=add(b,'div',null,'bubble-content');
    if(choice.message?.reasoning_content)details(c,'模型返回的推理文本',choice.message.reasoning_content);
    if(choice.message?.content!==undefined&&choice.message.content!==null)renderResponse(c,choice.message.content);else add(c,'p','没有返回正文。');
    if(choice.message?.tool_calls)details(c,'工具调用',choice.message.tool_calls)
  })}else if(row.response?.content){const b=bubble(body,'模型返回（测试或其他格式）','response');renderResponse(add(b,'div',null,'bubble-content'),row.response.content)}
  else{add(body,'div','该调用没有可显示的模型返回；如发生连接错误，服务端结果和用量可能未知。','error-box')}
  const raw=add(body,'div',null,'raw-section');details(raw,'查看实际请求 JSON（包含参数和完整消息）',row.wire_request||row.request||{});details(raw,'查看服务端完整返回 JSON',row.response||{error:row.error});details(raw,'查看完整调用日志记录',row);
  if(entry.attempt_id)add(body,'p','执行编号：'+entry.attempt_id,'source-line');add(body,'p',state.current.source+` · 第 ${entry.line} 行`,'source-line');
}

async function exportCase(){
  const c=state.current;if(!c)return;try{toast('正在整理本题完整调用…');const rows=[];for(const e of c.calls)rows.push(await api('/api/call',{batch:c.meta.batch,line:e.line}));
    const blob=new Blob([JSON.stringify({...c,full_calls:rows},null,2)],{type:'application/json;charset=utf-8'});const url=URL.createObjectURL(blob);const a=node('a');a.href=url;a.download=`question-${c.meta.question_id}-${c.meta.batch}.json`;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('本题对话已导出')
  }catch(e){toast(e.message)}
}

async function init(){
  try{const data=await api('/api/batches');state.batches=data.batches;data.batches.forEach(b=>{const o=add($('batch'),'option',b.name);o.value=b.id});
    const params=new URLSearchParams(location.search);$('query').value=params.get('q')||'';
    if(params.get('batch')&&[...$('batch').options].some(o=>o.value===params.get('batch')))$('batch').value=params.get('batch');
    else if(!params.get('q')&&data.batches.length)$('batch').value=data.batches[0].id;
    $('search-form').onsubmit=e=>{e.preventDefault();search()};$('batch').onchange=()=>search();
    $('refresh').onclick=async()=>{const remembered=state.current?.meta;await search(false);if(remembered){const i=state.cases.findIndex(c=>c.batch===remembered.batch&&c.run_id===remembered.run_id);if(i>=0)await selectCase(state.cases[i],i)}};
    document.querySelectorAll('[data-example]').forEach(b=>b.onclick=()=>{$('query').value=b.dataset.example;$('batch').value='all';search()});
    await search();
    if(params.get('run')){const i=state.cases.findIndex(c=>c.run_id===params.get('run')&&(!params.get('source')||c.batch===params.get('source')));if(i>=0)await selectCase(state.cases[i],i)}
  }catch(e){setNotice('无法加载日志：'+e.message,true)}
}
init();
