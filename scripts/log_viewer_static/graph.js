/* Saved provenance only. No network dependencies, HTML from logs, or model calls. */
(() => {
  const svgNS='http://www.w3.org/2000/svg';
  const kindNames={question:'题目',evidence:'文献片段',turn:'专家发言',claim:'具体观点',cluster:'展示的观点',view:'展示的意见',answer:'最终答案',unresolved:'未解析来源'};
  const colors={question:'#506879',evidence:'#3c8978',turn:'#547bba',claim:'#aa7e39',cluster:'#9270ae',view:'#9270ae',answer:'#226f59',unresolved:'#8c9399'};
  const polarities={support:'支持',contradict:'反对',limit:'限制',uncertain:'不确定'};
  function el(tag,text,cls){const n=document.createElement(tag);if(text!==undefined&&text!==null)n.textContent=String(text);if(cls)n.className=cls;return n}
  function append(p,tag,text,cls){const n=el(tag,text,cls);p.append(n);return n}
  function svg(tag,attrs={},text){const n=document.createElementNS(svgNS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!==undefined)n.textContent=String(text);return n}
  function btn(p,text,fn,cls='quiet'){const b=append(p,'button',text,cls);b.type='button';b.onclick=fn;return b}
  function select(p,label,values,fn){const wrap=append(p,'label',null,'graph-control');append(wrap,'span',label);const s=append(wrap,'select');for(const [v,t] of values){const o=append(s,'option',t);o.value=v}s.onchange=()=>fn(s.value);return s}
  function short(s,n=72){s=String(s||'').replace(/\s+/g,' ').trim();return s.length>n?s.slice(0,n-1)+'…':s}
  function roleText(r){return typeof role==='function'?role(r):r}
  function phaseText(n){return n.role==='host'?'主持人汇总':n.round_id===0?'独立作答 R0':typeof n.round_id==='number'?`第 ${n.round_id} 轮`:'已记录发言'}
  function lines(text,width,max){const chunks=[];let line='',size=0;for(const c of String(text||'')){const w=c.charCodeAt(0)>255?1:.54;if(size+w>width){chunks.push(line);line='';size=0}line+=c;size+=w;if(chunks.length===max)break}if(chunks.length<max&&line)chunks.push(line);if(chunks.join('').length<String(text||'').length&&chunks.length)chunks[chunks.length-1]=chunks[chunks.length-1].slice(0,-1)+'…';return chunks}

  class EvidenceGraph {
    constructor(container,params,onCall){this.container=container;this.params=params;this.onCall=onCall;this.seq=0;this.mode='overview';this.depth=2;this.exposed=false;this.zoom=.85;this.data=null;this.selected=null;this.focus=null}
    async load(attempt=''){
      const seq=++this.seq;this.container.replaceChildren(el('div','正在读取证据与引用关系…','loading'));
      try{const r=await fetch('/api/graph?'+new URLSearchParams({...this.params,attempt}));const data=await r.json();if(!r.ok)throw Error(data.error||'读取图谱失败');if(seq!==this.seq||!this.container.isConnected)return;
        this.data=data;this.nodes=new Map(data.nodes.map(n=>[n.id,n]));this.incoming=new Map();this.outgoing=new Map();
        for(const e of data.edges){if(!this.incoming.has(e.target))this.incoming.set(e.target,[]);this.incoming.get(e.target).push(e);if(!this.outgoing.has(e.source))this.outgoing.set(e.source,[]);this.outgoing.get(e.source).push(e)}
        this.selected=data.final_turn||data.nodes.find(n=>n.kind==='turn')?.id||'question';this.focus=this.selected;this.build();
      }catch(e){if(seq!==this.seq||!this.container.isConnected)return;this.container.replaceChildren(el('div',e.message,'error-box'));btn(this.container,'重试读取',()=>this.load(attempt))}
    }
    build(){
      const box=this.container;box.replaceChildren();const intro=append(box,'div',null,'graph-intro');const title=append(intro,'div');append(title,'h3','从材料到最终判断');append(title,'p',`${this.data.counts.evidence||0} 条文献 · ${this.data.counts.turn||0} 次发言 · ${this.data.counts.claim||0} 条观点`);
      const topActions=append(intro,'div',null,'header-actions');btn(topActions,'追溯最终答案',()=>{this.mode='focus';this.focus=this.data.final_turn||this.selected;this.modeSelect.value='focus';this.redraw();this.choose(this.focus)},'primary');btn(topActions,'导出图谱',()=>this.download());
      const note=append(box,'p',this.data.note,'graph-note');note.setAttribute('role','note');
      if(!this.data.counts.evidence)append(box,'p','当前记录没有文献输入；仍可查看专家之间的观点引用。','graph-empty-note');
      for(const warning of this.data.warnings)append(box,'p',warning,'graph-warning');
      const controls=append(box,'div',null,'graph-controls');
      select(controls,'执行记录',this.data.attempts.map(a=>[a.id,`${a.label} · ${a.turns} 次发言`]),v=>this.load(v)).value=this.data.attempt;
      this.modeSelect=select(controls,'图谱视角',[['overview','过程总览'],['focus','引用追溯']],v=>{this.mode=v;this.focus=this.selected;this.redraw()});this.modeSelect.value=this.mode;
      this.depthSelect=select(controls,'向前追溯',[['2','2 层来源'],['4','4 层来源'],['all','全部来源']],v=>{this.depth=v==='all'?Infinity:Number(v);this.redraw()});this.depthSelect.value=this.depth===Infinity?'all':String(this.depth);
      const check=append(controls,'label',null,'graph-check'),input=append(check,'input');input.type='checkbox';input.checked=this.exposed;input.onchange=()=>{this.exposed=input.checked;this.redraw()};check.append(document.createTextNode('同时显示“当时可见”'));
      const zoom=append(controls,'div',null,'graph-zoom');btn(zoom,'−',()=>this.resize(this.zoom/1.2)).setAttribute('aria-label','缩小图谱');this.zoomLabel=append(zoom,'span');btn(zoom,'+',()=>this.resize(this.zoom*1.2)).setAttribute('aria-label','放大图谱');btn(zoom,'适应窗口',()=>this.resize(Math.min(1,(this.viewport.clientWidth-30)/this.width)));
      const legend=append(box,'div',null,'graph-legend');for(const kind of ['question','evidence','turn','claim','cluster','answer']){const item=append(legend,'span',kindNames[kind]);item.dataset.kind=kind}append(legend,'small','实线：明确引用 / 归属　虚线：输入 / 可见性');
      this.caption=append(box,'p',null,'graph-caption');
      this.viewport=append(box,'div',null,'graph-viewport');this.viewport.tabIndex=0;this.viewport.setAttribute('aria-label','证据关系图，可拖动空白区域或使用滚动条查看');
      this.canvas=svg('svg',{'aria-label':'证据与发言关系图',role:'group'});this.viewport.append(this.canvas);
      let drag=null;
      this.viewport.addEventListener('pointerdown',e=>{if(e.button!==0||e.target.closest('[data-node]'))return;drag={x:e.clientX,y:e.clientY,left:this.viewport.scrollLeft,top:this.viewport.scrollTop};this.viewport.setPointerCapture(e.pointerId);this.viewport.classList.add('dragging')});
      this.viewport.addEventListener('pointermove',e=>{if(!drag)return;this.viewport.scrollLeft=drag.left+drag.x-e.clientX;this.viewport.scrollTop=drag.top+drag.y-e.clientY});
      for(const event of ['pointerup','pointercancel'])this.viewport.addEventListener(event,()=>{drag=null;this.viewport.classList.remove('dragging')});
      append(box,'p','点击节点阅读原文和来源；拖动空白处移动，或用上方按钮缩放。总览把同一来源到同一次发言的多条关系合并展示。','graph-help');
      this.detail=append(box,'div',null,'graph-detail');this.detail.setAttribute('aria-live','polite');this.redraw();this.choose(this.selected);
    }
    enabled(e){return this.exposed||e.relation!=='exposed'}
    owners(id,trail=new Set()){
      if(trail.has(id))return[];trail=new Set(trail);trail.add(id);const n=this.nodes.get(id);if(!n)return[];
      if(['question','evidence','turn','answer','unresolved'].includes(n.kind))return[id];
      if(n.owner_id)return[n.owner_id];
      return [...new Set((this.incoming.get(id)||[]).filter(e=>['member_of','displayed_as'].includes(e.relation)).flatMap(e=>this.owners(e.source,trail)))];
    }
    graphView(){
      if(this.mode==='overview'){
        const ns=this.data.nodes.filter(n=>['question','evidence','turn','answer','unresolved'].includes(n.kind)),map=new Map();
        for(const e of this.data.edges){if(!this.enabled(e)||['part_of','member_of','displayed_as'].includes(e.relation))continue;
          for(const source of this.owners(e.source))for(const target of this.owners(e.target)){if(source===target)continue;const key=source+'|'+target+'|'+e.relation;
            if(!map.has(key))map.set(key,{...e,source,target,count:0});map.get(key).count++}
        }
        return {nodes:ns,edges:[...map.values()]};
      }
      const visited=new Set([this.focus||this.selected]),queue=[{id:this.focus||this.selected,depth:0}];
      for(let i=0;i<queue.length;i++){const {id,depth}=queue[i];if(depth>=this.depth)continue;for(const e of this.incoming.get(id)||[]){if(!this.enabled(e)||visited.has(e.source))continue;visited.add(e.source);queue.push({id:e.source,depth:depth+1})}}
      // A selected claim keeps its speaker and that speaker's recorded final result in view.
      const focused=this.nodes.get(this.focus);const speaker=focused?.owner_id||this.focus;
      if(focused?.owner_id)visited.add(focused.owner_id);
      for(const e of this.incoming.get(speaker)||[])if(e.relation==='question_input')visited.add(e.source);
      for(const e of this.outgoing.get(speaker)||[])if(e.relation==='final_answer')visited.add(e.target);
      return {nodes:this.data.nodes.filter(n=>visited.has(n.id)),edges:this.data.edges.filter(e=>this.enabled(e)&&visited.has(e.source)&&visited.has(e.target))};
    }
    column(n){
      if(n.kind==='question')return 0;if(['evidence','unresolved'].includes(n.kind))return 1;if(n.kind==='answer')return this.mode==='overview'?8:21;
      const owner=n.owner_id?this.nodes.get(n.owner_id):null;let round=owner?.round_id??n.round_id;
      if(typeof round!=='number'){
        const authors=(this.incoming.get(n.id)||[]).filter(e=>['member_of','displayed_as'].includes(e.relation)).flatMap(e=>this.owners(e.source)).map(id=>this.nodes.get(id)?.round_id).filter(x=>typeof x==='number');
        round=authors.length?Math.max(...authors):0;
      }
      if(this.mode==='overview')return 2+Math.min(5,Math.max(0,round));
      return 2+Math.max(0,round)*3+(n.kind==='turn'?1:['cluster','view'].includes(n.kind)?2:0);
    }
    redraw(){
      if(!this.data)return;this.depthSelect.disabled=this.mode!=='focus';const graph=this.graphView();this.visible=graph;
      const grouped=new Map();for(const n of graph.nodes){const col=this.column(n);if(!grouped.has(col))grouped.set(col,[]);grouped.get(col).push(n)}
      const columns=[...grouped.keys()].sort((a,b)=>a-b),pos=new Map(),width=236,height=104,gapX=64,gapY=25;
      columns.forEach((col,index)=>{const ns=grouped.get(col).sort((a,b)=>(a.order||0)-(b.order||0)||a.seq-b.seq);ns.forEach((n,i)=>pos.set(n.id,{x:30+index*(width+gapX),y:85+i*(height+gapY)}))});
      this.positions=pos;this.width=Math.max(600,columns.length*(width+gapX));this.height=Math.max(230,Math.max(0,...[...grouped.values()].map(a=>a.length))*(height+gapY)+100);this.viewport.classList.toggle('compact',this.height<=300);this.canvas.replaceChildren();this.canvas.setAttribute('viewBox',`0 0 ${this.width} ${this.height}`);
      const defs=svg('defs');for(const [name,color] of [['citation','#789b91'],['visible','#aebbbc']]){const marker=svg('marker',{id:'graph-arrow-'+name,viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto-start-reverse'});marker.append(svg('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:color}));defs.append(marker)}this.canvas.append(defs);
      columns.forEach((col,index)=>{const sample=grouped.get(col)[0];let label=sample.kind==='question'?'题目输入':sample.kind==='evidence'?'文献材料':sample.kind==='answer'?'结果记录':this.mode==='overview'?phaseText(sample):`${sample.role?roleText(sample.role)+' · ':''}${kindNames[sample.kind]}`;this.canvas.append(svg('text',{x:30+index*(width+gapX),y:28,class:'graph-column'},short(label,30)))});
      this.paths=[];
      for(const e of graph.edges){
        const a=pos.get(e.source),b=pos.get(e.target);if(!a||!b)continue;
        const forward=b.x>a.x,x1=forward?a.x+width:a.x,x2=forward?b.x:b.x+width,y1=a.y+height/2,y2=b.y+height/2,delta=Math.max(36,Math.abs(x2-x1)*.4);
        // Long edges travel above the cards instead of appearing to pass through intervening nodes.
        let path=`M ${x1} ${y1} C ${x1+(forward?delta:-delta)} ${y1}, ${x2-(forward?delta:-delta)} ${y2}, ${x2} ${y2}`;
        if(forward&&b.x-a.x>width+gapX+1){const lane=48+(this.paths.length%5)*5;path=`M ${x1} ${y1} C ${x1+18} ${y1}, ${x1+18} ${lane}, ${x1+32} ${lane} L ${x2-32} ${lane} C ${x2-18} ${lane}, ${x2-18} ${y2}, ${x2} ${y2}`}
        const p=svg('path',{d:path,class:'graph-edge '+(['exposed','question_input'].includes(e.relation)?'is-visible':''),'marker-end':'url(#graph-arrow-'+(['exposed','question_input'].includes(e.relation)?'visible':'citation')+')'});
        p.append(svg('title',{},`${e.label}${e.count>1?' · '+e.count+' 条记录':''}`));this.canvas.append(p);this.paths.push({el:p,edge:e});
      }
      this.nodeEls=[];for(const n of graph.nodes){const p=pos.get(n.id);const g=svg('g',{transform:`translate(${p.x},${p.y})`,'data-node':n.id,class:'graph-node',tabindex:0,role:'button','aria-label':`${kindNames[n.kind]}：${n.role?roleText(n.role):n.label}${n.answer?'，答案 '+n.answer:''}`});g.append(svg('rect',{width,height,rx:10,fill:'#fff',stroke:colors[n.kind]||colors.unresolved,'stroke-width':1.5}));g.append(svg('rect',{x:0,y:16,width:4,height:height-32,rx:2,fill:colors[n.kind]||colors.unresolved}));
        const badge=n.kind==='turn'?phaseText(n):kindNames[n.kind];g.append(svg('text',{x:14,y:23,class:'graph-node-kind'},badge));
        if(n.answer!==undefined&&n.answer!==null)g.append(svg('text',{x:width-14,y:24,'text-anchor':'end',class:'graph-node-answer'},'选 '+n.answer));
        const title=n.kind==='turn'?roleText(n.role):n.kind==='evidence'?n.label:n.text||n.label;
        lines(title,17,2).forEach((text,i)=>g.append(svg('text',{x:14,y:47+i*17,class:'graph-node-title'},text)));
        const sub=n.kind==='turn'?short(n.text,38):n.kind==='claim'?`${polarities[n.data.polarity]||'未标注立场'}${n.role?' · '+roleText(n.role):''}`:n.kind==='evidence'?(n.data.document_id||'文献原文见下方'):n.kind==='cluster'?'消息版本 · '+(n.data.member_claim_ids?.length||0)+' 条原始观点':n.kind==='answer'?'来自该次执行结果':'点击查看内容与来源';g.append(svg('text',{x:14,y:89,class:'graph-node-sub'},short(sub,39)));
        g.append(svg('title',{},n.text||n.label));g.addEventListener('click',()=>this.choose(n.id));g.addEventListener('keydown',e=>{if(['Enter',' '].includes(e.key)){e.preventDefault();this.choose(n.id)}});this.canvas.append(g);this.nodeEls.push({el:g,id:n.id})}
      this.caption.textContent=this.mode==='overview'?`过程总览 · ${graph.nodes.length} 个节点 / ${graph.edges.length} 组关系。每个专家卡片可点开查看全部具体观点。`:`引用追溯 · 展示 ${graph.nodes.length} / ${this.data.nodes.length} 个节点；${this.depth===Infinity?'已展开全部上游来源':'当前向前 '+this.depth+' 层，可增加追溯层数'}。`;
      this.resize(this.zoom);this.highlight();this.viewport.scrollTop=0;this.viewport.scrollLeft=0;
    }
    resize(value){this.zoom=Math.max(.2,Math.min(2,value));this.canvas.setAttribute('width',Math.ceil(this.width*this.zoom));this.canvas.setAttribute('height',Math.ceil(this.height*this.zoom));this.zoomLabel.textContent=Math.round(this.zoom*100)+'%'}
    highlight(){const selected=this.selected;const related=new Set([selected]);for(const e of this.visible.edges)if(e.source===selected||e.target===selected){related.add(e.source);related.add(e.target)}for(const n of this.nodeEls){n.el.classList.toggle('selected',n.id===selected);n.el.classList.toggle('related',related.has(n.id))}for(const p of this.paths)p.el.classList.toggle('highlight',p.edge.source===selected||p.edge.target===selected)}
    link(parent,id,label){const n=this.nodes.get(id);return btn(parent,label||`${kindNames[n?.kind]||'来源'} · ${short(n?.role?roleText(n.role):n?.text||n?.label,70)}${n?.answer?' · 选 '+n.answer:''}`,()=>this.choose(id),'graph-link')}
    choose(id){const n=this.nodes.get(id);if(!n)return;this.selected=id;this.highlight();const d=this.detail;d.replaceChildren();
      const header=append(d,'div',null,'graph-detail-header');const titles=append(header,'div');append(titles,'p',kindNames[n.kind],'eyebrow');append(titles,'h3',n.kind==='turn'?`${roleText(n.role)} · ${phaseText(n)}`:n.label||kindNames[n.kind]);const actions=append(header,'div',null,'header-actions');btn(actions,'以此为中心追溯',()=>{this.mode='focus';this.focus=id;this.modeSelect.value='focus';this.redraw();this.choose(id);this.viewport.scrollIntoView({block:'start',behavior:'smooth'})});if(n.call_line)btn(actions,'查看对应模型对话',()=>this.onCall(n.call_line));
      if(n.answer!==undefined&&n.answer!==null){const answer=append(d,'div',null,'graph-answer');append(answer,'strong','选择 '+n.answer);if(n.data.probabilities)append(answer,'span',Object.entries(n.data.probabilities).map(([k,v])=>`${k} ${Number.isFinite(Number(v))?(Number(v)*100).toFixed(1)+'%':v}`).join('　'))}
      if(n.text)append(d,'p',n.text,'graph-full-text');
      if(n.data.conditions)append(d,'p','适用条件：'+n.data.conditions,'graph-condition');
      if(n.data.polarity)append(d,'p','观点立场：'+(polarities[n.data.polarity]||n.data.polarity),'graph-condition');
      if(n.owner_id){const author=append(d,'div',null,'graph-author');append(author,'span','这条观点来自：');this.link(author,n.owner_id)}
      if(n.kind==='question'){for(const [key,value] of Object.entries(n.data.options||{}))append(d,'p',`${key}. ${value}`);if(n.data.context)append(d,'p',n.data.context,'graph-full-text')}
      if(n.kind==='evidence'&&n.data.source){try{const url=new URL(n.data.source);if(['https:','http:'].includes(url.protocol)){const a=append(d,'a','打开文献来源 ↗','graph-source');a.href=url.href;a.target='_blank';a.rel='noopener noreferrer'}}catch{}}
      const claims=(this.incoming.get(id)||[]).filter(e=>e.relation==='part_of');if(claims.length){const list=append(d,'div',null,'graph-claims');append(list,'h4',`本次发言的全部观点 · ${claims.length} 条`);claims.forEach((e,i)=>{const c=this.nodes.get(e.source);const item=append(list,'button',null,'graph-claim-row');append(item,'span',String(i+1).padStart(2,'0'));const text=append(item,'div');append(text,'p',c.text);append(text,'small',`${polarities[c.data.polarity]||'未标注'} · 针对选项 ${c.answer||'—'}${c.data.conditions?' · '+c.data.conditions:''}`);item.onclick=()=>this.choose(c.id)})}
      const relations=append(d,'div',null,'graph-relations');for(const [label,list] of [['来自哪些记录',this.incoming.get(id)||[]],['传给哪些后续记录',this.outgoing.get(id)||[]]]){const group=append(relations,'section');append(group,'h4',label);const filtered=list.filter(e=>e.relation!=='part_of');if(!filtered.length)append(group,'p','没有记录此类关系。','muted');const visible=filtered.slice(0,12),rest=filtered.slice(12);const draw=(parent,items)=>items.forEach(e=>{const row=append(parent,'div',null,'graph-relation');append(row,'span',e.label,'graph-relation-label');this.link(row,e.source===id?e.target:e.source)});draw(group,visible);if(rest.length){const more=append(group,'details');append(more,'summary',`展开其余 ${rest.length} 条关系`);draw(more,rest)}}
      const raw=append(d,'details',null,'graph-raw');append(raw,'summary','查看原始编号与日志字段');append(raw,'pre',JSON.stringify({id:n.raw_id,event_line:n.event_line,call_line:n.call_line,...n.data},null,2));
    }
    download(){const blob=new Blob([JSON.stringify(this.data,null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=el('a');a.href=url;a.download=this.params.run_id.replace(/[^\w.-]/g,'_')+'_graph.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
  }
  window.EvidenceGraph=EvidenceGraph;
})();
