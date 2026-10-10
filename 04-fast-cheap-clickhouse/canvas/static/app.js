import {createSurface} from './surface.js?v=20261007-worlds1';
import {createMap,createGallery} from './media.js?v=20261007-worlds1';
import {createMarks,plotPoints} from './marks.js?v=20261007-worlds1';
import {initMotion,animateCard,animatePending,setWorking,motionEnabled,exportOrbit} from './motion.js?v=20261007-worlds1';
import {createCamera} from './camera.js?v=20261007-worlds1';
import {renderExplore} from './explore-viz.js?v=20261007-worlds1';

const $=s=>document.querySelector(s),clone=v=>structuredClone(v),uuid=()=>crypto.randomUUID().replaceAll('-','');
const el=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!=null)n.textContent=text;return n;};
const button=(text,title,action,cls='')=>{const b=el('button',cls,text);b.type='button';b.title=title;b.setAttribute('aria-label',title);b.onclick=action;return b;};
const colors={teal:'#49e2c0',blue:'#65adff',coral:'#ff997f',violet:'#ba9dff',gold:'#f2d583'};
const labels={line:'Line',bar:'Bars',heatmap:'Heatmap',surface:'3D surface',table:'Table',scatter:'Scatter',map:'Map',gallery:'Photos',tracks:'Voyages',density:'Density map',seasons:'Seasonal wheel'};
const icons={line:'↗',bar:'▥',heatmap:'▦',surface:'◇',table:'▤',scatter:'⁙',map:'◎',gallery:'▧',tracks:'⌁',density:'▦',seasons:'◉'};
let state={cards:[],messages:[],undo:[],redo:[],version:0},selected=null,token=null,busy=false,addIntent=false,zoom=1;
let marked=null,markingCard=null,pendingCard=null,layoutKey='',pointerDown=false,layoutChanged=false;
let dirty=0,saved=0,saveJob=null,saveTimer=null,abort=null,toastTimer=null,conflict=false,switching=false;
let requestFinished=Promise.resolve(),localFinished=Promise.resolve(),sessionListRequest=0;
let datasets=[],linkedTime=null,timeFrame=0,lastPlotCursor=0;
const playbackCache=new Map();
const nodes=new Map(),viewport=$('#viewport'),board=$('#board');
window.canvasTelemetry=[];
const telemetry=(type,details={})=>window.canvasTelemetry.push({type,at:performance.now(),...details});
const camera=createCamera({viewport,board,spacer:$('#spacer'),label:$('#zoom-level'),getCards:()=>pendingCard?[...state.cards,pendingCard]:state.cards,motion:motionEnabled,onZoom:z=>zoom=z,onEvent:(type,details)=>{telemetry(type,details);if(type==='camera-settled')nodes.forEach(n=>n.renderer?.resize?.());}});
const frame=()=>new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
const fmt=n=>typeof n==='number'?new Intl.NumberFormat('en',{maximumFractionDigits:2}).format(n):String(n??'—');
function toast(text){$('#toast').textContent=text;$('#toast').hidden=false;clearTimeout(toastTimer);toastTimer=setTimeout(()=>$('#toast').hidden=true,6000);}
function status(text){$('#save-status').textContent=text;}
function cardBy(id){return state.cards.find(c=>c.id===id);}
const datasetId=()=>state.dataset_id||'wildlife';
function datasetInfo(){return datasets.find(d=>(d.id||d.dataset_id)===datasetId());}
function syncDataset(){
  const id=datasetId();document.body.dataset.dataset=id;$('#dataset-select').value=id;
  const info=datasetInfo(),loaded=info?.metadata?.coverage?.completed_dates||[];
  $('#dataset-caption').textContent=id==='maritime'?(info?.available===false?`Gulf & Florida · ${loaded.length} days loaded · importing`:'Gulf & Florida · 15 Sep–15 Oct 2024'):'Worldwide · observations & photos';
  $('#photos-mark').hidden=id!=='wildlife';
  $('#prompt').placeholder=id==='maritime'?'Explore ships, storms, and the days between…':'Explore species, places, and seasons…';
  document.title='Data Canvas · '+(id==='maritime'?'Ships & Storms':'Wildlife');
}
async function refreshDatasets(){
  const data=await(await api('datasets')).json();datasets=data.datasets||[];
  if(datasets.length)$('#dataset-select').replaceChildren(...datasets.map(d=>{const option=el('option','',d.label||d.name||d.id);option.value=d.id||d.dataset_id;return option;}));
  syncDataset();
}
async function playback(c){
  const session=state.session_id,key=[session,c.sql,JSON.stringify(c.data)].join('|');
  if(playbackCache.has(key))return playbackCache.get(key);
  const started=performance.now(),promise=(async()=>{
    await flush();
    for(let attempt=0;attempt<2;attempt++){
      if(state.session_id!==session||cardBy(c.id)?.revision!==c.revision)throw Error('This view changed while its tracks were loading.');
      const version=state.version;
      try{return await(await api('playback',{method:'POST',body:JSON.stringify({card_id:c.id,revision:c.revision,session_id:session,version})})).json();}
      catch(error){if(error.status!==409||attempt||state.version===version)throw error;}
    }
  })();
  playbackCache.set(key,promise);
  try{const data=await promise;telemetry('playback-ready',{card_id:c.id,elapsed_ms:performance.now()-started,rows:data.rows?.length||0,sampling:data.sampling});return data;}catch(e){playbackCache.delete(key);throw e;}
}
function syncTime(source,time){
  if(!Number.isFinite(time))return;
  if(linkedTime?.source!==source)nodes.forEach((n,id)=>{if(id!==source&&cardBy(id)?.chart==='tracks')n.renderer?.setPlaying?.(false);});
  linkedTime={source,time};
  if(timeFrame)return;
  timeFrame=requestAnimationFrame(()=>{timeFrame=0;const cursor=linkedTime;if(!cursor)return;
    const updatePlots=performance.now()-lastPlotCursor>120;if(updatePlots)lastPlotCursor=performance.now();
    nodes.forEach((n,id)=>{if(id!==cursor.source)n.renderer?.setTime?.(cursor.time,{silent:true});
      if(updatePlots&&n.plot&&n.plot._fullLayout?.xaxis?.type==='date'){
        const date=new Date(cursor.time).toISOString();
        Plotly.relayout(n.plot,{shapes:[{type:'line',xref:'x',yref:'paper',x0:date,x1:date,y0:0,y1:1,line:{color:'#f2d583',width:1.5,dash:'dot'}}]}).catch(()=>{});
      }
    });
  });
}
function markData(id,value){if(busy||switching)return;choose(id);const c=cardBy(id);marked=value?{...value,card_id:id,revision:c.revision}:null;syncMarks();telemetry('mark',{card_id:id,selection:marked});}
function syncMarks(){
  if(marked&&(!cardBy(marked.card_id)||cardBy(marked.card_id).revision!==marked.revision))marked=null;
  const active=marked&&marked.card_id===selected;$('#selection-chip').hidden=!active;
  if(active)$('#selection-label').textContent=marked.kind==='region'?'Map area marked':`${marked.row_indices.length} point${marked.row_indices.length===1?'':'s'} marked`;
  nodes.forEach((n,id)=>{const own=marked?.card_id===id?marked:null;n.marks?.set(own);n.marks?.enable(markingCard===id&&!busy&&!switching);n.markButton?.setAttribute('aria-pressed',String(markingCard===id));n.root.classList.toggle('has-marks',!!own);n.body.querySelectorAll('[data-row-index]').forEach(row=>row.classList.toggle('marked-row',!!own?.row_indices?.includes(Number(row.dataset.rowIndex))));});
}
function toggleMark(id){if(busy||switching)return;choose(id);markingCard=markingCard===id?null:id;syncMarks();if(markingCard)toast('Click a point or drag across the data. Then ask a follow-up.');}
function togglePoint(id,index){const prior=marked?.card_id===id&&marked.kind==='points'?marked.row_indices:[];const rows=prior.includes(index)?prior.filter(x=>x!==index):[...prior,index];markData(id,rows.length?{kind:'points',row_indices:rows}:null);}
function snap(c){const s=clone(c);delete s.history;return s;}

async function refresh(){
  const r=await fetch('/api/auth/refresh',{method:'POST',credentials:'include'});
  if(!r.ok)return false;const data=await r.json();token=data.token;return !!token;
}
async function api(path,options={},retry=true){
  const r=await fetch('/canvas/api/'+path,{...options,headers:{'Content-Type':'application/json','Authorization':'Bearer '+token,...options.headers}});
  if(r.status===401){if(retry&&await refresh())return api(path,options,false);$('#login-screen').hidden=false;throw Error('Please sign in again.');}
  if(!r.ok){let detail;try{detail=(await r.json()).detail;}catch{detail='Request failed ('+r.status+')';}if(r.status===409&&path!=='playback'){conflict=true;status('Reload needed');}const error=Error(typeof detail==='string'?detail:JSON.stringify(detail));error.status=r.status;throw error;}
  return r;
}
function trimHistory(){
  while(new TextEncoder().encode(JSON.stringify(state)).length>7_500_000){
    if(state.undo.length){state.undo.shift();continue;}
    if(state.redo.length){state.redo.shift();continue;}
    const c=state.cards.find(c=>c.history?.length);if(c){c.history.shift();continue;}break;
  }
}
function markDirty(){dirty++;status('Saving…');clearTimeout(saveTimer);saveTimer=setTimeout(()=>flush().catch(e=>toast(e.message)),350);}
async function flush(){
  clearTimeout(saveTimer);
  if(saveJob){await saveJob;if(dirty>saved)return flush();return;}
  if(conflict)throw Error('This workspace changed in another tab. Reload to continue.');
  if(dirty===saved)return;
  saveJob=(async()=>{while(saved<dirty){trimHistory();const serial=dirty;const r=await api('workspace',{method:'PUT',body:JSON.stringify(state)});const result=await r.json();state.version=result.version;saved=serial;}status('Saved');})();
  try{await saveJob;}catch(e){status('Not saved');throw e;}finally{saveJob=null;}
}
function history(label){state.undo=(state.undo||[]).concat([{label,cards:clone(state.cards)}]).slice(-25);state.redo=[];}
function edit(label,change){if(busy||switching||conflict)return;history(label);change();markDirty();render();telemetry('local',{action:label});}
function revise(c,label,update){c.history=(c.history||[]).concat([snap(c)]).slice(-12);Object.assign(c,update,{revision:uuid(),updated_at:new Date().toISOString(),change:label,reused_data:true});if(marked?.card_id===c.id)marked.revision=c.revision;}
function choose(id){const next=cardBy(id)?id:null;if(next!==selected){marked=null;markingCard=null;}selected=next;nodes.forEach((n,k)=>n.root.classList.toggle('selected',k===selected));const c=cardBy(selected);$('#target-text').textContent=c?(addIntent?'New view from · ':'Context · ')+c.title:'Canvas · new view';$('#clear-target').hidden=!c;syncMarks();}
function setIntent(value){addIntent=value;$('#intent-toggle').setAttribute('aria-pressed',String(value));choose(selected);}
function focusCard(id){choose(id);const c=cardBy(id);if(c)camera.focus(c);}
function dimensions(){
  camera.layout();
  const svg=$('#connections');svg.replaceChildren();for(const c of state.cards){const src=cardBy(c.derived_from);if(!src)continue;const p=document.createElementNS('http://www.w3.org/2000/svg','path'),x1=src.left+src.width,y1=src.top+src.height/2,x2=c.left,y2=c.top+c.height/2;const bend=Math.max(45,Math.abs(x2-x1)*.45);p.setAttribute('d',`M ${x1} ${y1} C ${x1+bend} ${y1}, ${x2-bend} ${y2}, ${x2} ${y2}`);p.setAttribute('stroke','#3ab9b1');p.setAttribute('stroke-width','1.5');p.setAttribute('stroke-dasharray','4 5');p.setAttribute('fill','none');svg.append(p);}
}
function setZoom(z){camera.zoomTo(z);}
function fit(options){camera.fit(options);}
function choices(c){
  const cols=c.data.columns,result=['table'],hasXY=cols.includes(c.x)&&cols.includes(c.y),hasZ=cols.includes(c.z);
  const geo=['map','density','tracks','gallery'].includes(c.chart)||(/lon/i.test(c.x)&&/lat/i.test(c.y));
  if(hasXY)result.unshift(...(geo?['map']:['line','bar','scatter']));
  if(hasZ&&hasXY){if(geo)result.unshift('density');if(c.chart!=='tracks')result.unshift('heatmap','surface');}
  if(datasetId()==='wildlife'&&hasXY&&/month/i.test(c.x))result.unshift('seasons');
  if(cols.includes('photo_url'))result.unshift('gallery');
  if(datasetId()==='maritime'&&(cols.includes('vessel_id')||cols.includes('storm_id'))&&hasXY)result.unshift('tracks');
  return [...new Set([c.chart,...result])];
}
function chartUpdate(c,chart){
  const {columns,rows}=c.data,yi=columns.indexOf(c.y),zi=columns.indexOf(c.z);
  const categorical=index=>index>=0&&rows.some(row=>typeof row[index]==='string'&&!Number.isFinite(Number(row[index])));
  // Grids put the category on y and the measure on z; comparison lines do the reverse.
  if(zi>=0&&((['line','bar','scatter'].includes(chart)&&categorical(yi)&&!categorical(zi))||(['seasons','heatmap','surface'].includes(chart)&&!categorical(yi)&&categorical(zi))))return {chart,y:c.z,z:c.y};
  return {chart};
}
function seriesCoordinates(c,rows,xi,yi){
  if(c.chart!=='line'||!rows.length||!rows.every(row=>typeof row[xi]==='string'&&/^\d{4}-\d{2}-\d{2}(?:[ T]|$)/.test(row[xi])))return {x:rows.map(row=>row[xi]),y:rows.map(row=>row[yi])};
  const stamp=value=>{const text=value.replace(' ','T');return Date.parse(text.length===10?text+'T00:00:00Z':/(?:Z|[+-]\d{2}:?\d{2})$/i.test(text)?text:text+'Z');};
  const points=rows.map(row=>({row,time:stamp(row[xi])})).sort((a,b)=>a.time-b.time);
  const cadence=/(^|_)hour(_|$)/i.test(c.x)?3600000:/(^|_)(day|date)(_|$)/i.test(c.x)?86400000:/(^|_)week(_|$)/i.test(c.x)?604800000:0;
  const x=[],y=[];let previous;
  for(const point of points){
    if(cadence&&previous&&point.time-previous.time>cadence*1.5){x.push(new Date(previous.time+cadence).toISOString());y.push(null);}
    x.push(point.row[xi]);y.push(point.row[yi]);previous=point;
  }
  return {x,y};
}
function destroyRenderer(n){n.generation=(n.generation||0)+1;n.marks?.destroy();n.marks=null;if(n.renderer?.view)n.mapView=n.renderer.view();n.renderer?.destroy?.();n.renderer=null;if(n.plot&&window.Plotly){Plotly.purge(n.plot);n.plot=null;}n.body.replaceChildren();}
async function draw(n,c){
  const key=[c.revision,c.chart,c.color].join('|');if(n.key===key)return n.ready;n.key=key;
  const start=performance.now();const priorPlot=n.plot&&['line','bar','scatter','heatmap'].includes(c.chart);
  n.marks?.destroy();n.marks=null;if(!priorPlot)destroyRenderer(n);
  const generation=n.generation=(n.generation||0)+1;
  const {columns,rows}=c.data;if(!rows.length){destroyRenderer(n);n.body.append(el('div','empty-data','No rows match this selection. Try a wider date range or area.'));return;}
  if(['tracks','density','seasons'].includes(c.chart)||(c.chart==='map'&&datasetId()==='maritime')){
    const renderer=await renderExplore(n.body,{...c,dataset_id:datasetId()},{previousView:n.mapView,fetchPlayback:()=>playback(c),onSelectRows:indices=>markData(c.id,indices.length?{kind:'points',row_indices:indices}:null),onRegion:bounds=>markData(c.id,{kind:'region',bounds}),onTimeChange:time=>syncTime(c.id,time)});
    if(generation!==n.generation||!n.root.isConnected){renderer?.destroy?.();return;}
    n.renderer=renderer;
    await n.renderer?.ready;
    if(linkedTime)n.renderer?.setTime?.(linkedTime.time,{silent:true});
  }
  else if(c.chart==='map'){n.renderer=await createMap(n.body,c,(id,prompt)=>{if(busy)return;choose(id);setIntent(true);send(prompt);},n.mapView);}
  else if(c.chart==='gallery'){n.renderer=createGallery(n.body,c,index=>togglePoint(c.id,index));n.renderer.ready.then(()=>telemetry('images-ready',{card_id:c.id,elapsed_ms:performance.now()-start,...n.renderer?.stats?.()}));}
  else if(c.chart==='surface'){n.renderer=createSurface(n.body,c);await n.renderer.ready;}
  else if(c.chart==='table'){
    const wrap=el('div','table-view'),table=el('table'),thead=el('thead'),tr=el('tr'),tbody=el('tbody');
    let order=1;function fill(data){tbody.replaceChildren();data.forEach(row=>{const tr=el('tr');tr.dataset.rowIndex=rows.indexOf(row);tr.tabIndex=0;tr.title='Click to mark this row';tr.onclick=()=>togglePoint(c.id,rows.indexOf(row));tr.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();togglePoint(c.id,rows.indexOf(row));}};row.forEach(value=>tr.append(el('td','',fmt(value))));tbody.append(tr);});}
    columns.forEach((name,i)=>{const th=el('th','',name.replaceAll('_',' '));th.tabIndex=0;const sort=()=>{order*=-1;fill([...rows].sort((a,b)=>order*(typeof a[i]==='number'&&typeof b[i]==='number'?a[i]-b[i]:String(a[i]??'').localeCompare(String(b[i]??'')))));telemetry('local',{action:'sort-table'});};th.onclick=sort;th.onkeydown=e=>{if(e.key==='Enter')sort();};tr.append(th);});thead.append(tr);table.append(thead,tbody);wrap.append(table);n.body.append(wrap);fill(rows);
  }else{
    if(!window.Plotly)throw Error('The chart library has not loaded. Reload the page.');
    if(!n.plot){n.plot=el('div','plot');n.body.append(n.plot);}
    const xi=columns.indexOf(c.x),yi=columns.indexOf(c.y),zi=columns.indexOf(c.z),x=rows.map(r=>r[xi]),y=rows.map(r=>r[yi]),color=colors[c.color];let trace,margin={l:58,r:18,t:18,b:42};
    const numeric=a=>a.every(v=>v==null||Number.isFinite(Number(v)));
    if(c.chart==='heatmap'){
      const xs=[...new Set(x)].sort((a,b)=>numeric([a,b])?a-b:String(a).localeCompare(String(b))),ys=[...new Set(y)].sort((a,b)=>numeric([a,b])?a-b:String(a).localeCompare(String(b)));const lookup=new Map(rows.map(r=>[JSON.stringify([r[xi],r[yi]]),r[zi]]));
      trace={type:'heatmap',x:xs,y:ys,z:ys.map(b=>xs.map(a=>lookup.get(JSON.stringify([a,b]))??null)),colorscale:[[0,'#132f40'],[.45,'#226577'],[1,color]],colorbar:{thickness:7,len:.75,tickfont:{size:8},outlinewidth:0},hoverongaps:false};margin.r=42;
    }else if(c.chart==='bar'){trace={type:'bar',orientation:'h',y:x,x:y,marker:{color,opacity:.88},hovertemplate:'%{y}<br>%{x:,.2f}<extra></extra>'};margin.l=Math.min(145,Math.max(60,...x.map(v=>String(v).length*5)));}
    else trace={type:'scatter',mode:c.chart==='scatter'?'markers':'lines+markers',...seriesCoordinates(c,rows,xi,yi),line:{color,width:2.5,shape:'linear'},marker:{color,size:c.chart==='scatter'?6:4},fill:c.chart==='line'?'tozeroy':undefined,fillcolor:color+'1a',connectgaps:false};
    const axis={tickfont:{size:9,color:'#8a9cae'},gridcolor:'#233745',zeroline:false,linecolor:'#2b4555',automargin:true};
    const layout={autosize:true,margin,paper_bgcolor:'transparent',plot_bgcolor:'transparent',font:{family:'Inter,system-ui,sans-serif',color:'#758ca4',size:10},xaxis:{...axis,title:{text:(c.chart==='bar'?c.y:c.x).replaceAll('_',' '),font:{size:9}},type:c.chart==='bar'?undefined:(numeric(x)?'linear':undefined)},yaxis:{...axis,title:{text:c.chart==='bar'?'':c.y.replaceAll('_',' '),font:{size:9}},autorange:c.chart==='bar'?'reversed':true},showlegend:false,transition:{duration:150},uirevision:c.revision};
    let traces=[trace];
    if(['line','scatter'].includes(c.chart)&&zi>=0&&rows.some(r=>typeof r[zi]==='string')){
      const groups=[...new Set(rows.map(r=>String(r[zi]??'Unknown')))];
      if(groups.length<=12){const palette=[color,...Object.values(colors).filter(v=>v!==color)];traces=groups.map((group,i)=>{const subset=rows.filter(r=>String(r[zi]??'Unknown')===group);return {...trace,name:group,...seriesCoordinates(c,subset,xi,yi),line:{...trace.line,color:palette[i%palette.length]},marker:{...trace.marker,color:palette[i%palette.length]},fill:undefined,hovertemplate:'%{x}<br>%{y:,.2f}<extra>'+group.replaceAll('<','&lt;').replaceAll('>','&gt;')+'</extra>'};});layout.showlegend=true;layout.legend={orientation:'h',x:0,y:1.15,font:{size:9,color:'#98b6c4'},bgcolor:'transparent'};layout.margin.t=40;}
    }
    await Plotly.react(n.plot,traces,layout,{responsive:true,displayModeBar:false,scrollZoom:false});
  }
  if(generation!==n.generation||!n.root.isConnected)return;
  if(['map','density','tracks','seasons','surface','line','bar','scatter','heatmap'].includes(c.chart)){
    const adapter=n.plot?{points:()=>plotPoints(n.plot,c)}:n.renderer;
    n.marks=createMarks(n.body,adapter,mark=>markData(c.id,mark));
    if(n.plot){n.plot.removeAllListeners?.('plotly_relayout');n.plot.on('plotly_relayout',()=>n.body.dispatchEvent(new Event('projectionchange')));}
  }
  syncMarks();telemetry('render',{card_id:c.id,chart:c.chart,elapsed_ms:performance.now()-start});
}

function makeCard(c){
  const root=el('article','data-card');root.dataset.id=c.id;root.tabIndex=0;root.setAttribute('aria-label',c.title);root.addEventListener('pointerdown',()=>choose(c.id));root.addEventListener('focus',()=>choose(c.id));
  const heading=el('div','card-heading'),icon=el('span','card-icon'),titleWrap=el('div','card-title-wrap'),title=el('h2','card-title'),sub=el('div','card-subtitle'),tools=el('div','card-tools');titleWrap.append(title,sub);
  tools.append(button('⧉','Duplicate view',()=>duplicate(c.id)),button('↶','View history',()=>details(c.id,'history')),button('×','Remove view',()=>edit('Remove view',()=>{state.cards=state.cards.filter(x=>x.id!==c.id);if(selected===c.id)selected=null;})));heading.append(icon,titleWrap,tools);
  const settings=el('div','card-settings'),select=el('select','view-select');select.setAttribute('aria-label','Visualization type');select.onchange=()=>edit('Change visualization',()=>revise(cardBy(c.id),'Changed visualization',chartUpdate(cardBy(c.id),select.value)));
  const swatches=el('div','swatches');Object.keys(colors).forEach(name=>{const swatch=button('',`Use ${name}`,()=>edit('Change color',()=>revise(cardBy(c.id),'Changed color',{color:name})),'swatch');swatch.style.setProperty('--swatch',colors[name]);swatch.dataset.color=name;swatches.append(swatch);});const interaction=el('div','card-interaction');const markButton=button('⌖ Mark','Mark data to ask a follow-up',()=>toggleMark(c.id),'mark-button');markButton.setAttribute('aria-pressed','false');
  const orbitButton=button('▶ Orbit','Play or pause 3D orbit',()=>{const r=nodes.get(c.id)?.renderer;if(!motionEnabled()){toast('Choose Flowing in canvas settings to play the orbit.');return;}r?.setPlaying?.(!r.isPlaying());},'orbit-button');
  const gifButton=button('GIF ↗','Export a rotating 3D GIF',async()=>{const r=nodes.get(c.id)?.renderer;if(!r?.exportFrames)return;gifButton.disabled=true;try{await exportOrbit(r,cardBy(c.id).title,toast);telemetry('gif-export',{card_id:c.id,frames:36});}catch(e){toast(e.message);}finally{gifButton.disabled=false;}},'gif-button');
  interaction.append(markButton,orbitButton,gifButton);settings.append(select,interaction,swatches);
  const body=el('div','card-body'),footer=el('div','card-footer'),footnote=el('span');footer.append(footnote,button('View SQL ↗','View SQL',()=>details(c.id,'sql')));root.append(heading,settings,body,footer);board.append(root);
  const n={root,icon,title,sub,select,swatches,body,footnote,tools,markButton,orbitButton,gifButton,key:null};body.addEventListener('orbitstate',e=>{orbitButton.textContent=e.detail?'Ⅱ Pause':'▶ Orbit';orbitButton.setAttribute('aria-pressed',String(e.detail));});
  const ro=new ResizeObserver(()=>{if(n.plot)Plotly.Plots.resize(n.plot);n.renderer?.resize?.();const current=cardBy(c.id);if(!current||busy||switching)return;const w=Math.round(root.offsetWidth),h=Math.round(root.offsetHeight);if(Math.abs(current.width-w)>1||Math.abs(current.height-h)>1){current.width=Math.max(300,Math.min(1600,w));current.height=Math.max(250,Math.min(1200,h));dimensions();markDirty();layoutChanged=true;if(!pointerDown){layoutChanged=false;fit({reason:'card-resize'});}}});ro.observe(root);n.resize=ro;
  heading.addEventListener('pointerdown',e=>{
    if(busy||switching||e.target.closest('button')||e.button!==0)return;e.preventDefault();heading.setPointerCapture(e.pointerId);const current=cardBy(c.id),x=e.clientX,y=e.clientY,left=current.left,top=current.top;const before=clone(state.cards);let moved=false;
    const move=e=>{moved=true;current.left=Math.max(0,Math.min(6000,left+(e.clientX-x)/zoom));current.top=Math.max(110,Math.min(6000,top+(e.clientY-y)/zoom));root.style.left=current.left+'px';root.style.top=current.top+'px';dimensions();};
    const end=()=>{heading.removeEventListener('pointermove',move);heading.removeEventListener('pointerup',end);heading.removeEventListener('pointercancel',end);if(moved){state.undo=state.undo.concat([{label:'Move view',cards:before}]).slice(-25);state.redo=[];markDirty();controls();fit({reason:'arranged'});}};heading.addEventListener('pointermove',move);heading.addEventListener('pointerup',end);heading.addEventListener('pointercancel',end);
  });return n;
}
function controls(){
  const locked=busy||switching;
  $('#send').disabled=locked||!token;
  document.querySelectorAll('[data-prompt],.view-select,.swatch,.card-tools button,.mark-button,#intent-toggle').forEach(b=>b.disabled=locked);
  $('#new-session').disabled=$('#session-select').disabled=$('#dataset-select').disabled=switching||!token;$('#prompt').disabled=switching;
  choose(selected);
}
async function render({reveal=[]}={}){
  for(const [id,n]of nodes){if(!cardBy(id)){n.resize.disconnect();destroyRenderer(n);n.root.remove();nodes.delete(id);}}
  const arrivals=[];
  const jobs=state.cards.map(async c=>{
    let n=nodes.get(c.id);if(!n){n=makeCard(c);nodes.set(c.id,n);arrivals.push(c.id);n.root.classList.add('materializing');}if(reveal.includes(c.id))n.root.classList.add('materializing');n.root.dataset.chart=c.chart;n.root.style.cssText=`left:${c.left}px;top:${c.top}px;width:${c.width}px;height:${c.height}px;--accent:${colors[c.color]}`;n.title.textContent=c.title;n.root.setAttribute('aria-label',c.title);n.icon.textContent=icons[c.chart];n.sub.textContent=`${c.data.rows.length.toLocaleString()} rows · ${c.history?.length?c.history.length+1:1} version${c.history?.length?'s':''}`;
    n.select.replaceChildren(...choices(c).map(value=>{const o=el('option','',labels[value]);o.value=value;return o;}));n.select.value=c.chart;n.orbitButton.hidden=n.gifButton.hidden=c.chart!=='surface';n.markButton.hidden=['table','gallery'].includes(c.chart);n.swatches.querySelectorAll('button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.color===c.color)));
    n.footnote.textContent=['map','density','seasons'].includes(c.chart)&&datasetId()==='wildlife'?'Reported observations · not population':c.chart==='tracks'?'Recorded positions · UTC playback':c.chart==='gallery'?'Photographs from iNaturalist':c.reused_data?'Existing result · no new query':`ClickHouse · ${Math.round(c.query_ms)} ms`;
    try{await draw(n,c);}catch(e){destroyRenderer(n);n.body.append(el('div','empty-data',e.message));telemetry('render-error',{card_id:c.id,error:e.message});}
  });dimensions();controls();await Promise.all(jobs);await frame();
  // Start framing when the new content can actually appear, not while a chart
  // is still being built invisibly on the main thread.
  const nextKey=JSON.stringify(state.cards.map(c=>[c.id,c.left,c.top,c.width,c.height]));if(nextKey!==layoutKey){layoutKey=nextKey;fit({reason:'views-changed'});}
  for(const id of arrivals){if(reveal.includes(id))continue;const n=nodes.get(id);if(!n)continue;n.root.classList.remove('materializing');const animation=animateCard(n.root,true);animation?.finished.then(()=>telemetry('view-revealed',{card_id:id})).catch(()=>{});}
}
function duplicate(id,version){const old=version||cardBy(id);if(!old)return;edit('Duplicate view',()=>{if(state.cards.length>=12){toast('Keep at most 12 views on this canvas.');return;}const c=clone(old);c.id=uuid();c.derived_from=id;c.left=Math.min(6000,c.left+60);c.top=Math.min(6000,c.top+65);c.history=[];c.revision=uuid();c.title+=' · copy';state.cards.push(c);selected=c.id;});}
function undo(redo=false){if(busy||switching)return;const from=redo?'redo':'undo',to=redo?'undo':'redo';const entry=state[from].pop();if(!entry)return;state[to]=(state[to]||[]).concat([{label:entry.label,cards:clone(state.cards)}]).slice(-25);state.cards=clone(entry.cards);if(!cardBy(selected))selected=state.cards.at(-1)?.id||null;markDirty();render();telemetry('local',{action:redo?'redo':'undo'});toast((redo?'Redid: ':'Undid: ')+entry.label);}
function details(id,type){
  const c=cardBy(id);if(!c)return;$('#detail-panel').hidden=false;$('#detail-title').textContent=type==='sql'?'Query & provenance':'View history';const b=$('#detail-body');b.replaceChildren(el('p','detail-card-title',c.title));
  if(type==='sql'){b.append(el('p','detail-label','Executed SQL'),el('pre','sql-code',c.sql),el('p','detail-note',`Returned ${c.data.rows.length} rows. Last data query: ${Math.round(c.query_ms)} ms${c.reused_data?' · reused existing results':''}.`));b.append(el('p','detail-note',datasetId()==='wildlife'?'iNaturalist snapshot: 27 September 2026. Observations reflect reporting effort, not population size. Public coordinates may be obscured.':'NOAA AIS records transmitted positions in US waters; reception is incomplete. Speeds and timestamps describe observations, not inferred intent. IBTrACS supplies recorded storm tracks. Playback sampling is shown inside each voyage view.'));}
  else{const versions=[snap(c),...(c.history||[]).slice().reverse()];versions.forEach((v,i)=>{const row=el('div','revision-row'+(!i?' current':''));row.append(el('p','',i===0?'Current view':v.change||v.title),el('small','',`${labels[v.chart]} · ${v.data.rows.length} rows · ${new Date(v.updated_at||v.created_at).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}`));if(i){const actions=el('div','revision-actions');actions.append(button('Restore','Restore this version',()=>{edit('Restore version',()=>{const current=cardBy(id);const keep={id:current.id,left:current.left,top:current.top,width:current.width,height:current.height,history:(current.history||[]).concat([snap(current)]).slice(-12)};Object.assign(current,clone(v),keep,{revision:uuid(),change:'Restored earlier version'});});details(id,'history');}),button('Add as copy','Add this version alongside',()=>duplicate(id,v)));row.append(actions);}b.append(row);});b.append(el('p','detail-note','Restoring reuses the saved result. It does not call the model or rerun the query.'));}
}
function renderMessages(){
  const list=$('#messages'),previous=list.querySelectorAll('.message').length;list.replaceChildren();
  state.messages.forEach((m,i)=>{const row=appendMessage(m);if(i<previous)row.style.animation='none';});list.scrollTop=list.scrollHeight;
}
function appendMessage(m){
  const row=el('div','message '+m.role),body=el('div','message-body');if(m.role!=='user')row.append(el('span','mini-orb','✦'));body.append(el('p','',m.text));
  (m.changes||[]).forEach(change=>{const b=button(`${change.action==='add'?'＋':'↳'} ${change.title}`,'Focus '+change.title,()=>focusCard(change.id),'change-link');b.append(el('small','',`${change.rows} rows${change.reused_data?' · reused data':''}`));body.append(b);});if(m.elapsed_ms)body.append(el('div','timing-note',`${(m.elapsed_ms/1000).toFixed(2)}s · model + query + save`));row.append(body);$('#messages').append(row);$('#messages').scrollTop=$('#messages').scrollHeight;return row;
}
async function send(text){
  const prompt=(text??$('#prompt').value).trim();if(!prompt||busy||switching)return;
  localFinished=localRequest(prompt);if(await localFinished)return;
  try{await flush();}catch(e){toast(e.message);return;}
  if(switching)return;
  let finishRequest;requestFinished=new Promise(resolve=>finishRequest=resolve);
  busy=true;setWorking('thinking');abort=new AbortController();const target=selected,requestId=uuid(),start=performance.now(),intent=addIntent?'add':'auto';$('#prompt').value='';$('#working').hidden=false;$('#working-text').textContent='Shaping your view…';controls();const pending=appendMessage({role:'user',text:prompt});
  let ghost,stopPending=()=>{};
  function pendingView(mode){
    if(mode==='update'){if(nodes.has(target))nodes.get(target).root.classList.add('card-busy');return;}
    if(ghost)return;
    const count=state.cards.length;pendingCard={left:38+(count%2)*580,top:130+Math.floor(count/2)*465,width:548,height:425};
    ghost=el('div','data-card blueprint');ghost.setAttribute('role','status');ghost.setAttribute('aria-label','Shaping a new view');
    ghost.style.cssText=`left:${pendingCard.left}px;top:${pendingCard.top}px;width:${pendingCard.width}px;height:${pendingCard.height}px`;
    const seed=el('div','blueprint-seed');seed.setAttribute('aria-hidden','true');seed.append(el('i','blueprint-ring'),el('i','blueprint-ring'),el('i','blueprint-ring'),el('span','blueprint-core','✦'));
    ghost.append(seed,el('p','','Shaping a new view'),el('small','','Reading your request…'));board.append(ghost);stopPending=animatePending(ghost);controls();dimensions();fit({reason:'new-view-pending'});telemetry('pending-view',{request_id:requestId,motion:motionEnabled()});
  }const requestMark=marked?clone(marked):null;
  let resultReceived=false;
  try{
    const response=await api('paint',{method:'POST',signal:abort.signal,body:JSON.stringify({prompt,selected_id:target,intent,version:state.version,request_id:requestId,selection:requestMark})});
    const reader=response.body.getReader(),decoder=new TextDecoder();let buffer='';
    for(;;){const {value,done}=await reader.read();buffer+=decoder.decode(value||new Uint8Array(),{stream:!done});let idx;
      while((idx=buffer.indexOf('\n\n'))>=0){const raw=buffer.slice(0,idx);buffer=buffer.slice(idx+2);const kind=raw.split('\n').find(x=>x.startsWith('event:'))?.slice(6).trim(),line=raw.split('\n').find(x=>x.startsWith('data:'));if(!line)continue;const data=JSON.parse(line.slice(5));
        if(kind==='status'){if(data.intent)pendingView(data.intent);$('#working-text').textContent=data.text;setWorking(data.phase);if(ghost)ghost.querySelector('small').textContent=data.title||data.text;}
        if(kind==='error')throw Error(data.detail||data.message);
        if(kind==='result'){
          resultReceived=true;pendingCard=null;marked=null;markingCard=null;setWorking('drawing');state=data.workspace;dirty=saved=0;status('Saved');if(selected===target&&data.changes.length)selected=data.changes.at(-1).id;renderMessages();$('#working-text').textContent='Drawing your view…';await render({reveal:data.changes.filter(c=>c.action==='add').map(c=>c.id)});const elapsed=performance.now()-start;
          let fromScale=.78;
          if(ghost){fit({reason:'data-ready'});const retiring=ghost,style=getComputedStyle(retiring);fromScale=Math.min(.78,new DOMMatrix(style.transform).a);const initial={opacity:style.opacity,transform:style.transform};stopPending();ghost=null;if(motionEnabled()){retiring.classList.add('blueprint-resolved');retiring.animate([initial,{opacity:0,transform:'scale(1.035)'}],{duration:1400,easing:'cubic-bezier(.42,0,.28,1)'}).finished.catch(()=>{}).finally(()=>retiring.remove());}else retiring.remove();}
          for(const change of data.changes){const n=nodes.get(change.id);if(n){n.root.classList.remove('materializing');const animation=animateCard(n.root,change.action==='add',{fromScale});animation?.finished.then(()=>telemetry('motion-settled',{request_id:requestId,card_id:change.id,elapsed_ms:performance.now()-start})).catch(()=>{});}}
          telemetry('paint',{request_id:requestId,prompt,elapsed_ms:elapsed,model_ms:data.model_ms,query_ms:data.query_ms,changes:data.changes});
          const last=$('#messages .message:last-child .timing-note');if(last)last.textContent=`${(elapsed/1000).toFixed(2)}s to data · model ${(data.model_ms/1000).toFixed(2)}s · data ${Math.round(data.query_ms)}ms`;
          setIntent(false);
          refreshSessions().catch(e=>toast(e.message));
        }
      }if(done)break;
    }
    if(!resultReceived)throw Error('The connection ended before a result arrived. Reload to check the saved workspace before retrying.');
  }catch(e){pending.remove();if(e.name==='AbortError'){toast('Stopped. Checking the last saved workspace…');try{state=await(await api('workspace')).json();await render();renderMessages();}catch{toast('Reload to check the saved workspace.');}}else{appendMessage({role:'assistant',text:e.message});toast(e.message);}telemetry('request-error',{request_id:requestId,error:e.message});}
  finally{stopPending();const abandoned=!!pendingCard;pendingCard=null;ghost?.remove();if(abandoned){dimensions();fit({reason:'pending-cleared'});}setWorking(null);busy=false;abort=null;$('#working').hidden=true;nodes.forEach(n=>n.root.classList.remove('card-busy','materializing'));controls();if(!switching)$('#prompt').focus();finishRequest();}
}

async function refreshSessions(){
  const serial=++sessionListRequest,data=await(await api('sessions')).json();if(serial!==sessionListRequest)return;
  const select=$('#session-select');select.replaceChildren();
  for(const id of ['wildlife','maritime']){const sessions=data.sessions.filter(s=>(s.dataset_id||'wildlife')===id);if(!sessions.length)continue;const group=el('optgroup');group.label=id==='maritime'?'Ships & Storms':'Wildlife';sessions.forEach(s=>{const option=el('option','',s.title);option.value=s.id;group.append(option);});select.append(group);}
  select.value=state.session_id||'';select.title=select.selectedOptions[0]?.textContent||'Saved chats';
}
async function changeSession(id=null,dataset=datasetId()){
  if(switching||id===state.session_id)return;
  switching=true;controls();const fades=[];
  try{
    if(busy){abort?.abort();await requestFinished;}
    await localFinished;
    await flush();status(id?'Opening chat…':'Starting a new chat…');
    const response=api(id?'sessions/'+encodeURIComponent(id)+'/activate':'sessions',{method:'POST',body:JSON.stringify({version:state.version,...(!id?{dataset_id:dataset}:{})})}).then(r=>r.json());
    if(motionEnabled())for(const node of [viewport,$('#messages')])fades.push(node.animate([{opacity:1},{opacity:0}],{duration:420,easing:'ease-in-out',fill:'forwards'}));
    const [next]=await Promise.all([response,Promise.all(fades.map(a=>a.finished.catch(()=>{})))]);
    camera.stop();state=next;selected=next.cards.at(-1)?.id||null;marked=markingCard=pendingCard=null;layoutKey='';dirty=saved=0;conflict=false;linkedTime=null;playbackCache.clear();cancelAnimationFrame(timeFrame);timeFrame=0;
    $('#prompt').value='';syncDataset();$('#detail-panel').hidden=true;setIntent(false);renderMessages();await render();
    if(motionEnabled())for(const node of [viewport,$('#messages')])node.animate([{opacity:0},{opacity:1}],{duration:1000,easing:'ease-in-out'});
    fades.forEach(a=>a.cancel());status('Saved');await refreshSessions();telemetry('session-opened',{session_id:state.session_id,new_session:!id});
  }catch(e){toast(e.message);$('#session-select').value=state.session_id||'';syncDataset();if(!conflict)status('Saved');}
  finally{fades.forEach(a=>a.cancel());switching=false;controls();$('#prompt').focus();}
}

async function localRequest(prompt){
  const c=cardBy(selected);if(!c||addIntent||conflict)return false;
  const normalized=prompt.toLowerCase().replace(/[.!]+$/,'').trim();let update,label;
  for(const color of Object.keys(colors))if([`make it ${color}`,`make this ${color}`,`use ${color}`,`change the color to ${color}`,`change the colour to ${color}`].includes(normalized)){update={color};label=`Changed the color to ${color}.`;}
  const forms={table:['table','a table'],bar:['bars','a bar chart'],line:['a line chart'],heatmap:['a heatmap'],surface:['3d','a 3d surface'],map:['a map'],gallery:['photos'],density:['a density map'],seasons:['a seasonal wheel'],tracks:['voyages','a voyage map']};
  for(const [chart,words]of Object.entries(forms))for(const word of words)if([`show as ${word}`,`show it as ${word}`,`make it ${word}`,`switch to ${word}`].includes(normalized)&&choices(c).includes(chart)){update=chartUpdate(c,chart);label=`Changed the view to ${labels[chart].toLowerCase()}.`;}
  if(!update)return false;
  const start=performance.now();$('#prompt').value='';history(label);revise(c,label,update);
  state.messages=state.messages.concat([{role:'user',text:prompt,target_id:c.id},{role:'assistant',text:label,changes:[{id:c.id,action:'update',title:c.title,rows:c.data.rows.length,reused_data:true}]}]).slice(-100);
  markDirty();renderMessages();await render();const elapsed=performance.now()-start;const note=el('div','timing-note',`${Math.round(elapsed)}ms · existing data`);$('#messages .message:last-child .message-body')?.append(note);telemetry('local-request',{prompt,elapsed_ms:elapsed,card_id:c.id});return true;
}

initMotion();
$('#new-session').onclick=()=>changeSession();$('#session-select').onchange=e=>changeSession(e.target.value);
$('#dataset-select').onchange=e=>{if(e.target.value!==datasetId())changeSession(null,e.target.value);};
$('#dataset-details').onclick=()=>{
  $('.chat-settings').open=false;$('#detail-panel').hidden=false;const maritime=datasetId()==='maritime',info=datasetInfo();$('#detail-title').textContent=maritime?'Ships & Storms':'Wildlife';const body=$('#detail-body');body.replaceChildren();
  body.append(el('p','detail-card-title',info?.label||info?.name||(maritime?'Ships & Storms':'Wildlife worldwide')));
  body.append(el('p','detail-note',info?.description||(maritime?'Explore recorded vessel movement in the Gulf and around Florida, alongside Hurricane Milton and neighbouring storms.':'Explore worldwide reported species observations and licensed photographs.')));
  body.append(el('p','detail-note',maritime?'Vessel positions have gaps and playback may be sampled. A stationary vessel is not necessarily waiting for a port. Storm intensity symbols are not wind footprints.':'Observation counts reflect reporting effort, not animal abundance. Public coordinates may be obscured. Photos retain their creator and license.'));
  const examples=maritime?['Replay 12 moving cargo ships in the Gulf on October 9, 2024, with average speed above 5 knots and the Milton storm overlay.','Compare daily cargo and tanker activity throughout the dataset.','Show the hourly average vessel speed on October 9 and 10, 2024.']:['Show a worldwide density map of monarch observations in 2025.','Show the monthly seasonal pattern of birds, insects and plants in 2025.','Show photos of red foxes in Europe.'];
  body.append(el('p','detail-label','Try exploring'));examples.forEach(text=>body.append(button(text,text,()=>{$('#detail-panel').hidden=true;$('#prompt').value=text;$('#prompt').focus();},'dataset-example')));
  const sources=maritime?[['NOAA vessel traffic','https://github.com/ocm-marinecadastre/ais-vessel-traffic'],['NOAA hurricane archive','https://www.ncei.noaa.gov/products/international-best-track-archive']]:[['iNaturalist open data','https://github.com/inaturalist/inaturalist-open-data']];
  sources.forEach(([label,url])=>{const a=el('a','dataset-source',label+' ↗');a.href=url;a.target='_blank';a.rel='noopener noreferrer';body.append(a);});
};
$('#clear-mark').onclick=()=>{marked=null;markingCard=null;syncMarks();};
$('#ask-mark').onclick=()=>{$('#prompt').focus();$('#prompt').placeholder='Ask about the marked data…';};
$('#photos-mark').onclick=()=>{setIntent(true);send('Add a photo gallery from the marked data, using up to 36 observations with licensed photos.');};
$('#composer').onsubmit=e=>{e.preventDefault();send();};$('#prompt').onkeydown=e=>{if(e.key==='Enter'&&!e.shiftKey&&!e.isComposing){e.preventDefault();send();}};
$('#cancel').onclick=()=>abort?.abort();$('#clear-target').onclick=()=>choose(null);$('#intent-toggle').onclick=()=>setIntent(!addIntent);
$('#zoom-in').onclick=()=>setZoom(zoom+.1);$('#zoom-out').onclick=()=>setZoom(zoom-.1);$('#zoom-level').onclick=()=>setZoom(1);$('#fit').onclick=()=>fit();
function showChat(open){document.body.classList.toggle('show-chat',open);$('#mobile-chat').setAttribute('aria-expanded',String(open));}
$('#close-detail').onclick=()=>$('#detail-panel').hidden=true;$('#mobile-chat').onclick=()=>showChat(true);$('#close-chat').onclick=()=>showChat(false);
document.querySelectorAll('[data-prompt]').forEach(b=>b.onclick=()=>send(b.dataset.prompt));
document.addEventListener('keydown',e=>{if(e.key==='Escape'){markingCard=null;syncMarks();$('#detail-panel').hidden=true;$('.chat-settings').open=false;showChat(false);}if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='z'&&!e.target.closest('input,textarea,select')){e.preventDefault();undo(e.shiftKey);}});
viewport.addEventListener('pointerdown',()=>pointerDown=true,{capture:true});
const endPointer=()=>{pointerDown=false;if(layoutChanged){layoutChanged=false;fit({reason:'card-resize'});}};window.addEventListener('pointerup',endPointer);window.addEventListener('pointercancel',endPointer);
window.addEventListener('beforeunload',e=>{if(dirty!==saved||busy){e.preventDefault();e.returnValue='';}});
$('#login-form').onsubmit=async e=>{e.preventDefault();const b=e.target.querySelector('button[type=submit]');b.disabled=true;$('#login-error').textContent='';try{const r=await fetch('/api/auth/login',{method:'POST',credentials:'include',headers:{'Content-Type':'application/json'},body:JSON.stringify({email:$('#email').value,password:$('#password').value})});if(!r.ok)throw Error('Check your email and password, then try again.');const d=await r.json();token=d.token;$('#password').value='';await openWorkspace();}catch(e){$('#login-error').textContent=e.message;}finally{b.disabled=false;}};
async function openWorkspace(){state=await(await api('workspace')).json();await refreshDatasets();$('#login-screen').hidden=true;selected=state.cards.at(-1)?.id||null;conflict=false;dirty=saved=0;renderMessages();await render();fit({reason:'workspace-open'});status('Saved');await refreshSessions();}
try{if(await refresh())await openWorkspace();else{$('#login-screen').hidden=false;status('Sign in');}}catch(e){$('#login-screen').hidden=false;$('#login-error').textContent=e.message;status('Sign in');}
