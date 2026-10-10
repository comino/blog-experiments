// Marking uses saved result row indices. SQL scope is derived independently on the server.
export function createMarks(host, adapter, onChange) {
  const layer=document.createElement('div');layer.className='mark-layer';
  const svg=document.createElementNS('http://www.w3.org/2000/svg','svg');svg.classList.add('mark-overlay');
  const box=document.createElement('div');box.className='mark-box';box.hidden=true;
  const tip=document.createElement('div');tip.className='mark-tip';tip.textContent='Click a point or drag an area · Shift to add';
  layer.append(svg,box,tip);host.append(layer);
  let active=false, mark=null, drag=null;
  function paint(){
    svg.replaceChildren();
    if(mark?.kind==='region'&&adapter.regionPixels){const b=adapter.regionPixels(mark.bounds);const r=document.createElementNS(svg.namespaceURI,'rect');for(const [k,v]of Object.entries({x:b.x,y:b.y,width:b.width,height:b.height}))r.setAttribute(k,v);svg.append(r);}
    else if(mark?.row_indices){const indices=new Set(mark.row_indices);for(const p of adapter.points()){if(!indices.has(p.index))continue;const r=document.createElementNS(svg.namespaceURI,'circle');r.setAttribute('cx',p.x);r.setAttribute('cy',p.y);r.setAttribute('r',8);svg.append(r);}}
    tip.hidden=!active;
  }
  function pos(e){const r=host.getBoundingClientRect();return{x:(e.clientX-r.left)*host.clientWidth/r.width,y:(e.clientY-r.top)*host.clientHeight/r.height};}
  function down(e){if((!active&&!e.shiftKey)||e.button!==0||e.target.closest('button,input,select,a,label,.explore-playback,.explore-layer-controls'))return;e.preventDefault();e.stopImmediatePropagation();const p=pos(e);drag={...p,add:e.shiftKey};layer.setPointerCapture(e.pointerId);layer.classList.add('dragging');box.hidden=false;Object.assign(box.style,{left:p.x+'px',top:p.y+'px',width:'0px',height:'0px'});}
  function move(e){if(!drag)return;const p=pos(e);Object.assign(box.style,{left:Math.min(drag.x,p.x)+'px',top:Math.min(drag.y,p.y)+'px',width:Math.abs(p.x-drag.x)+'px',height:Math.abs(p.y-drag.y)+'px'});}
  function up(e){if(!drag)return;e.preventDefault();e.stopImmediatePropagation();const p=pos(e),d=drag;drag=null;box.hidden=true;layer.classList.remove('dragging');const r={x:Math.min(d.x,p.x),y:Math.min(d.y,p.y),width:Math.abs(d.x-p.x),height:Math.abs(d.y-p.y)};let result;
    if(r.width>5&&r.height>5&&adapter.region)result={kind:'region',bounds:adapter.region(r)};
    else {const points=adapter.points();let indices;if(r.width>5||r.height>5)indices=points.filter(q=>q.x>=r.x&&q.x<=r.x+r.width&&q.y>=r.y&&q.y<=r.y+r.height).map(q=>q.index);else{const near=points.map(q=>({...q,d:Math.hypot(q.x-p.x,q.y-p.y)})).sort((a,b)=>a.d-b.d)[0];indices=near&&near.d<30?[near.index]:[];}
      if(d.add&&mark?.kind==='points')indices=[...new Set([...mark.row_indices,...indices])];result=indices.length?{kind:'points',row_indices:indices}:null;
    }mark=result;paint();onChange(result);
  }
  host.addEventListener('pointerdown',down,true);layer.addEventListener('pointermove',move);layer.addEventListener('pointerup',up);layer.addEventListener('pointercancel',()=>{drag=null;box.hidden=true;layer.classList.remove('dragging');});
  host.addEventListener('projectionchange',paint);const ro=new ResizeObserver(paint);ro.observe(host);
  return {enable(value){active=value;layer.classList.toggle('active',active);paint();},set(value){mark=value;paint();},destroy(){ro.disconnect();host.removeEventListener('pointerdown',down,true);host.removeEventListener('projectionchange',paint);layer.remove();}};
}

export function plotPoints(plot,card){
  const layout=plot._fullLayout;if(!layout?.xaxis||!layout.yaxis)return [];
  const xi=card.data.columns.indexOf(card.x),yi=card.data.columns.indexOf(card.y),xa=layout.xaxis,ya=layout.yaxis;
  return card.data.rows.map((row,index)=>({index,x:xa._offset+xa.d2p(row[card.chart==='bar'?yi:xi]),y:ya._offset+ya.d2p(row[card.chart==='bar'?xi:yi])})).filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y));
}
