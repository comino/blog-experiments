// One camera owns the screen transform. Native scrolling never moves the board.
export function createCamera({viewport,board,spacer,label,getCards,motion,onZoom,onEvent}) {
  const FIT_MS=2200,MIN_ZOOM=.08,MAX_ZOOM=4;
  let current={x:0,y:0,zoom:1},journey=null,raf=0,following=true;
  let bounds,area,drag=null,space=false;
  const clamp=z=>Math.max(MIN_ZOOM,Math.min(MAX_ZOOM,z));
  const close=(a,b)=>Math.abs(a.zoom-b.zoom)<.00001&&Math.abs(a.x-b.x)<.01&&Math.abs(a.y-b.y)<.01;
  const smooth=t=>t*t*t*(t*(t*6-15)+10);

  function measure() {
    const cards=getCards(),r=viewport.getBoundingClientRect(),pad=r.width<640?18:28;
    const bottom=document.querySelector('.canvas-bottom').getBoundingClientRect();
    area={left:pad,right:pad,top:pad,bottom:Math.max(pad,r.bottom-bottom.top+18)};
    area.width=Math.max(1,viewport.clientWidth-area.left-area.right);
    area.height=Math.max(1,viewport.clientHeight-area.top-area.bottom);
    area.cx=area.left+area.width/2;area.cy=area.top+area.height/2;
    bounds=cards.length?{
      left:Math.min(...cards.map(c=>c.left)),top:Math.min(...cards.map(c=>c.top)),
      right:Math.max(...cards.map(c=>c.left+c.width)),bottom:Math.max(...cards.map(c=>c.top+c.height))
    }:{left:0,top:0,right:0,bottom:0};
    board.style.width=Math.max(1,bounds.right)+'px';board.style.height=Math.max(1,bounds.bottom)+'px';
  }
  function paint(view) {
    current={...view};
    board.style.transform=`translate3d(${view.x}px,${view.y}px,0) scale(${view.zoom})`;
    label.textContent=Math.round(view.zoom*100)+'%';onZoom(view.zoom);
    // The dots move with the scene, making the camera's motion visible too.
    const grid=Math.max(6,24*view.zoom);
    viewport.style.backgroundSize=`${grid}px ${grid}px`;
    viewport.style.backgroundPosition=`${view.x+grid/2}px ${view.y+grid/2}px`;
  }
  function stop() {
    cancelAnimationFrame(raf);raf=0;journey=null;
    board.dataset.framing='false';board.style.willChange='';
  }
  function step(now) {
    raf=0;const run=journey;if(!run)return;
    // A heavy chart render or a background tab must not consume the movement
    // before the user gets any frames. Resume from the last painted position.
    run.elapsed+=Math.min(48,Math.max(0,now-run.last));run.last=now;
    const t=Math.min(1,run.elapsed/run.duration),k=run.gesture?1-(1-t)**3:smooth(t);
    paint({x:run.from.x+(run.to.x-run.from.x)*k,y:run.from.y+(run.to.y-run.from.y)*k,zoom:run.from.zoom+(run.to.zoom-run.from.zoom)*k});
    if(t<1){raf=requestAnimationFrame(step);return;}
    stop();paint(run.to);onEvent('camera-settled',{reason:run.reason,...current,duration_ms:run.duration});
  }
  function move(next,{animate=true,reason='fit',duration=FIT_MS,gesture=false}={}) {
    if(journey&&close(journey.to,next))return;
    const from={...current};stop();
    duration=animate&&motion()&&!close(from,next)?duration:0;
    onEvent('camera-start',{reason,from,to:next,duration_ms:duration,motion:motion()});
    if(!duration){paint(next);onEvent('camera-settled',{reason,...next,duration_ms:0});return;}
    journey={from,to:next,reason,duration,gesture,elapsed:0,last:performance.now()};
    board.dataset.framing='true';board.style.willChange='transform';
    raf=requestAnimationFrame(step);
  }
  function layout(){measure();}
  function fit(options={}) {
    following=true;measure();
    if(!getCards().length){move({x:0,y:0,zoom:1},options);return;}
    const zoom=Math.min(2,area.width/(bounds.right-bounds.left),area.height/(bounds.bottom-bounds.top));
    move({zoom,x:area.cx-(bounds.left+bounds.right)*zoom/2,y:area.cy-(bounds.top+bounds.bottom)*zoom/2},options);
  }
  function zoomTo(value,{anchor,duration=320,reason='zoom'}={}) {
    following=false;measure();const p=anchor||{x:area.cx,y:area.cy},zoom=clamp(value);
    move({zoom,x:p.x-(p.x-current.x)*zoom/current.zoom,y:p.y-(p.y-current.y)*zoom/current.zoom},{duration,reason,gesture:true});
  }
  function focus(card) {
    following=false;measure();const zoom=Math.min(2,area.width/card.width,area.height/card.height);
    move({zoom,x:area.cx-(card.left+card.width/2)*zoom,y:area.cy-(card.top+card.height/2)*zoom},{reason:'focus'});
  }
  viewport.addEventListener('wheel',e=>{
    // A table/gallery keeps its own scrolling. Pinch/Ctrl-wheel always zooms.
    const scroll=e.target.closest('.table-view,.gallery-view,.leaflet-popup-content,select,textarea');
    if(!e.ctrlKey&&!e.metaKey&&scroll&&(scroll.matches('select,textarea')||scroll.scrollHeight>scroll.clientHeight+1))return;
    e.preventDefault();e.stopPropagation();
    const unit=e.deltaMode===1?16:e.deltaMode===2?viewport.clientHeight:1;
    if(e.shiftKey){following=false;stop();paint({...current,x:current.x-(e.deltaY||e.deltaX)*unit});return;}
    if(!e.deltaY)return;
    const r=viewport.getBoundingClientRect(),delta=Math.max(-600,Math.min(600,e.deltaY*unit));
    const base=journey?.gesture?journey.to.zoom:current.zoom;
    zoomTo(base*Math.exp(-delta*(e.ctrlKey ? .006 : .002)),{anchor:{x:e.clientX-r.left,y:e.clientY-r.top},duration:180,reason:'wheel'});
  },{passive:false,capture:true});

  viewport.addEventListener('pointerdown',e=>{
    const canPan=e.button===1||(e.button===0&&(space||!e.target.closest('.data-card')));
    stop();
    if(!canPan)return;
    following=false;e.preventDefault();e.stopPropagation();viewport.setPointerCapture(e.pointerId);
    drag={id:e.pointerId,x:e.clientX,y:e.clientY,view:{...current}};viewport.dataset.panning='true';
  },{capture:true});
  viewport.addEventListener('pointermove',e=>{
    if(!drag||drag.id!==e.pointerId)return;e.preventDefault();
    paint({...drag.view,x:drag.view.x+e.clientX-drag.x,y:drag.view.y+e.clientY-drag.y});
  });
  function endDrag(e){if(!drag||e.pointerId!==drag.id)return;drag=null;viewport.dataset.panning='false';if(viewport.hasPointerCapture(e.pointerId))viewport.releasePointerCapture(e.pointerId);}
  viewport.addEventListener('pointerup',endDrag);viewport.addEventListener('pointercancel',endDrag);viewport.addEventListener('lostpointercapture',endDrag);
  viewport.addEventListener('auxclick',e=>{if(e.button===1)e.preventDefault();});
  window.addEventListener('keydown',e=>{if(e.code==='Space'&&!e.target.closest('input,textarea,select,button,[contenteditable="true"]')){space=true;viewport.dataset.panReady='true';e.preventDefault();}});
  const releaseSpace=()=>{space=false;viewport.dataset.panReady='false';};
  window.addEventListener('keyup',e=>{if(e.code==='Space')releaseSpace();});window.addEventListener('blur',releaseSpace);
  window.addEventListener('canvasmotionchange',()=>{if(!motion()&&journey){const next=journey.to;stop();paint(next);onEvent('camera-settled',{reason:'motion-off',...next,duration_ms:0});}});
  // No scroll offsets, scrollable spacer, or browser scroll anchoring to snap.
  spacer.style.width='100%';spacer.style.height='100%';viewport.scrollLeft=viewport.scrollTop=0;
  measure();paint(current);
  new ResizeObserver(()=>{if(following)fit({reason:'resize'});else layout();}).observe(viewport);
  return {fit,zoomTo,focus,layout,stop};
}
