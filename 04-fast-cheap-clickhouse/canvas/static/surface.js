import {motionEnabled} from './motion.js?v=20261007-worlds1';
// A small, rotatable 3D surface drawn with Canvas2D. No WebGL shader startup.
// Missing grid values leave holes; sparse data is drawn as points, never filled with zero.
export function createSurface(host, card) {
  const canvas=document.createElement('canvas'),tip=document.createElement('div');
  canvas.setAttribute('aria-label',card.title+' — drag to rotate the 3D chart');canvas.tabIndex=0;
  tip.className='surface-tip';tip.textContent='Drag to rotate · Arrow keys to turn';canvas.style.cssText='display:block;width:100%;height:100%;touch-action:none';host.replaceChildren(canvas,tip);
  let yaw=-.65,pitch=.65,drag=null,frame=0,w=400,h=250,playing=false,loop=0,last=0,visible=true,viewScale=100,viewX=200,viewY=150;
  let c=canvas.getContext('2d');const ix=card.data.columns.indexOf(card.x),iy=card.data.columns.indexOf(card.y),iz=card.data.columns.indexOf(card.z);
  const xs=[...new Set(card.data.rows.map(r=>r[ix]))].sort((a,b)=>typeof a==='number'?a-b:String(a).localeCompare(String(b))),ys=[...new Set(card.data.rows.map(r=>r[iy]))].sort((a,b)=>typeof a==='number'?a-b:String(a).localeCompare(String(b)));
  const number=v=>v==null||v===''?NaN:Number(v);
  const values=card.data.rows.map(r=>number(r[iz])).filter(Number.isFinite),lo=Math.min(0,...values),hi=Math.max(...values),range=hi-lo||1;
  const map=new Map(card.data.rows.map(r=>[JSON.stringify([r[ix],r[iy]]),number(r[iz])]));
  const coordinate=(values,i)=>values.every(v=>typeof v==='number')?(values[i]-values[0])/(values.at(-1)-values[0]||1):i/Math.max(1,values.length-1);
  const points=ys.map((y,j)=>xs.map((x,i)=>{const value=map.get(JSON.stringify([x,y]));return Number.isFinite(value)?{x:(coordinate(xs,i)-.5)*1.8,y:(coordinate(ys,j)-.5)*1.8,z:(value-lo)/range*.95,value,index:card.data.rows.findIndex(r=>r[ix]===x&&r[iy]===y)}:null}));
  const hues={teal:174,blue:220,coral:16,violet:267,gold:40};
  function raw(p){const a=p.x*Math.cos(yaw)-p.y*Math.sin(yaw),b=p.x*Math.sin(yaw)+p.y*Math.cos(yaw);return{x:a,y:b*Math.sin(pitch)-p.z*Math.cos(pitch),depth:b*Math.cos(pitch)+p.z*Math.sin(pitch)}}
  function fitProjection(){const bounds=[...points.flat().filter(Boolean),{x:0,y:-1.16,z:0},{x:-1.19,y:0,z:0},{x:-.9,y:-.9,z:1.24},{x:.9,y:.9,z:0},{x:-.9,y:.9,z:0},{x:.9,y:-.9,z:0}].map(raw);const minX=Math.min(...bounds.map(p=>p.x)),maxX=Math.max(...bounds.map(p=>p.x)),minY=Math.min(...bounds.map(p=>p.y)),maxY=Math.max(...bounds.map(p=>p.y));viewScale=Math.min((w-80)/(maxX-minX||1),(h-48)/(maxY-minY||1));viewX=w/2-(minX+maxX)/2*viewScale;viewY=h/2-(minY+maxY)/2*viewScale-4;}
  function project(p){const q=raw(p);return{x:viewX+q.x*viewScale,y:viewY+q.y*viewScale,depth:q.depth}}

  function path(vertices,fill,stroke){c.beginPath();vertices.forEach((p,i)=>{const q=project(p);i?c.lineTo(q.x,q.y):c.moveTo(q.x,q.y)});c.closePath();if(fill){c.fillStyle=fill;c.fill()}if(stroke){c.strokeStyle=stroke;c.lineWidth=.65;c.stroke()}}
  function label(text,p,color='#94a3b4'){const q=project(p);c.font='9px system-ui';c.fillStyle=color;c.textAlign='center';c.fillText(String(text),q.x,q.y)}
  function draw(){frame=0;fitProjection();c.fillStyle='#0c1a25';c.fillRect(0,0,w,h);if(!values.length)return;
    for(let k=0;k<=4;k++){const v=-.9+k*.45;path([{x:-.9,y:v,z:0},{x:.9,y:v,z:0}],null,'#29424e');path([{x:v,y:-.9,z:0},{x:v,y:.9,z:0}],null,'#29424e')}
    const triangles=[];for(let j=0;j<ys.length-1;j++)for(let i=0;i<xs.length-1;i++){const a=points[j][i],b=points[j][i+1],d=points[j+1][i],e=points[j+1][i+1];if(a&&b&&e)triangles.push([a,b,e]);if(a&&d&&e)triangles.push([a,e,d])}
    triangles.sort((a,b)=>a.reduce((s,p)=>s+project(p).depth,0)-b.reduce((s,p)=>s+project(p).depth,0));
    for(const t of triangles){const z=t.reduce((s,p)=>s+p.z,0)/3;path(t,`hsl(${(hues[card.color]||174)-z*15} ${58+z*22}% ${30+z*28}%)`,'#ffffff24')}
    {for(const p of points.flat().filter(Boolean)){const q=project(p);c.fillStyle=`hsl(${hues[card.color]||174} 75% 65%)`;c.beginPath();c.arc(q.x,q.y,triangles.length<2?4:1.4,0,Math.PI*2);c.fill()}}
    path([{x:-.9,y:-.9,z:0},{x:.9,y:-.9,z:0}],null,'#577887');path([{x:-.9,y:-.9,z:0},{x:-.9,y:.9,z:0}],null,'#577887');path([{x:-.9,y:-.9,z:0},{x:-.9,y:-.9,z:1.1}],null,'#577887');
    label(card.x,{x:0,y:-1.16,z:0});label(card.y,{x:-1.19,y:0,z:0});label(card.z,{x:-.9,y:-.9,z:1.24});
    label(xs[0],{x:-.9,y:-1.02,z:0});label(xs.at(-1),{x:.9,y:-1.02,z:0});label(ys[0],{x:-1.02,y:-.9,z:0});label(ys.at(-1),{x:-1.02,y:.9,z:0});label(new Intl.NumberFormat('en',{notation:'compact',maximumFractionDigits:1}).format(hi),{x:-1.06,y:-.9,z:1.0});
  }
  function schedule(){if(!frame)frame=requestAnimationFrame(()=>{draw();host.dispatchEvent(new Event('projectionchange'));})}
  function tick(t){loop=0;if(!playing)return;if(visible&&!document.hidden&&motionEnabled()){yaw+=Math.min(50,t-(last||t))*.00022;schedule();}last=t;loop=requestAnimationFrame(tick);}
  function setPlaying(value){playing=!!value&&motionEnabled();cancelAnimationFrame(loop);loop=0;last=0;if(playing)loop=requestAnimationFrame(tick);host.dispatchEvent(new CustomEvent('orbitstate',{detail:playing}));}
  const io=new IntersectionObserver(entries=>visible=entries[0].isIntersecting);io.observe(host);
  const motionChange=()=>{if(!motionEnabled())setPlaying(false);};window.addEventListener('canvasmotionchange',motionChange);
  const resize=new ResizeObserver(()=>{const box=host.getBoundingClientRect();w=host.clientWidth||box.width;h=host.clientHeight||box.height;const ratio=Math.min(devicePixelRatio||1,2);canvas.width=w*ratio;canvas.height=h*ratio;c.setTransform(ratio,0,0,ratio,0,0);schedule()});resize.observe(host);
  canvas.onpointerdown=e=>{e.stopPropagation();setPlaying(false);drag={x:e.clientX,y:e.clientY,yaw,pitch};canvas.setPointerCapture(e.pointerId);host.style.cursor='grabbing'};
  canvas.onpointermove=e=>{if(!drag)return;yaw=drag.yaw+(e.clientX-drag.x)*.008;pitch=Math.max(.15,Math.min(1.4,drag.pitch+(e.clientY-drag.y)*.006));schedule()};
  canvas.onpointerup=()=>{drag=null;host.style.cursor='grab'};canvas.onpointercancel=canvas.onpointerup;
  canvas.onkeydown=e=>{if(['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)){e.preventDefault();yaw+=(e.key==='ArrowRight'?.1:e.key==='ArrowLeft'?-.1:0);pitch=Math.max(.15,Math.min(1.4,pitch+(e.key==='ArrowDown'?.08:e.key==='ArrowUp'?-.08:0)));schedule()}};
  return {setPlaying,isPlaying:()=>playing,points:()=>points.flat().filter(Boolean).map(p=>({...project(p),index:p.index})),exportFrames(count,width,height){const prior={c,w,h,yaw},off=document.createElement('canvas');off.width=width;off.height=height;c=off.getContext('2d',{willReadFrequently:true});w=width;h=height;const frames=[];try{for(let i=0;i<count;i++){yaw=prior.yaw+i/count*Math.PI*2;draw();frames.push(new Uint8Array(c.getImageData(0,0,w,h).data.buffer));}return frames;}finally{({c,w,h,yaw}=prior);schedule();}},destroy(){resize.disconnect();io.disconnect();window.removeEventListener('canvasmotionchange',motionChange);cancelAnimationFrame(frame);cancelAnimationFrame(loop)},ready:new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))};
}
