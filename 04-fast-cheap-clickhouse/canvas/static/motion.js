const media=matchMedia('(prefers-reduced-motion: reduce)');
const preferenceKey='canvas-motion-v2';
// Flowing is the canvas default. An explicit choice always wins over the OS;
// following the device preference is a separate, persistent setting.
// This revision explicitly enables the requested motion instead of carrying
// forward a legacy off flag. Subsequent choices remain persistent.
let preference=localStorage.getItem(preferenceKey)||'on';
if(!['on','off','system'].includes(preference))preference='on';
let enabled=preference==='on'||(preference==='system'&&!media.matches);
export const motionEnabled=()=>enabled;
function apply(){
  enabled=preference==='on'||(preference==='system'&&!media.matches);
  document.body.dataset.motion=preference;
  document.body.classList.toggle('motion-off',!enabled);
  if(!enabled)document.querySelectorAll('.data-card,#viewport,#messages').forEach(node=>node.getAnimations({subtree:true}).forEach(a=>a.cancel()));
  document.querySelector('#motion-select').value=preference;
  window.dispatchEvent(new Event('canvasmotionchange'));
}
export function initMotion(){
  document.querySelector('#motion-select').onchange=e=>{preference=e.target.value;localStorage.setItem(preferenceKey,preference);apply();};
  media.addEventListener('change',()=>{if(preference==='system')apply();});
  apply();
}
export function animateCard(node,adding=false,{fromScale=.78}={}){if(!enabled)return;return node.animate(adding?[{opacity:0,transform:`scale(${fromScale})`},{opacity:1,transform:'scale(1)'}]:[{filter:'brightness(1.5)',transform:'scale(.985)'},{filter:'brightness(1)',transform:'scale(1)'}],{duration:adding?1800:850,easing:'cubic-bezier(.42,0,.28,1)',fill:'backwards'});}
export function animatePending(node){
  let animations=[];
  const cancel=()=>{animations.forEach(a=>a.cancel());animations=[];};
  const breathe=()=>{
    cancel();if(!enabled)return;
    // A visible, continuous pulse while the actual request is outstanding.
    // These animations are owned by the pending view, independently of CSS.
    animations.push(node.querySelector('.blueprint-seed').animate([
      {transform:'scale(.86)',opacity:.55},
      {transform:'scale(1.08)',opacity:1},
      {transform:'scale(.86)',opacity:.55}
    ],{duration:2600,iterations:Infinity,easing:'ease-in-out'}));
    node.querySelectorAll('.blueprint-ring').forEach((ring,i)=>animations.push(ring.animate([
      {transform:'rotate(0deg)'},{transform:`rotate(${i===1?-360:360}deg)`}
    ],{duration:[6000,9000,15000][i],iterations:Infinity,easing:'linear'})));
  };
  const arrival=animateCard(node,true,{fromScale:.72});
  breathe();window.addEventListener('canvasmotionchange',breathe);
  return ()=>{arrival?.cancel();cancel();window.removeEventListener('canvasmotionchange',breathe);};
}
export function setWorking(phase){document.body.dataset.phase=phase||'idle';}
export async function exportOrbit(renderer,title,notify){
  const frames=renderer.exportFrames(36,480,320),worker=new Worker('/canvas/gif-worker.js',{type:'module'});
  try {const bytes=await new Promise((resolve,reject)=>{worker.onmessage=e=>{if(e.data.error)reject(Error(e.data.error));else if(e.data.progress)notify(`Encoding GIF · ${e.data.progress}%`);else resolve(e.data.bytes);};worker.onerror=e=>reject(Error(e.message));worker.postMessage({frames,width:480,height:320},frames.map(x=>x.buffer));});
    const url=URL.createObjectURL(new Blob([bytes],{type:'image/gif'}));const dialog=document.createElement('dialog');dialog.className='photo-dialog gif-dialog';const close=document.createElement('button');close.className='photo-close';close.textContent='×';close.setAttribute('aria-label','Close animation');close.onclick=()=>dialog.close();const img=document.createElement('img');img.src=url;img.alt=title+' — animated 3D orbit';const text=document.createElement('p');text.textContent='3D camera orbit · 3.6 seconds · same query result';const link=document.createElement('a');link.textContent='Download GIF ↓';link.href=url;link.download='wildlife-canvas-orbit.gif';dialog.append(close,img,text,link);document.body.append(dialog);dialog.onclose=()=>{URL.revokeObjectURL(url);dialog.remove();};dialog.showModal();notify('GIF ready');
  }finally{worker.terminate();}
}
