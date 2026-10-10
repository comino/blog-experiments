const el = (tag, cls, text) => { const n=document.createElement(tag); if(cls)n.className=cls;if(text!=null)n.textContent=text;return n; };
const colors={teal:'#49e2c0',blue:'#65adff',coral:'#ff997f',violet:'#ba9dff',gold:'#f2d583'};
let world;
const countryData=()=>world ||= fetch('/canvas/vendor/world.json').then(r=>{if(!r.ok)throw Error('Basemap unavailable');return r.json();});
export async function createMap(host,card,ask,previousView) {
  const L=window.L;
  const container=el('div','map-view');host.append(container);
  const map=L.map(container,{crs:L.CRS.EPSG4326,zoomControl:true,attributionControl:true,minZoom:0,maxZoom:7,scrollWheelZoom:false,zoomAnimation:false,preferCanvas:true}).setView([15,0],0);
  map.attributionControl.setPrefix(false);
  map.attributionControl.addAttribution('<a href="https://www.naturalearthdata.com/" target="_blank" rel="noopener">Natural Earth</a> · Public iNaturalist locations');
  L.geoJSON(await countryData(),{interactive:false,style:{color:'#2c4c5c',weight:.65,fillColor:'#152d3b',fillOpacity:1}}).addTo(map);
  const {columns,rows}=card.data, xi=columns.indexOf(card.x),yi=columns.indexOf(card.y),zi=columns.indexOf(card.z);
  const nums=rows.map(r=>zi>=0?Number(r[zi]):1).filter(Number.isFinite),max=Math.max(1,...nums),points=[];
  rows.forEach(r=>{
    if(r[xi]==null||r[yi]==null)return;
    const lng=Number(r[xi]),lat=Number(r[yi]);if(!Number.isFinite(lng)||!Number.isFinite(lat)||Math.abs(lng)>180||Math.abs(lat)>90)return;
    points.push([lat,lng]);const value=zi>=0?Number(r[zi]):1;
    const marker=L.circleMarker([lat,lng],{radius:zi>=0?3+12*Math.sqrt(Math.max(0,value)/max):5,weight:.7,color:'#8af9e5',fillColor:colors[card.color]||colors.teal,fillOpacity:.8}).addTo(map);
    const popup=el('div','map-popup');
    columns.slice(0,8).forEach((c,i)=>{if(c==='observation_uuid')return;popup.append(el('div','',c.replaceAll('_',' ')+': '+(r[i]??'—')));});
    marker.bindPopup(popup);
  });
  const fitData=()=>{if(points.length)map.fitBounds(L.latLngBounds(points),{padding:[25,25],maxZoom:5,animate:false});};
  fitData();
  if(previousView)map.setView(previousView.center,previousView.zoom,{animate:false});
  let moved=!!previousView;
  map.on('movestart zoomstart',()=>moved=true);map.on('move zoom resize',()=>host.dispatchEvent(new Event('projectionchange')));
  const fit=el('button','map-fit-button','Fit data');fit.type='button';fit.title='Fit the map to all returned locations';fit.onclick=()=>{fitData();moved=false;};host.append(fit);
  const action=el('button','map-photo-button','▧ Photos in this area');action.type='button';action.title='Add a photo card using this map’s subject and visible bounds';
  action.addEventListener('click',()=>{const b=map.getBounds();ask(card.id,`Add a photo gallery of the same subject and dates as this map, within the current map area: latitude between ${Math.max(-90,b.getSouth()).toFixed(4)} and ${Math.min(90,b.getNorth()).toFixed(4)}, longitude between ${Math.max(-180,b.getWest()).toFixed(4)} and ${Math.min(180,b.getEast()).toFixed(4)}. Select up to 36 observations for licensed-photo lookup.`);});
  host.append(action);
  const ro=new ResizeObserver(()=>map.invalidateSize({animate:false,pan:false}));ro.observe(host);
  await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
  return {destroy(){ro.disconnect();map.remove();action.remove();fit.remove();},ready:Promise.resolve(),map,points:()=>rows.map((r,index)=>{const p=map.latLngToContainerPoint([Number(r[yi]),Number(r[xi])]);return {index,x:p.x+container.offsetLeft,y:p.y+container.offsetTop};}),region:r=>{const a=map.containerPointToLatLng([r.x-container.offsetLeft,r.y-container.offsetTop]),b=map.containerPointToLatLng([r.x+r.width-container.offsetLeft,r.y+r.height-container.offsetTop]);return {west:Math.max(-180,a.lng),east:Math.min(180,b.lng),south:Math.max(-90,b.lat),north:Math.min(90,a.lat)};},regionPixels:b=>{const a=map.latLngToContainerPoint([b.north,b.west]),d=map.latLngToContainerPoint([b.south,b.east]);return {x:a.x+container.offsetLeft,y:a.y+container.offsetTop,width:d.x-a.x,height:d.y-a.y};},view:()=>moved?{center:map.getCenter(),zoom:map.getZoom()}:undefined};
}

function photoURL(value){return typeof value==='string'&&/^https:\/\/inaturalist-open-data\.s3\.amazonaws\.com\/photos\/\d+\/medium\.(jpg|jpeg|png|webp)$/.test(value)?value:null;}
function link(label,url){const a=el('a','',label);a.href=url;a.target='_blank';a.rel='noopener noreferrer';return a;}
export function createGallery(host,card,onMark) {
  const gallery=el('div','gallery-view'),{columns,rows}=card.data;
  host.append(gallery);
  const get=(r,k)=>r[columns.indexOf(k)];
  const photos=rows.filter(r=>photoURL(get(r,'photo_url')));
  const note=el('div','gallery-note',`${photos.length} licensed photos from ${rows.length} selected observations · CC0 / CC-BY`);gallery.append(note);
  const grid=el('div','gallery-grid');gallery.append(grid);
  let loaded=0,errors=0;
  const promises=[];
  photos.forEach((row,i)=>{
    const url=photoURL(get(row,'photo_url')),name=get(row,'scientific_name')||'Wildlife observation',date=get(row,'observed_on')||'',credit=get(row,'photo_credit')||'iNaturalist contributor',license=get(row,'photo_license');
    const figure=el('figure','photo-tile'),button=el('button','photo-open'),img=el('img');
    figure.dataset.rowIndex=rows.indexOf(row);button.type='button';button.setAttribute('aria-label',`View ${name}, ${date}`);img.alt=`${name}, observed ${date}`;img.loading=i<6?'eager':'lazy';img.decoding='async';
    let resolve;const ready=new Promise(r=>resolve=r);if(i<6)promises.push(ready);
    img.onload=()=>{loaded++;figure.classList.add('loaded');resolve();};img.onerror=()=>{errors++;img.remove();button.append(el('span','photo-missing','Photo no longer available'));resolve();};img.src=url;
    button.append(img);figure.append(button);const cap=el('figcaption');cap.append(el('strong','',name),el('span','',String(date)));
    const source='https://www.inaturalist.org/photos/'+get(row,'photo_id');
    cap.append(link(`${license==='CC0'?'':'© '}${credit} · ${license}`,source));figure.append(cap);grid.append(figure);const mark=el('button','photo-mark','⌖');mark.type='button';mark.title='Mark this observation';mark.setAttribute('aria-label','Mark this observation');mark.onclick=()=>onMark?.(rows.indexOf(row));figure.append(mark);
    button.addEventListener('click',()=>{
      const dialog=el('dialog','photo-dialog'),close=el('button','photo-close','×');close.setAttribute('aria-label','Close photo');close.onclick=()=>dialog.close();
      const large=el('img');large.alt=img.alt;large.src=url.replace('/medium.','/large.');
      const detail=el('div','photo-detail');detail.append(el('h2','',name),el('p','',String(date)),link(`${license==='CC0'?'':'© '}${credit} · ${license}`,source));
      if(/^[0-9a-f-]{36}$/i.test(get(row,'observation_uuid')||''))detail.append(link('Open original observation ↗','https://www.inaturalist.org/observations/'+get(row,'observation_uuid')));
      dialog.append(close,large,detail);document.body.append(dialog);dialog.addEventListener('close',()=>dialog.remove());dialog.addEventListener('click',e=>{if(e.target===dialog)dialog.close();});dialog.showModal();
    });
  });
  if(!photos.length)gallery.append(el('div','gallery-empty','No CC0 or CC-BY photos were found in this selection. Try a broader area or a different species.'));
  return {destroy(){gallery.remove();},ready:Promise.all(promises),stats:()=>({photos:photos.length,loaded,errors})};
}
