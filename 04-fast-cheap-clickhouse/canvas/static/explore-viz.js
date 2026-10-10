// Local, query-driven visualizations. Playback never calls the model.
// Tracks contain measured fixes; short gaps may be interpolated, long gaps stay empty.
const palette = ['#65e4d0', '#77baff', '#ceb0ff', '#f0d383', '#ffaf91', '#a6de92'];
const monthNames = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const number = value => value == null || value === '' ? NaN : Number(value);
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const compact = value => new Intl.NumberFormat('en', {notation:'compact', maximumFractionDigits:1}).format(value);
const formatNumber = value => new Intl.NumberFormat('en', {maximumFractionDigits:2}).format(value);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text != null) node.textContent = text;
  return node;
};
const button = (label, title, action, cls = '') => {
  const node = el('button', `explore-button ${cls}`, label);
  node.type = 'button'; node.title = title; node.setAttribute('aria-label', title);
  node.addEventListener('click', action);
  return node;
};
const fieldIndex = (columns, ...names) => names.map(name => columns.indexOf(name)).find(index => index >= 0) ?? -1;
function timestamp(value) {
  if (typeof value === 'number') return value < 1e12 ? value * 1000 : value;
  if (typeof value !== 'string' || !value.trim()) return NaN;
  const text = value.trim().replace(' ', 'T');
  return Date.parse(/^\d{4}-\d{2}-\d{2}$/.test(text) ? text + 'T00:00:00Z' : /(?:Z|[+-]\d{2}:?\d{2})$/i.test(text) ? text : text + 'Z');
}
const utcLabel = time => new Date(time).toLocaleString('en-GB', {timeZone:'UTC', day:'2-digit', month:'short', hour:'2-digit', minute:'2-digit', hour12:false}) + ' UTC';
const utcDay = time => new Date(time).toISOString().slice(0, 10);
const durationLabel = ms => ms >= 3600000 ? `${formatNumber(ms / 3600000)}h` : `${formatNumber(ms / 60000)}m`;
const flowing = () => !document.body.classList.contains('motion-off');
let worldPromise;
function worldData() {
  return worldPromise ||= fetch('/canvas/vendor/world.json').then(response => {
    if (!response.ok) throw Error('Basemap unavailable');
    return response.json();
  }).catch(error => { worldPromise = null; throw error; });
}
function localReady() { return new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve))); }
function addLegend(host, label, color) {
  const item = el('span', 'explore-legend-item'), swatch = el('i');
  swatch.style.background = color; item.append(swatch, document.createTextNode(label)); host.append(item);
}
function normalizeData(data, card, sourceIndices = null, forceStorm = false) {
  const columns = data?.columns || [], rows = data?.rows || [];
  const ix = fieldIndex(columns, card.x, 'longitude', 'lon', 'lng');
  const iy = fieldIndex(columns, card.y, 'latitude', 'lat');
  const it = fieldIndex(columns, 'observed_at', 'timestamp', 'time', 'date');
  const iid = fieldIndex(columns, 'vessel_id', 'mmsi', 'storm_id', 'sid');
  const iname = fieldIndex(columns, 'vessel_name', 'storm_name', 'name');
  const ikind = fieldIndex(columns, 'track_kind', 'kind');
  const ispeed = fieldIndex(columns, 'speed_knots', 'speed', 'sog');
  const icourse = fieldIndex(columns, 'course_deg', 'heading_deg', 'course', 'cog');
  const iwind = fieldIndex(columns, 'wind_knots', 'usa_wind');
  const icategory = fieldIndex(columns, 'category_code', 'category', 'usa_sshs');
  const iinterpolation = fieldIndex(columns, 'usa_interpolation');
  const ivalue = fieldIndex(columns, card.z, 'count', 'observations', 'value');
  const itype = fieldIndex(columns, 'vessel_type');
  return rows.map((row, index) => {
    const lon = number(row[ix]), lat = number(row[iy]);
    if (!Number.isFinite(lon) || !Number.isFinite(lat) || Math.abs(lon) > 180 || Math.abs(lat) > 90) return null;
    const storm = forceStorm || row[ikind] === 'storm' || columns.includes('storm_id') && !columns.includes('vessel_id');
    const id = String(row[iid] ?? (storm ? row[iname] || 'storm' : 'vessel'));
    const categoryValue = row[icategory], category = number(categoryValue);
    return {lon, lat, time:timestamp(row[it]), id, name:String(row[iname] || id), storm,
      speed:number(row[ispeed]), course:number(row[icourse]), wind:number(row[iwind]),
      category:Number.isFinite(category)?category:number(typeof categoryValue==='string'?categoryValue.match(/(?:category|cat|c)\s*([1-5])/i)?.[1]:null),
      sourceInterpolation:String(row[iinterpolation] || ''),
      type:String(row[itype] || ''), value:ivalue >= 0 ? number(row[ivalue]) : 1,
      index:sourceIndices ? sourceIndices.get(id) ?? -1 : index, row};
  }).filter(Boolean);
}

async function baseMap(host, card, options) {
  const L = window.L;
  if (!L) throw Error('Map renderer is unavailable');
  const geographic = card.chart !== 'tracks' && card.dataset_id !== 'maritime';
  const latitudeLimit = geographic ? 90 : 85;
  const root = el('div', 'explore-viz explore-map-shell'), container = el('div', 'explore-map');
  root.append(container); host.append(root);
  const map = L.map(container, {crs:geographic?L.CRS.EPSG4326:L.CRS.EPSG3857, zoomControl:false, attributionControl:true, minZoom:geographic?-2:0, maxZoom:12, zoomSnap:.25, zoomDelta:.5,
    scrollWheelZoom:false, zoomAnimation:false, fadeAnimation:false, preferCanvas:true, worldCopyJump:false}).setView([22, -82], 3);
  map.attributionControl.setPrefix(false);
  map.attributionControl.addAttribution('<a href="https://www.naturalearthdata.com/" target="_blank" rel="noopener">Natural Earth</a>');
  const controls = el('div', 'explore-map-actions');
  controls.append(button('+', 'Zoom into map', () => map.zoomIn()), button('−', 'Zoom out of map', () => map.zoomOut()));
  const fitButton = button('⌗', 'Fit all returned locations', () => fit()); controls.append(fitButton); root.append(controls);
  // Let the page camera retain wheel/pinch; buttons and drag operate the map.
  L.DomEvent.disableClickPropagation(controls); L.DomEvent.disableScrollPropagation(controls);
  const top = el('div', 'explore-map-top'), badge = el('span', 'explore-badge'); top.append(badge); root.append(top);
  const status = el('div', 'explore-map-status'); root.append(status);
  const tooltip = el('div', 'explore-tooltip'); tooltip.hidden = true; root.append(tooltip);
  let destroyed = false, bounds = null, moved = !!options.previousView, countries;
  let painted = () => {};
  const project = point => map.latLngToContainerPoint([clamp(point.lat, -latitudeLimit, latitudeLimit), point.lon]);
  function fit() {
    if (bounds?.isValid()) map.fitBounds(bounds, {paddingTopLeft:[30, 50], paddingBottomRight:[35, card.chart === 'tracks' ? 118 : 62], maxZoom:8, animate:false});
    moved = false;
  }
  function setBounds(points) {
    if (!points.length) return;
    bounds = L.latLngBounds(points.map(point => [clamp(point.lat, -latitudeLimit, latitudeLimit), point.lon])); fit();
    if (options.previousView) map.setView(options.previousView.center, options.previousView.zoom, {animate:false});
  }
  function showTip(point, lines) {
    tooltip.replaceChildren(...lines.map((line, i) => el(i ? 'span' : 'strong', '', line)));
    tooltip.hidden = false;
    const p = project(point);
    tooltip.style.left = clamp(p.x + 12, 8, Math.max(8, container.clientWidth - 205)) + 'px';
    tooltip.style.top = clamp(p.y - 15, 42, Math.max(42, container.clientHeight - 155)) + 'px';
  }
  map.on('movestart zoomstart', () => {moved = true; tooltip.hidden = true;});
  map.on('move zoom resize', () => {painted(); host.dispatchEvent(new Event('projectionchange'));});
  map.on('mouseout', () => tooltip.hidden = true);
  const resize = () => {if (!destroyed) {map.invalidateSize({animate:false, pan:false}); if(bounds&&!moved)fit(); painted();}};
  const observer = new ResizeObserver(resize); observer.observe(container);
  try {
    const world = await worldData();
    if (!destroyed) countries = L.geoJSON(world, {interactive:false, style:{color:'#385a65', weight:.7, fillColor:'#17303a', fillOpacity:1}}).addTo(map);
  } catch { status.textContent = 'Coastline unavailable · query locations remain visible'; }
  return {root, container, map, top, badge, status, controls, tooltip, project, fit, setBounds, showTip,
    setDraw(callback) {painted = callback;}, resize, countries,
    region(rect) {
      const a = map.containerPointToLatLng([rect.x - root.offsetLeft, rect.y - root.offsetTop]);
      const b = map.containerPointToLatLng([rect.x + rect.width - root.offsetLeft, rect.y + rect.height - root.offsetTop]);
      return {west:clamp(a.lng,-180,180), east:clamp(b.lng,-180,180), south:clamp(b.lat,-90,90), north:clamp(a.lat,-90,90)};
    },
    regionPixels(bounds) {
      const a = map.latLngToContainerPoint([bounds.north, bounds.west]), b = map.latLngToContainerPoint([bounds.south, bounds.east]);
      return {x:a.x + root.offsetLeft, y:a.y + root.offsetTop, width:b.x-a.x, height:b.y-a.y};
    },
    view:() => ({center:map.getCenter(), zoom:map.getZoom()}),
    destroy() {if(destroyed)return;destroyed = true; observer.disconnect(); map.remove(); root.remove();}
  };
}

function canvasLayer(base) {
  const canvas = el('canvas', 'explore-overlay'); base.root.append(canvas);
  const context = canvas.getContext('2d');
  function start() {
    const width = base.container.clientWidth, height = base.container.clientHeight;
    const cameraScale = base.container.getBoundingClientRect().width / Math.max(1, width);
    const ratio = Math.min(4, (window.devicePixelRatio || 1) * Math.max(1, cameraScale));
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0); context.clearRect(0, 0, width, height);
    return {c:context, width, height};
  }
  return {canvas, start};
}

function gridStep(values) {
  const sorted = [...new Set(values.map(value => Math.round(value * 1e6) / 1e6))].sort((a,b) => a-b);
  if (sorted.length < 3) return NaN;
  const gaps = sorted.slice(1).map((value,index) => value-sorted[index]).filter(gap => gap > 1e-5);
  const step = Math.min(...gaps);
  if (step < .005 || step > 10) return NaN;
  return gaps.every(gap => Math.abs(gap / step - Math.round(gap / step)) < .005) ? step : NaN;
}

async function createDensity(host, card, options) {
  const base = await baseMap(host, card, options), layer = canvasLayer(base);
  if(!host.isConnected){base.destroy();return{ready:Promise.resolve(),points:()=>[],destroy(){}};}
  const points = normalizeData(card.data, card), values = points.map(point => point.value).filter(Number.isFinite);
  const maximum = Math.max(0, ...values), minimum = Math.min(0, ...values);
  const explicitStepIndex = card.data.columns.indexOf('cell_size_deg');
  const explicitStep = number(card.cell_size_deg ?? card.data.rows[0]?.[explicitStepIndex]);
  const dx = explicitStep > 0 ? explicitStep : gridStep(points.map(point => point.lon));
  const dy = explicitStep > 0 ? explicitStep : gridStep(points.map(point => point.lat));
  const hasGrid = card.chart === 'density' && Number.isFinite(dx) && Number.isFinite(dy);
  let cells = hasGrid, destroyed = false, hitPoints = [];
  base.setBounds(points);
  base.badge.textContent = `${compact(points.length)} ${hasGrid ? 'cells' : 'locations'}`;
  base.map.attributionControl.addAttribution(card.dataset_id === 'maritime' ? 'NOAA MarineCadastre / IBTrACS' : 'Public iNaturalist locations');
  const legend = el('div', 'explore-scale');
  const title = el('span', '', (card.z || 'Locations').replaceAll('_',' '));
  const gradient = el('i', 'explore-gradient');
  const range = el('div', 'explore-scale-range'); range.append(el('span','', compact(minimum)), el('span','', compact(maximum)));
  const scaleNote = el('small', '', maximum > 0 && minimum >= 0 ? 'Log colour · returned values' : 'Linear colour · returned values');
  legend.append(title, gradient, range, scaleNote); base.root.append(legend);
  if (hasGrid) {
    const mode = button('Cells', 'Toggle geographic cells and weighted points', () => {cells = !cells; mode.textContent = cells ? 'Cells' : 'Points'; draw();});
    base.top.append(mode);
    base.badge.title = `${formatNumber(dx)}° longitude × ${formatNumber(dy)}° latitude; returned coordinate centres`;
  }
  function draw() {
    if (destroyed) return;
    const {c, width, height} = layer.start(); hitPoints = [];
    for (const point of points) {
      const p = base.project(point); if (p.x < -30 || p.y < -30 || p.x > width+30 || p.y > height+30 || !Number.isFinite(point.value)) continue;
      const ratio = minimum >= 0 ? Math.log1p(Math.max(0, point.value)) / (Math.log1p(maximum) || 1) : (point.value-minimum)/(maximum-minimum || 1);
      c.fillStyle = `hsl(${188-ratio*30} ${55+ratio*20}% ${24+ratio*44}%)`;
      if (cells) {
        const a = base.project({lat:point.lat+dy/2,lon:point.lon-dx/2});
        const b = base.project({lat:point.lat-dy/2,lon:point.lon+dx/2});
        c.globalAlpha = .85; c.fillRect(a.x+.4,a.y+.4,Math.max(1,b.x-a.x-.8),Math.max(1,b.y-a.y-.8));
      } else {
        const radius = card.z ? 3+11*Math.sqrt(clamp(ratio,0,1)) : 4;
        c.globalAlpha = .73; c.beginPath(); c.arc(p.x,p.y,radius,0,Math.PI*2); c.fill();
        c.globalAlpha = .9; c.strokeStyle = '#baffea55'; c.lineWidth = .6; c.stroke();
      }
      hitPoints.push({...point,x:p.x,y:p.y});
    }
    c.globalAlpha = 1;
  }
  const nearest = e => hitPoints.map(point => ({point,d:Math.hypot(point.x-e.containerPoint.x,point.y-e.containerPoint.y)})).sort((a,b)=>a.d-b.d)[0];
  base.map.on('mousemove', event => {
    const hit = nearest(event); if (!hit || hit.d > 25) {base.tooltip.hidden = true; return;}
    base.showTip(hit.point, [`${formatNumber(hit.point.value)} ${(card.z || 'locations').replaceAll('_',' ')}`, `${hit.point.lat.toFixed(3)}°, ${hit.point.lon.toFixed(3)}°`, cells ? `${formatNumber(dx)}° × ${formatNumber(dy)}° cell` : 'Returned query location']);
  });
  base.map.on('click', event => {const hit=nearest(event); if (hit?.d < 25) options.onSelectRows?.([hit.point.index]);});
  base.setDraw(draw); draw();
  if (!points.length) {base.status.textContent = 'No geographic locations in this result'; base.status.classList.add('is-visible');}
  await localReady();
  return {...base, ready:Promise.resolve(), points:() => points.map(point => {const p=base.project(point);return {index:point.index,x:p.x+base.root.offsetLeft,y:p.y+base.root.offsetTop};}),
    destroy() {destroyed = true; base.destroy();}};
}

function groupTracks(points) {
  const groups = new Map();
  for (const point of points) {
    if (!Number.isFinite(point.time)) continue;
    const key = `${point.storm ? 'storm' : 'vessel'}:${point.id}`;
    if (!groups.has(key)) groups.set(key, {id:point.id,name:point.name,storm:point.storm,points:[],color:point.storm?'#ffb183':palette[groups.size % palette.length]});
    groups.get(key).points.push(point);
  }
  for (const group of groups.values()) {
    group.points.sort((a,b)=>a.time-b.time);
    group.points = group.points.filter((point,index,array)=>!index || point.time !== array[index-1].time);
  }
  return [...groups.values()];
}

function headAt(track, time, maxGap) {
  const points = track.points;
  let low=0,high=points.length;
  while (low<high) {const mid=(low+high)>>1; if (points[mid].time<=time) low=mid+1; else high=mid;}
  const a=points[low-1], b=points[low];
  if (!a || time-a.time>maxGap) return null;
  if (!b || b.time-a.time>maxGap || time===a.time) return {...a,interpolated:false};
  const ratio=(time-a.time)/(b.time-a.time);
  let longitudeDelta=b.lon-a.lon;
  if (longitudeDelta>180) longitudeDelta-=360; if (longitudeDelta < -180) longitudeDelta+=360;
  return {...a,lon:((a.lon+longitudeDelta*ratio+540)%360)-180,lat:a.lat+(b.lat-a.lat)*ratio,
    course:Number.isFinite(a.course)?a.course:Math.atan2(longitudeDelta, b.lat-a.lat)*180/Math.PI,
    interpolated:true,fixTime:a.time};
}

async function createTracks(host, card, options) {
  const base = await baseMap(host, card, options), layer = canvasLayer(base), abort = new AbortController();
  let destroyed=false, payload=card.data, tracks=[], current=NaN, start=NaN, end=NaN, frame=0, last=0, playing=false, visible=true, heads=[];
  let satellite=null, satelliteDay='', satelliteTimer=0, stormsVisible=true, rate=1;
  const loading = el('span','explore-badge explore-loading','Loading measured tracks…'); base.top.replaceChildren(loading);
  const sourceIndices = new Map();
  const idIndex = fieldIndex(card.data.columns,'vessel_id','mmsi','storm_id','sid');
  card.data.rows.forEach((row,index)=>{if(idIndex>=0) sourceIndices.set(String(row[idIndex]),index);});
  try {
    if (options.fetchPlayback) payload = await options.fetchPlayback(card,{signal:abort.signal});
  } catch (error) {
    base.status.textContent = error.message || 'Playback data could not be loaded'; base.status.classList.add('is-visible');
    loading.textContent = 'Tracks unavailable';
    return {...base,ready:Promise.resolve(),points:()=>[],destroy(){destroyed=true;abort.abort();base.destroy();}};
  }
  if(!host.isConnected){destroyed=true;abort.abort();base.destroy();return{ready:Promise.resolve(),points:()=>[],destroy(){}};}
  const samples = normalizeData(payload, card, options.fetchPlayback?sourceIndices:null);
  const stormIndices = card.data.columns.includes('storm_id') ? sourceIndices : new Map();
  const stormSamples = normalizeData(payload.storm, {x:'longitude',y:'latitude'}, stormIndices, true);
  tracks=groupTracks([...samples,...stormSamples]);
  const times=tracks.flatMap(track=>track.points.map(point=>point.time));
  if (!times.length) {
    loading.textContent='No measured fixes'; base.status.textContent='This result has no timestamped track positions.'; base.status.classList.add('is-visible');
    return {...base,ready:Promise.resolve(),points:()=>[],destroy(){destroyed=true;abort.abort();base.destroy();}};
  }
  start=timestamp(payload.range?.start); end=timestamp(payload.range?.end);
  if (!Number.isFinite(start)) start=Math.min(...times); if(!Number.isFinite(end)) end=Math.max(...times);
  if (end<start) [start,end]=[end,start];
  // Arrive with context already visible. Playback stays paused, and the UTC
  // slider can return to the first report or follow another linked voyage.
  current=start+(end-start)/2;
  const span=Math.max(1,end-start), sampleInterval=number(payload.sampling?.interval_seconds)*1000 || 60000;
  const vesselGap=clamp(sampleInterval*2,20*60000,6*3600000), stormGap=8*3600000;
  const tail=clamp(Math.ceil(span*.18/3600000),3,24)*3600000;
  const vessels=tracks.filter(track=>!track.storm), storms=tracks.filter(track=>track.storm);
  base.top.replaceChildren(base.badge);
  base.badge.textContent=`${vessels.length} vessel${vessels.length===1?'':'s'} · ${compact(samples.length)} fixes`;
  base.badge.title=payload.truncated?'Bounded sample: additional fixes or vessels may exist outside this returned result.':'Playback of measured positions returned for this query.';
  if (!vessels.length) base.badge.textContent=`${storms.length} storm${storms.length===1?'':'s'} · ${compact(stormSamples.length || samples.length)} fixes`;
  if (payload.truncated) base.badge.textContent+=' · sampled';
  base.map.attributionControl.addAttribution('NOAA MarineCadastre · NOAA IBTrACS');
  base.setBounds([...samples,...stormSamples]);
  const layerControls=el('div','explore-layer-controls');
  if(storms.length){const toggle=button('Storms','Show or hide hurricane tracks',()=>{stormsVisible=!stormsVisible;toggle.setAttribute('aria-pressed',String(stormsVisible));draw();});toggle.setAttribute('aria-pressed','true');layerControls.append(toggle);}
  const satelliteButton=button('Satellite','Toggle daily NASA Aqua / MODIS imagery',()=>toggleSatellite());satelliteButton.setAttribute('aria-pressed','false');layerControls.append(satelliteButton);base.root.append(layerControls);
  const playback=el('div','explore-playback'), row=el('div','explore-playback-row');
  const play=button('▶','Play measured tracks',()=>setPlaying(!playing),'explore-play');
  play.setAttribute('aria-pressed','false');
  const clock=el('time','explore-clock'), rateSelect=el('select','explore-rate');rateSelect.setAttribute('aria-label','Playback speed');
  for(const value of [.5,1,2,4]){const option=el('option','',value+'×');option.value=value;rateSelect.append(option);}rateSelect.value='1';rateSelect.onchange=()=>rate=Number(rateSelect.value);
  row.append(play,clock,rateSelect);
  const slider=el('input','explore-time-slider');slider.type='range';slider.min='0';slider.max='10000';slider.step='1';slider.value='0';slider.setAttribute('aria-label','Track time in UTC');
  slider.addEventListener('input',()=>{setPlaying(false);setTime(start+span*Number(slider.value)/10000);});
  const legend=el('div','explore-track-legend');if(vessels.length)addLegend(legend,'Vessel fixes',palette[0]);if(storms.length)addLegend(legend,'Nearby storm tracks','#ffb183');
  const interpolation=el('span','explore-track-note',`Short gaps interpolated · ${durationLabel(tail)} trails`);
  interpolation.title=`Measured fixes sampled in ${durationLabel(sampleInterval)} bins. Vessel gaps over ${durationLabel(vesselGap)} are never joined. Storm rings show intensity, not wind-field size. Daily imagery is not live.`;
  const range=el('div','explore-time-range');range.append(el('span','',utcDay(start)),el('span','',utcDay(end)));
  playback.append(row,slider,range,legend,interpolation);base.root.append(playback);
  window.L.DomEvent.disableClickPropagation(playback);window.L.DomEvent.disableScrollPropagation(playback);
  window.L.DomEvent.disableClickPropagation(layerControls);
  const satelliteNote=el('div','explore-satellite-note');satelliteNote.hidden=true;base.root.append(satelliteNote);
  function imageURL(day){return `https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Aqua_CorrectedReflectance_TrueColor/default/${day}/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg`;}
  function updateSatellite(force=false){
    clearTimeout(satelliteTimer);if(!satellite)return;const day=utcDay(current);
    satelliteNote.textContent=`Aqua / MODIS · ${day} · daily image`;satelliteNote.hidden=false;
    if(day===satelliteDay&&!force)return;
    // Debounce scrubbing so one gesture doesn't fetch a full month of satellite tiles.
    satelliteTimer=setTimeout(()=>{if(!destroyed&&satellite){satelliteDay=day;satellite.setUrl(imageURL(day));}},playing?600:180);
  }
  function toggleSatellite(){
    if(satellite){base.map.removeLayer(satellite);satellite=null;clearTimeout(satelliteTimer);satelliteNote.hidden=true;base.countries?.setStyle({fillOpacity:1});}
    else{
      satelliteDay=utcDay(current);satellite=window.L.tileLayer(imageURL(satelliteDay),{maxNativeZoom:9,maxZoom:12,noWrap:true,opacity:.72,
        attribution:'<a href="https://earthdata.nasa.gov/eosdis/science-system-description/eosdis-standard-products/global-imagery-browse-services-gibs" target="_blank" rel="noopener">NASA GIBS / Aqua MODIS</a>',crossOrigin:true}).addTo(base.map);
      base.countries?.setStyle({fillOpacity:0});
      satellite.on('tileerror',()=>{satelliteNote.textContent=`Imagery unavailable for ${satelliteDay} · query tracks remain visible`;});
      updateSatellite();
    }
    satelliteButton.setAttribute('aria-pressed',String(!!satellite));
  }
  function draw(){
    if(destroyed)return;const{c,width,height}=layer.start();heads=[];
    for(const track of tracks){
      if(track.storm&&!stormsVisible)continue;
      const gap=track.storm?stormGap:vesselGap, points=track.points;
      c.lineCap='round';c.lineJoin='round';
      for(let index=1;index<points.length;index++){
        const a=points[index-1],b=points[index];if(b.time-a.time>gap||Math.abs(b.lon-a.lon)>180)continue;
        const p=base.project(a),q=base.project(b);if((p.x<0&&q.x<0)||(p.y<0&&q.y<0)||(p.x>width&&q.x>width)||(p.y>height&&q.y>height))continue;
        const past=b.time<=current,recent=past&&b.time>=current-tail;
        c.strokeStyle=track.color;c.globalAlpha=track.storm?(past?.75:.24):(recent?.55:.11);c.lineWidth=track.storm?2:recent?1.8:.8;
        c.setLineDash(track.storm&&!past?[4,5]:[]);c.beginPath();c.moveTo(p.x,p.y);c.lineTo(q.x,q.y);c.stroke();
      }
      c.setLineDash([]);
      const head=headAt(track,current,gap);if(!head)continue;const p=base.project(head);if(p.x < -50 || p.y < -50 || p.x > width+50 || p.y > height+50)continue;
      if(head.interpolated){const prior=points.findLast(point=>point.time<=current),a=base.project(prior);c.globalAlpha=.7;c.strokeStyle=track.color;c.lineWidth=track.storm?2:1.8;c.beginPath();c.moveTo(a.x,a.y);c.lineTo(p.x,p.y);c.stroke();}
      c.globalAlpha=1;heads.push({...head,x:p.x,y:p.y,color:track.color});
      if(track.storm){
        const radius=12+clamp(head.category||0,0,5)*2;c.strokeStyle='#ffd1ac';c.lineWidth=1.2;c.fillStyle='#ff9e7318';
        c.beginPath();c.arc(p.x,p.y,radius,0,Math.PI*2);c.fill();c.stroke();c.strokeStyle='#ffb88766';c.beginPath();c.arc(p.x,p.y,radius+7,0,Math.PI*2);c.stroke();
        c.fillStyle='#ffdfb7';c.beginPath();c.arc(p.x,p.y,4,0,Math.PI*2);c.fill();
        c.font='600 10px system-ui';c.textAlign='left';const label=head.name+(head.category>=1?` · C${head.category}`:'');
        const tw=c.measureText(label).width;c.fillStyle='#0b1a24dd';c.fillRect(p.x+radius+5,p.y-9,tw+9,18);c.fillStyle='#ffd3a9';c.fillText(label,p.x+radius+9,p.y+4);
      }else{
        c.fillStyle=track.color;c.strokeStyle='#d9fff3';c.lineWidth=.65;c.save();c.translate(p.x,p.y);c.rotate((Number.isFinite(head.course)?head.course:0)*Math.PI/180);
        c.beginPath();c.moveTo(0,-6);c.lineTo(3.8,4.5);c.lineTo(0,2.3);c.lineTo(-3.8,4.5);c.closePath();c.fill();c.stroke();c.restore();
        c.globalAlpha=.15;c.fillStyle=track.color;c.beginPath();c.arc(p.x,p.y,11,0,Math.PI*2);c.fill();c.globalAlpha=1;
      }
    }
    c.globalAlpha=1;c.setLineDash([]);
    clock.textContent=utcLabel(current);clock.dateTime=new Date(current).toISOString();slider.value=String(Math.round((current-start)/span*10000));slider.setAttribute('aria-valuetext',utcLabel(current));
    const active=heads.filter(head=>!head.storm).length;playback.dataset.activeVessels=String(active);playback.dataset.time=String(current);
  }
  function setTime(value,{silent=false}={}){
    const parsed=typeof value==='number'?value:timestamp(value);if(!Number.isFinite(parsed)||destroyed)return;
    current=clamp(parsed,start,end);draw();updateSatellite();host.dispatchEvent(new Event('projectionchange'));
    if(!silent)options.onTimeChange?.(current);
  }
  function tick(now){
    frame=0;if(!playing||destroyed)return;
    if(visible&&!document.hidden&&flowing()){
      const elapsed=Math.min(64,now-(last||now));
      if(last&&elapsed<32){frame=requestAnimationFrame(tick);return;}
      setTime(current+elapsed*(span/30000)*rate);
      if(current>=end){setPlaying(false);return;}
    }
    last=now;frame=requestAnimationFrame(tick);
  }
  function setPlaying(value){
    playing=!!value&&flowing()&&end>start&&!destroyed;cancelAnimationFrame(frame);frame=0;last=0;
    play.textContent=playing?'Ⅱ':'▶';play.setAttribute('aria-label',playing?'Pause measured tracks':'Play measured tracks');play.setAttribute('aria-pressed',String(playing));
    if(playing){if(current>=end)setTime(start);if(visible&&!document.hidden)frame=requestAnimationFrame(tick);}
  }
  const resumeVisibility=()=>{cancelAnimationFrame(frame);frame=0;last=0;if(playing&&visible&&!document.hidden)frame=requestAnimationFrame(tick);};
  const visibility=new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;resumeVisibility();});visibility.observe(base.root);
  document.addEventListener('visibilitychange',resumeVisibility);
  const motionChange=()=>{if(!flowing())setPlaying(false);play.title=flowing()?'Play measured tracks':'Enable flowing motion to animate; the time slider remains available';};window.addEventListener('canvasmotionchange',motionChange);motionChange();
  const nearest=e=>heads.map(point=>({point,d:Math.hypot(point.x-e.containerPoint.x,point.y-e.containerPoint.y)})).sort((a,b)=>a.d-b.d)[0];
  base.map.on('mousemove',event=>{
    const hit=nearest(event);if(!hit||hit.d>25){base.tooltip.hidden=true;return;}
    const p=hit.point,detail=p.storm?`${Number.isFinite(p.wind)?formatNumber(p.wind)+' kt sustained wind':'Wind unavailable'}${p.category>=1?' · Category '+p.category:''}`:`${Number.isFinite(p.speed)?formatNumber(p.speed)+' kt':'Speed unavailable'}${p.type?' · '+p.type:''}`;
    const source=p.storm?'IBTrACS storm position':'Reported AIS position';
    base.showTip(p,[p.name,detail,p.interpolated?'Interpolated between source positions':source,`Last fix ${utcLabel(p.fixTime||p.time)}`]);
  });
  base.map.on('click',event=>{const hit=nearest(event);if(hit?.d<25&&hit.point.index>=0)options.onSelectRows?.([hit.point.index]);});
  base.setDraw(draw);draw();await localReady();
  return {...base,ready:Promise.resolve(),setTime,getTime:()=>current,setPlaying,isPlaying:()=>playing,
    points:()=>heads.filter(point=>point.index>=0).map(point=>({index:point.index,x:point.x+base.root.offsetLeft,y:point.y+base.root.offsetTop})),
    stats:()=>({vessels:vessels.length,storms:storms.length,fixes:samples.length+stormSamples.length,active:heads.length,sampling:payload.sampling,truncated:!!payload.truncated}),
    destroy(){destroyed=true;abort.abort();setPlaying(false);clearTimeout(satelliteTimer);visibility.disconnect();document.removeEventListener('visibilitychange',resumeVisibility);window.removeEventListener('canvasmotionchange',motionChange);base.destroy();}
  };
}

const svgNS='http://www.w3.org/2000/svg';
function svgElement(tag,attrs={}){const node=document.createElementNS(svgNS,tag);for(const[key,value]of Object.entries(attrs))node.setAttribute(key,String(value));return node;}
function monthNumber(value){
  const n=number(value);if(Number.isInteger(n)&&n>=1&&n<=12)return n;
  if(typeof value==='string'){
    const index=monthNames.findIndex(month=>month.toLowerCase()===value.slice(0,3).toLowerCase());if(index>=0)return index+1;
    const time=timestamp(value);if(Number.isFinite(time))return new Date(time).getUTCMonth()+1;
  }
  return NaN;
}
function sector(inner,outer,start,end){
  const p=(radius,angle)=>[180+radius*Math.cos(angle),180+radius*Math.sin(angle)];
  const a=p(inner,start),b=p(outer,start),c=p(outer,end),d=p(inner,end);
  return `M${a.join(',')} L${b.join(',')} A${outer},${outer} 0 0 1 ${c.join(',')} L${d.join(',')} A${inner},${inner} 0 0 0 ${a.join(',')} Z`;
}

function createSeasons(host,card,options){
  const root=el('div','explore-viz explore-seasons'), plot=el('div','explore-season-plot'), detail=el('div','explore-season-detail');root.append(plot,detail);host.append(root);
  const {columns,rows}=card.data,xi=columns.indexOf(card.x),yi=columns.indexOf(card.y),zi=columns.indexOf(card.z);
  const valueIndex=zi>=0?zi:yi,groupIndex=zi>=0&&yi>=0&&yi!==valueIndex?yi:-1;
  const groups=new Map();
  rows.forEach((row,index)=>{
    const month=monthNumber(row[xi]),value=number(row[valueIndex]);if(!Number.isFinite(month)||!Number.isFinite(value)||value<0)return;
    const group=String(groupIndex>=0?row[groupIndex]??'Unspecified':'All observations');if(!groups.has(group))groups.set(group,new Map());
    const months=groups.get(group),prior=months.get(month);if(prior){prior.value+=value;prior.indices.push(index);}else months.set(month,{month,value,indices:[index]});
  });
  const names=[...groups.keys()],select=el('select','explore-season-select');select.setAttribute('aria-label','Seasonal group');
  names.forEach(name=>{const option=el('option','',name);option.value=name;select.append(option);});
  if(names.length>1)detail.append(select);
  const label=el('span','explore-season-label','Returned observations'),total=el('strong','explore-season-total'),peak=el('div','explore-season-peak'),note=el('p','explore-season-note');
  const metric=(columns[valueIndex]||'value').replaceAll('_',' ');label.textContent=metric;detail.append(label,total,peak,note);
  const svg=svgElement('svg',{viewBox:'0 0 360 360',role:'img','aria-label':card.title+' — monthly cycle'});plot.append(svg);
  let screenPoints=[],destroyed=false;
  function draw(){
    if(destroyed)return;svg.replaceChildren();screenPoints=[];
    const values=groups.get(select.value||names[0])||new Map(),actual=[...values.values()],max=Math.max(0,...actual.map(item=>item.value)),sum=actual.reduce((acc,item)=>acc+item.value,0);
    total.textContent=compact(sum);total.title=formatNumber(sum)+' '+metric;
    const peakMonth=actual.reduce((best,item)=>!best||item.value>best.value?item:best,null);
    peak.replaceChildren(el('span','',peakMonth?`${monthNames[peakMonth.month-1]} peaks`:'No monthly values'),el('strong','',peakMonth?formatNumber(peakMonth.value):'—'));
    note.textContent=`${actual.length} of 12 months returned. Missing months stay empty.`+(card.dataset_id==='maritime'?'':' Counts reflect observation effort, not population size.');
    for(const radius of[60,90,120,147])svg.append(svgElement('circle',{cx:180,cy:180,r:radius,fill:'none',stroke:'#2b454f','stroke-width':.7}));
    for(let month=1;month<=12;month++){
      const angle=(month-1)/12*Math.PI*2-Math.PI/2,opening=Math.PI*2/12-.055,value=values.get(month),inner=54;
      // Annular sector area is proportional to the returned value.
      const outer=value?Math.sqrt(inner*inner+(value.value/(max||1))*(145*145-inner*inner)):inner+3;
      const path=svgElement('path',{d:sector(inner,Math.max(inner+2,outer),angle-opening/2,angle+opening/2),fill:value?`hsl(${180-(value.value/(max||1))*24} 63% ${32+(value.value/(max||1))*34}%)`:'#183039',stroke:value?'#a1f9e744':'#38505a','stroke-width':.6});
      if(!value)path.setAttribute('stroke-dasharray','2 2');
      const title=svgElement('title');title.textContent=value?`${monthNames[month-1]}: ${formatNumber(value.value)} ${metric}`:`${monthNames[month-1]}: no returned value`;path.append(title);
      if(value){
        path.classList.add('explore-season-sector');path.setAttribute('tabindex','0');path.setAttribute('role','button');path.setAttribute('aria-label',title.textContent);
        const choose=()=>options.onSelectRows?.(value.indices);path.addEventListener('click',choose);path.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();choose();}});
        const center=inner+(outer-inner)/2;value.indices.forEach(index=>screenPoints.push({index,x:180+center*Math.cos(angle),y:180+center*Math.sin(angle)}));
      }
      svg.append(path);
      const text=svgElement('text',{x:180+164*Math.cos(angle),y:180+164*Math.sin(angle)+3,'text-anchor':'middle',fill:value?'#9cc6cd':'#506d78','font-size':10});text.textContent=monthNames[month-1];svg.append(text);
    }
    const center=svgElement('text',{x:180,y:179,'text-anchor':'middle',fill:'#bcdfda','font-size':15,'font-weight':550});center.textContent='12';
    const caption=svgElement('text',{x:180,y:195,'text-anchor':'middle',fill:'#72929d','font-size':8,'letter-spacing':1.2});caption.textContent='MONTHS';svg.append(center,caption);
    host.dispatchEvent(new Event('projectionchange'));
  }
  select.addEventListener('change',draw);draw();
  const resized=()=>host.dispatchEvent(new Event('projectionchange')),observer=new ResizeObserver(resized);observer.observe(plot);
  return {ready:localReady(),resize:resized,points:()=>{
    const rect=svg.getBoundingClientRect(),hostRect=host.getBoundingClientRect(),sx=host.clientWidth/hostRect.width,sy=host.clientHeight/hostRect.height;
    const size=Math.min(rect.width,rect.height),left=rect.left+(rect.width-size)/2,top=rect.top+(rect.height-size)/2;
    return screenPoints.map(point=>({index:point.index,x:(left-hostRect.left+point.x/360*size)*sx,y:(top-hostRect.top+point.y/360*size)*sy}));
  },destroy(){destroyed=true;observer.disconnect();root.remove();}};
}

/**
 * Contracts: density/map x=longitude,y=latitude,z=value; seasons x=month,
 * y=value or y=group,z=value; tracks accepts measured fixes or fetchPlayback(card).
 * fetchPlayback resolves {columns,rows,storm:{columns,rows},range,sampling,truncated}.
 * Call destroy before replacing a renderer. onTimeChange receives epoch milliseconds.
 */
export async function renderExplore(host,card,options={}){
  if(card.chart==='tracks')return createTracks(host,card,options);
  if(card.chart==='seasons')return createSeasons(host,card,options);
  if(card.chart==='density'||card.chart==='map')return createDensity(host,card,options);
  throw Error(`Unsupported exploration view: ${card.chart}`);
}

// Pure helpers are exported so regression checks can exercise UTC and gap semantics.
export const exploreData={timestamp,monthNumber,headAt,groupTracks,gridStep,normalizeData};
