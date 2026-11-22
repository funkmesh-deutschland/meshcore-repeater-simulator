/* Funkmesh Repeater-Simulator – Oberfläche */
'use strict';
const $ = (s) => document.querySelector(s);
const api = async (url, opt = {}) => {
  const r = await fetch(url, { headers: { 'Content-Type': 'application/json' }, ...opt,
    body: opt.body && typeof opt.body !== 'string' ? JSON.stringify(opt.body) : opt.body });
  if (!r.ok) { let t = await r.text(); try { t = JSON.parse(t).detail || t; } catch (e) {} throw new Error(t); }
  return r.json();
};
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v)) ? '–' : Number(v).toFixed(d).replace('.', ',');
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

let S = null;            // Serverzustand
let selected = null;     // ausgewählte Repeater-ID
let placing = false;
let profileMode = false;
let resultVersion = 0;
let loadVersion = 0;
let curLayer = 'margin';
const markers = {};
let linkLayer = null;

// ------------------------------------------------------------------ Karte --
const map = L.map('map', { zoomControl: true, preferCanvas: true });
const base = {
  'basemap.de (Farbe)': L.tileLayer('https://sgx.geodatenzentrum.de/wmts_basemapde/tile/1.0.0/de_basemapde_web_raster_farbe/default/GLOBAL_WEBMERCATOR/{z}/{y}/{x}.png',
    { maxZoom: 19, maxNativeZoom: 18, attribution: '© <a href="https://basemap.de">basemap.de</a> / BKG' }),
  'basemap.de (Grau)': L.tileLayer('https://sgx.geodatenzentrum.de/wmts_basemapde/tile/1.0.0/de_basemapde_web_raster_grau/default/GLOBAL_WEBMERCATOR/{z}/{y}/{x}.png',
    { maxZoom: 19, maxNativeZoom: 18, attribution: '© <a href="https://basemap.de">basemap.de</a> / BKG' }),
  'OpenStreetMap': L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png',
    { maxZoom: 19, attribution: '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>-Mitwirkende' }),
  'Luftbild NRW (DOP)': L.tileLayer.wms('https://www.wms.nrw.de/geobasis/wms_nw_dop', { layers: 'nw_dop_rgb', format: 'image/jpeg',
    maxZoom: 20, attribution: '© Geobasis NRW (dl-de/zero-2-0)' }),
};
base['basemap.de (Grau)'].addTo(map);
L.control.layers(base, {}, { position: 'topright' }).addTo(map);
L.control.scale({ imperial: false }).addTo(map);
const resultLayer = L.tileLayer('', { opacity: 0.8, maxZoom: 20, minZoom: 5, zIndex: 400 }).addTo(map);

const LAYERS = [
  ['margin', 'Linkreserve'], ['prob', 'Versorgungswahrsch.'], ['rssi', 'Empfangspegel'], ['count', 'Redundanz'],
  ['best', 'Bester Repeater'], ['load', 'Kanalbelegung'], ['terrain', 'Gelände'], ['buildings', 'Gebäude'],
  ['vegetation', 'Bewuchs'], ['none', 'aus'],
];

function setLayer(name) {
  curLayer = name;
  if (name === 'none') { resultLayer.setUrl(''); resultLayer.remove(); }
  else {
    if (!map.hasLayer(resultLayer)) resultLayer.addTo(map);
    const v = name === 'load' ? `${resultVersion}${loadVersion}` : resultVersion;
    resultLayer.setUrl(`/tiles/${name}/{z}/{x}/{y}.png?v=${v}&a=${encodeURIComponent(S?.region?.name || '')}`);
  }
  renderLegend();
}

function renderLegend() {
  const el = $('#legend');
  const grad = (stops, ticks) => `<div class="bar" style="background:linear-gradient(90deg,${stops})"></div><div class="ticks">${ticks.map((t) => `<span>${t}</span>`).join('')}</div>`;
  const items = (arr) => arr.map(([c, t]) => `<div class="item"><span class="sw" style="background:${c}"></span>${t}</div>`).join('');
  const sens = S?.derived?.sens_dbm;
  const L = {
    margin: `<b>Linkreserve</b> min(Downlink, Uplink) über Empfindlichkeit (${fmt(sens)} dBm)` + items([
      ['#0f766e', '≥ 20 dB – sehr gut (auch Innenräume)'], ['#5eead4', '10 … 20 dB – gut'], ['#facc15', '5 … 10 dB – ausreichend'],
      ['#fc790d', '0 … 5 dB – grenzwertig'], ['#dc2626', '−10 … 0 dB – nur zeitweise']]),
    prob: `<b>Wahrscheinlichkeit</b>, mindestens einen Repeater zu erreichen (Ortsvariabilität)` + grad('#dc2626,#fc790d,#facc15,#14b8a6,#0f766e', ['50 %', '70 %', '85 %', '90 %', '100 %']),
    rssi: `<b>Empfangspegel</b> des besten Repeaters am Endgerät` + grad('#440154,#3b528b,#21918c,#5ec962,#fde725', ['−135', '−120', '−105', '−90', '−75 dBm']),
    count: `<b>Redundanz</b>: Repeater mit Ziel-Ortswahrscheinlichkeit` + items([['#fc790d', '1'], ['#facc15', '2'], ['#14b8a6', '3'], ['#0f766e', '4'], ['#084642', '≥ 5']]),
    best: `<b>Bester Repeater</b> je Ort (Farbe je Repeater)`,
    load: `<b>Kanalbelegung</b>: Summe der Sendezeitanteile hörbarer Repeater (aus Auslastungssimulation)` + grad('#14b8a6,#facc15,#fc790d,#b91c1c', ['0 %', '10 %', '20 %', '≥ 30 %']),
    terrain: `<b>Gelände</b> (${esc(S?.region?.sources?.terrain || '')}, Schummerung)`,
    buildings: `<b>Gebäudehöhe</b> (${esc(S?.region?.sources?.buildings || '')})` + grad('#b4aac8,#3c1e6e', ['0 m', '25 m']),
    vegetation: `<b>Bewuchshöhe</b> (${esc(S?.region?.sources?.vegetation || '')})` + grad('#aadc8c,#145a28', ['0 m', '30 m']),
    none: '',
  };
  el.innerHTML = L[curLayer] || '';
}

// --------------------------------------------------------------- Repeater --
function repIcon(r) {
  const cls = ['rep-icon', r.id === selected ? 'sel' : '', r.enabled ? '' : 'off'].join(' ');
  return L.divIcon({ className: cls, html: '<div>R</div>', iconSize: [28, 28], iconAnchor: [14, 14] });
}

function drawRepeaters() {
  const reps = S.project?.repeaters || [];
  for (const id of Object.keys(markers)) if (!reps.find((r) => r.id === id)) { markers[id].remove(); delete markers[id]; }
  for (const r of reps) {
    let m = markers[r.id];
    if (!m) {
      m = L.marker([r.lat, r.lon], { draggable: true, icon: repIcon(r), zIndexOffset: 1000 }).addTo(map);
      m.on('click', (e) => { L.DomEvent.stopPropagation(e); select(r.id); });
      m.on('dragend', async () => {
        const ll = m.getLatLng();
        await api(`/api/repeaters/${m._rid}`, { method: 'PUT', body: { lat: ll.lat, lon: ll.lng } });
        await refresh(); select(m._rid);
      });
      markers[r.id] = m;
    }
    m._rid = r.id;
    m.setLatLng([r.lat, r.lon]);
    m.setIcon(repIcon(r));
    m.bindTooltip(`${esc(r.name)} · ${fmt(r.height_m)} m`, { direction: 'top', offset: [0, -12] });
  }
  $('#rep-count').textContent = reps.length;
  for (const id of [...checked]) if (!reps.find((r) => r.id === id)) checked.delete(id);
  const q = $('#rep-filter').value.trim().toLowerCase();
  const shown = q ? reps.filter((r) => r.name.toLowerCase().includes(q)) : reps;
  $('#rep-list').innerHTML = shown.slice(0, 1500).map((r) => `<div class="rep chk ${r.id === selected ? 'sel' : ''} ${r.enabled ? '' : 'off'}" data-id="${r.id}">
      <input type="checkbox" data-c="${r.id}" ${checked.has(r.id) ? 'checked' : ''}>
      <span class="dot"></span><div><div class="nm">${esc(r.name)}</div>
      <div class="meta">${fmt(r.height_m)} m · ${fmt(r.tx_dbm)} dBm · ${fmt(r.gain_dbi)} dBi${r.source && r.source !== 'manuell' ? ' · ' + esc(r.source) : ''}</div></div>
      <span class="meta">${r.enabled ? '' : 'inaktiv'}</span></div>`).join('') + (shown.length > 1500 ? `<div class="hint">… ${shown.length - 1500} weitere (Filter nutzen)</div>` : '')
    || (reps.length ? '<div class="hint">Kein Treffer.</div>' : '<div class="hint">Noch keine Repeater. Setze einen oder importiere die Repeater von map.meshcore.io / mapme.sh.</div>');
  document.querySelectorAll('#rep-list .rep').forEach((el) => el.addEventListener('click', (ev) => {
    if (ev.target.matches('input')) return;
    select(el.dataset.id); const r = reps.find((x) => x.id === el.dataset.id); map.panTo([r.lat, r.lon]);
  }));
  document.querySelectorAll('#rep-list input[data-c]').forEach((el) => el.addEventListener('change', () => {
    if (el.checked) checked.add(el.dataset.c); else checked.delete(el.dataset.c); updBulk();
  }));
  updBulk();
}

// ------------------------------------------------- Mehrfachauswahl (Bulk) --
const checked = new Set();
function updBulk() {
  $('#bulk-act').hidden = checked.size === 0;
  $('#sel-count').textContent = `${checked.size} markiert`;
  for (const [id, m] of Object.entries(markers)) m.getElement()?.classList.toggle('chk', checked.has(id));
}
function setChecked(ids) { checked.clear(); ids.forEach((i) => checked.add(i)); drawRepeaters(); }
$('#sel-all').onclick = () => { const q = $('#rep-filter').value.trim().toLowerCase(); setChecked(S.project.repeaters.filter((r) => !q || r.name.toLowerCase().includes(q)).map((r) => r.id)); };
$('#sel-none').onclick = () => setChecked([]);
$('#sel-off').onclick = () => setChecked(S.project.repeaters.filter((r) => !r.enabled).map((r) => r.id));
$('#sel-view').onclick = () => { const b = map.getBounds(); setChecked(S.project.repeaters.filter((r) => b.contains([r.lat, r.lon])).map((r) => r.id)); };
$('#rep-filter').oninput = () => drawRepeaters();
async function bulk(action, extra = {}) {
  const ids = [...checked];
  if (!ids.length) return;
  const r = await api('/api/repeaters/bulk', { method: 'POST', body: { ids, action, ...extra } });
  if (action === 'delete') { checked.clear(); if (ids.includes(selected)) { selected = null; $('#editor').hidden = true; } }
  await refresh();
  return r;
}
$('#bulk-del').onclick = () => { if (confirm(`${checked.size} Repeater löschen?`)) bulk('delete'); };
$('#bulk-on').onclick = () => bulk('enable');
$('#bulk-off').onclick = () => bulk('disable');
$('#bulk-h').onclick = () => { const h = prompt(`Antennenhöhe über Grund [m] für ${checked.size} Repeater:`, '10'); if (h !== null && +h > 0) bulk('height', { height_m: +h }); };

async function select(id) {
  selected = id;
  const r = S.project.repeaters.find((x) => x.id === id);
  drawRepeaters();
  if (!r) { $('#editor').hidden = true; return; }
  $('#editor').hidden = false;
  $('#ed-name').value = r.name; $('#ed-height').value = r.height_m; $('#ed-tx').value = r.tx_dbm;
  $('#ed-gain').value = r.gain_dbi; $('#ed-cable').value = r.cable_db; $('#ed-tilt').value = r.tilt_deg; $('#ed-en').checked = r.enabled;
  updEirp();
  $('#ed-site').textContent = 'Standort wird geladen …';
  try {
    const p = await api(`/api/point?lat=${r.lat}&lon=${r.lon}`);
    $('#ed-site').innerHTML = p.inside ? `Gelände ${fmt(p.terrain)} m ü. NHN · Gebäude ${p.building > 0 ? fmt(p.building) + ' m' : '–'} · Bewuchs ${p.vegetation > 0 ? fmt(p.vegetation) + ' m' : '–'}`
      + (p.building > 0 && r.height_m < p.building ? ' · <span class="warn">Antenne unter Dachhöhe (Innenraum)</span>' : '') : '<span class="bad">außerhalb der Geodaten</span>';
  } catch (e) { $('#ed-site').textContent = ''; }
}

function updEirp() {
  const eirp = +$('#ed-tx').value + +$('#ed-gain').value - +$('#ed-cable').value;
  const erp = eirp - 2.15;
  $('#ed-eirp').innerHTML = `EIRP ${fmt(eirp)} dBm · ERP ${fmt(erp)} dBm` + (erp > 27.01 ? ' · <span class="bad">über 500 mW ERP (EU 869,4–869,65 MHz)</span>' : '');
}
['#ed-tx', '#ed-gain', '#ed-cable'].forEach((s) => $(s).addEventListener('input', updEirp));

$('#ed-save').onclick = async () => {
  await api(`/api/repeaters/${selected}`, { method: 'PUT', body: {
    name: $('#ed-name').value, height_m: +$('#ed-height').value, tx_dbm: +$('#ed-tx').value, gain_dbi: +$('#ed-gain').value,
    cable_db: +$('#ed-cable').value, tilt_deg: +$('#ed-tilt').value, enabled: $('#ed-en').checked } });
  await refresh(); select(selected);
};
$('#ed-del').onclick = async () => {
  const r = S.project.repeaters.find((x) => x.id === selected);
  if (!r || !confirm(`Repeater „${r.name}“ löschen?`)) return;
  await api(`/api/repeaters/${selected}`, { method: 'DELETE' });
  selected = null; $('#editor').hidden = true; await refresh();
};
$('#ed-close').onclick = () => { selected = null; $('#editor').hidden = true; drawRepeaters(); };
$('#ed-profile').onclick = () => {
  profileMode = true; document.body.classList.add('placing');
  $('#ed-profile').classList.add('on'); $('#ed-profile').textContent = 'Zielpunkt in Karte klicken …';
};

$('#btn-place').onclick = () => setPlacing(!placing);
function setPlacing(on) {
  placing = on; document.body.classList.toggle('placing', on);
  $('#btn-place').classList.toggle('on', on); $('#place-hint').hidden = !on;
}
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { setPlacing(false); endProfileMode(); } });
function endProfileMode() {
  profileMode = false; document.body.classList.remove('placing');
  $('#ed-profile').classList.remove('on'); $('#ed-profile').textContent = 'Höhenprofil …';
}

let justDrew = 0;
map.on('click', async (e) => {
  if (Date.now() - justDrew < 400 || drawing) return;
  const { lat, lng } = e.latlng;
  if (placing) {
    setPlacing(false);
    let h = 10;
    try { const p = await api(`/api/point?lat=${lat}&lon=${lng}`); if (p.building > 0) h = Math.round((p.building + 2) * 2) / 2; } catch (err) {}
    const r = await api('/api/repeaters', { method: 'POST', body: { lat, lon: lng, height_m: h } });
    await refresh(); select(r.id); return;
  }
  if (profileMode && selected) { endProfileMode(); showProfile(selected, lat, lng); return; }
  pointPopup(e.latlng);
});

async function pointPopup(ll) {
  const pop = L.popup({ maxWidth: 380 }).setLatLng(ll).setContent('…').openOn(map);
  try {
    const p = await api(`/api/point?lat=${ll.lat}&lon=${ll.lng}`);
    if (!p.inside) { pop.setContent('Außerhalb der Geodaten'); return; }
    let h = `<div class="popup"><h4>${fmt(ll.lat, 5)}, ${fmt(ll.lng, 5)}</h4><table>
      <tr><td>Gelände</td><td>${fmt(p.terrain)} m ü. NHN</td></tr>
      <tr><td>Gebäude</td><td>${p.building > 0 ? fmt(p.building) + ' m (Innenraum)' : '–'}</td></tr>
      <tr><td>Bewuchs</td><td>${p.vegetation > 0 ? fmt(p.vegetation) + ' m' : '–'}</td></tr>`;
    if (p.rssi !== undefined) {
      h += `<tr><td>Empfangspegel</td><td><b>${fmt(p.rssi)} dBm</b></td></tr>
        <tr><td>Linkreserve</td><td><b class="${p.margin >= 10 ? 'good' : p.margin >= 0 ? 'warn' : 'bad'}">${fmt(p.margin)} dB</b></td></tr>
        <tr><td>Versorgung</td><td>${p.prob} % · ${p.count} Repeater</td></tr><tr><td>Bester</td><td>${esc(p.best)}</td></tr></table>`;
      if (p.repeaters?.length) h += `<table><tr><th>Repeater</th><th class="num">Verlust</th><th class="num">↓ dBm</th><th class="num">↑ dBm</th><th class="num">Reserve</th></tr>` +
        p.repeaters.map((r) => `<tr><td>${esc(r.name)}</td><td class="num">${fmt(r.loss)}</td><td class="num">${fmt(r.down)}</td><td class="num">${fmt(r.up)}</td><td class="num">${fmt(r.margin)}</td></tr>`).join('') + '</table>';
    } else h += '</table><div class="hint">Noch keine Berechnung.</div>';
    pop.setContent(h + '</div>');
  } catch (err) { pop.setContent('Fehler: ' + esc(err.message)); }
}

// ------------------------------------------------------------- Höhenprofil --
async function showProfile(rid, lat, lon) {
  const r = S.project.repeaters.find((x) => x.id === rid);
  const p = await api(`/api/profile?rid=${rid}&lat=${lat}&lon=${lon}`);
  $('#profile').hidden = false;
  $('#prof-title').textContent = `Höhenprofil ${r.name} → Punkt (${fmt(p.dist_m / 1000, 2)} km)`;
  const W = 900, H = 260, n = p.d.length;
  const top = p.terrain.map((t, i) => t + p.building[i] + p.vegetation[i]);
  const ys = [...p.terrain, ...top, ...p.los.map((v, i) => v + p.fresnel[i]), ...p.los.map((v, i) => v - p.fresnel[i])];
  const ymin = Math.min(...p.terrain) - 5, ymax = Math.max(...ys) + 5;
  const X = (i) => (i / (n - 1)) * W, Y = (v) => H - ((v - ymin) / (ymax - ymin)) * H;
  const poly = (arr) => arr.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' ');
  const area = (lo, hi) => `${hi.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`).join(' ')} ${lo.map((v, i) => `${X(n - 1 - i).toFixed(1)},${Y(lo[n - 1 - i]).toFixed(1)}`).join(' ')}`;
  const tb = p.terrain.map((t, i) => t + p.building[i]);
  const tbv = tb.map((t, i) => t + p.vegetation[i]);
  const f60 = p.los.map((v, i) => v - 0.6 * p.fresnel[i]);
  $('#prof-svg').innerHTML = `
    <polygon points="${area(p.terrain.map(() => ymin), p.terrain)}" fill="#c8b89a"/>
    <polygon points="${area(p.terrain, tb)}" fill="#6d4aa0" opacity=".8"/>
    <polygon points="${area(tb, tbv)}" fill="#3f8f4f" opacity=".7"/>
    <polyline points="${poly(p.los.map((v, i) => v + p.fresnel[i]))}" fill="none" stroke="#14b8a6" stroke-width="1"/>
    <polyline points="${poly(p.los.map((v, i) => v - p.fresnel[i]))}" fill="none" stroke="#14b8a6" stroke-width="1"/>
    <polyline points="${poly(f60)}" fill="none" stroke="#14b8a6" stroke-dasharray="4 4" stroke-width="1"/>
    <polyline points="${poly(p.los)}" fill="none" stroke="#fc790d" stroke-width="2"/>
    <text x="6" y="14" font-size="12" fill="#4a6464">${fmt(ymax, 0)} m</text><text x="6" y="${H - 4}" font-size="12" fill="#4a6464">${fmt(ymin, 0)} m</text>`;
}
$('#prof-close').onclick = () => { $('#profile').hidden = true; };

// ------------------------------------------------------------------ Links --
async function drawLinks() {
  if (linkLayer) { linkLayer.remove(); linkLayer = null; }
  if (!$('#show-links').checked || !S.result) return;
  const links = await api('/api/links');
  const byId = Object.fromEntries(S.project.repeaters.map((r) => [r.id, r]));
  linkLayer = L.layerGroup();
  for (const l of links) {
    const a = byId[l.a], b = byId[l.b];
    if (!a || !b || l.margin < 0) continue;
    const col = l.margin >= 10 ? '#0f766e' : l.margin >= 0 ? '#fc790d' : '#dc2626';
    L.polyline([[a.lat, a.lon], [b.lat, b.lon]], { color: col, weight: l.margin >= 10 ? 3 : 2, opacity: 0.85, dashArray: l.margin < 10 ? '5 6' : null })
      .bindTooltip(`${esc(a.name)} ↔ ${esc(b.name)}<br>${fmt(l.dist_m / 1000, 1)} km · Verlust ${fmt(l.loss_db)} dB<br>Pegel ${fmt(l.rssi_ab)} / ${fmt(l.rssi_ba)} dBm · Reserve <b>${fmt(l.margin)} dB</b>`, { sticky: true })
      .addTo(linkLayer);
  }
  linkLayer.addTo(map);
}
$('#show-links').onchange = drawLinks;

// ----------------------------------------------------------------- Import --
let impData = [];
$('#btn-import').onclick = () => { $('#import-box').hidden = false; };
$('#import-close').onclick = () => { $('#import-box').hidden = true; };
function impCount() { $('#imp-count').textContent = `${document.querySelectorAll('#imp-list input:checked').length} ausgewählt`; }
$('#imp-load').onclick = async () => {
  $('#imp-info').textContent = 'lade …'; $('#imp-list').innerHTML = '';
  try {
    const r = await api(`/api/import?source=${$('#imp-src').value}&max_age_days=${$('#imp-age').value}`);
    impData = r.items.sort((a, b) => a.name.localeCompare(b.name));
    $('#imp-list').innerHTML = impData.map((x, i) => `<label class="rep" style="grid-template-columns:18px 1fr auto">
      <input type="checkbox" data-i="${i}" ${!x.known && x.inside_data ? 'checked' : ''} ${x.known || !x.inside_data ? 'disabled' : ''}>
      <div><div class="nm">${esc(x.name)}</div><div class="meta">${fmt(x.lat, 4)}, ${fmt(x.lon, 4)} · vor ${fmt(x.age_days, 0)} Tagen · ${esc(x.sources.join(' + '))}${x.params?.sf ? ` · ${x.params.freq} MHz SF${x.params.sf}` : ''}</div></div>
      <span class="meta">${x.known ? 'vorhanden' : x.inside_data ? '' : 'außerhalb'}</span></label>`).join('');
    document.querySelectorAll('#imp-list input').forEach((el) => el.addEventListener('change', impCount));
    $('#imp-info').textContent = `${impData.length} Repeater gefunden` + (r.errors.length ? ` · ${r.errors.join(' · ')}` : '');
    $('#imp-do').disabled = false; $('#imp-sel').hidden = false; impCount();
  } catch (e) { $('#imp-info').textContent = 'Fehler: ' + e.message; }
};
$('#imp-all').onclick = () => { document.querySelectorAll('#imp-list input:not(:disabled)').forEach((el) => { el.checked = true; }); impCount(); };
$('#imp-none').onclick = () => { document.querySelectorAll('#imp-list input').forEach((el) => { el.checked = false; }); impCount(); };
$('#imp-do').onclick = async () => {
  const items = [...document.querySelectorAll('#imp-list input:checked')].map((el) => impData[+el.dataset.i]);
  const added = await api('/api/import', { method: 'POST', body: { items, height_m: +$('#imp-h').value } });
  $('#imp-info').textContent = `${added.length} übernommen – Antennenhöhen bitte prüfen!`;
  await refresh(); $('#imp-load').click();
};

// ------------------------------------------------------------ Berechnung --
$('#btn-calc').onclick = async () => {
  const n = S.project.repeaters.filter((r) => r.enabled).length;
  const big = S.region.km2 > 20000 || n > 300;
  if (big) {
    const cells = (2 * +$('#c-rad').value * 1000 / +$('#c-res').value) ** 2;
    const per = cells * (S.backend === 'cuda' ? 1.1e-6 : 1.5e-5) + 1.5;
    if (!confirm(`Großes Gebiet: ${n} aktive Repeater auf ${S.region.km2.toLocaleString('de-DE')} km².\n\nGeschätzte Rechenzeit ca. ${dur(per * n)} (bereits berechnete Repeater kommen aus dem Cache), danach werden die Repeater-Verbindungen gerechnet.\nDer Rechner ist in dieser Zeit stark ausgelastet, die Berechnung lässt sich jederzeit abbrechen.\n\nStarten?`)) return;
  }
  await api('/api/settings/compute', { method: 'PUT', body: { out_res: +$('#c-res').value, radius_km: +$('#c-rad').value } });
  try { await api('/api/compute', { method: 'POST' }); } catch (e) { alert(e.message); }
  pollJob();
};
$('#btn-cancel').onclick = () => api('/api/compute/cancel', { method: 'POST' });

const dur = (s) => s < 90 ? `${Math.round(s)} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${fmt(s / 3600, 1)} h`;
async function pollJob() {
  const { job, result_version } = await api('/api/job');
  if (!job) return;
  $('#prog').style.width = `${job.state === 'done' ? 100 : Math.round(job.progress * 100)}%`;
  $('#job-msg').textContent = job.state === 'idle' ? '' : `${job.message} · ${fmt(job.elapsed, 0)} s`;
  $('#job-log').textContent = job.log.join('\n');
  $('#job-log').scrollTop = 1e9;
  const running = job.state === 'running';
  $('#btn-calc').disabled = running; $('#btn-cancel').disabled = !running;
  if (running) { setTimeout(pollJob, 1000); return; }
  if (result_version !== resultVersion) { await refresh(); }
}

function renderSummary() {
  const r = S.result;
  if (!r) { $('#summary-card').hidden = true; return; }
  $('#summary-card').hidden = false;
  $('#summary').innerHTML = `<div class="kpis"><div class="kpi"><b>${r.repeaters.length}</b><span>Repeater</span></div>
    <div class="kpi"><b>${fmt(r.out_res, 0)} m</b><span>Raster</span></div><div class="kpi"><b>${fmt(r.radius_m / 1000, 0)} km</b><span>Radius</span></div></div>
    <div class="hint">Empfindlichkeit ${fmt(r.sens_dbm)} dBm · Ziel-Ortswahrscheinlichkeit ${Math.round(r.target_prob * 100)} % · Stand ${new Date(r.version * 1000).toLocaleString('de-DE')}</div>`;
}

// ------------------------------------------------------------- Auslastung --
const LOADF = [
  ['users', 'Aktive Endgeräte', 1], ['outdoor_share', 'Anteil draußen (0–1)', 0.05], ['chan_msgs_per_user_h', 'Kanalnachr. je Gerät/h', 0.1], ['dm_per_user_h', 'Direktnachr. je Gerät/h', 0.1],
  ['dm_flood_share', 'Anteil DN per Flood', 0.05], ['user_adverts_per_h', 'Adverts je Gerät/h', 0.05], ['repeater_advert_h', 'Repeater-Advert alle … h', 1],
  ['msg_bytes', 'Nachricht [Byte]', 1], ['advert_bytes', 'Advert [Byte]', 1], ['txdelay', 'txdelay-Faktor', 0.1],
  ['airtime_factor', 'Airtime-Faktor (af)', 0.1], ['capture_db', 'Capture-Schwelle [dB]', 0.5], ['hours', 'Simulierte Stunden', 0.5],
];
const LOADDEF = { users: 200, outdoor_share: 0.3, chan_msgs_per_user_h: 2, dm_per_user_h: 1, dm_flood_share: 0.3, user_adverts_per_h: 0.25, repeater_advert_h: 12,
  msg_bytes: 90, advert_bytes: 140, txdelay: 0.5, airtime_factor: 1, capture_db: 6, hours: 1 };
$('#load-form').innerHTML = LOADF.map(([k, t, st]) => `<label>${t}<input type="number" step="${st}" id="lf-${k}" value="${LOADDEF[k]}"></label>`).join('');
$('#btn-load').onclick = async () => {
  const body = Object.fromEntries(LOADF.map(([k]) => [k, +$(`#lf-${k}`).value]));
  $('#load-msg').textContent = 'Simulation läuft …'; $('#btn-load').disabled = true;
  try { const r = await api('/api/load', { method: 'POST', body }); S.load = r; loadVersion++; renderLoad(); $('#load-msg').textContent = ''; if (curLayer === 'load') setLayer('load'); }
  catch (e) { $('#load-msg').textContent = 'Fehler: ' + e.message; }
  $('#btn-load').disabled = false;
};
function renderLoad() {
  const r = S.load;
  if (!r) { $('#load-res').hidden = true; return; }
  $('#load-res').hidden = false;
  const over = r.repeaters.filter((x) => x.over_limit).length;
  $('#load-res').innerHTML = `<h3>Ergebnis</h3><div class="kpis">
    <div class="kpi"><b>${r.packets}</b><span>Pakete (${fmt(r.params.hours, 1)} h)</span></div>
    <div class="kpi"><b>${fmt(r.flood_reach_mean * 100, 0)} %</b><span>Flood-Reichweite Ø</span></div>
    <div class="kpi"><b>${fmt(100 * r.users_covered / Math.max(r.users, 1), 0)} %</b><span>Geräte versorgt (${r.users_covered}/${r.users})</span></div></div>
    <div class="hint"><b>Versorgt</b> = Gerät erreicht mindestens einen Repeater über der Empfindlichkeit (Uplink, Median ohne Schwundreserve).
    Draußen: ${r.covered_outdoor ?? '–'}/${r.users_outdoor ?? '–'} (${fmt(100 * (r.covered_outdoor ?? 0) / Math.max(r.users_outdoor ?? 0, 1), 0)} %) ·
    in Gebäuden: ${r.covered_indoor ?? '–'}/${r.users_indoor ?? '–'} (${fmt(100 * (r.covered_indoor ?? 0) / Math.max(r.users_indoor ?? 0, 1), 0)} %, inkl. Gebäudedämpfung).
    Die Geräte sind eine Zufallsstichprobe auf Gebäuden im Kreis – bei weniger Geräten schwankt die Zahl, der Anteil bleibt ähnlich.</div>
    <div class="hint">Airtime: Nachricht ${fmt(r.airtime_msg_ms, 0)} ms, Advert ${fmt(r.airtime_advert_ms, 0)} ms · 10 %-Flood-Reichweite: ${fmt(r.flood_reach_p10 * 100, 0)} %
    ${over ? ` · <span class="bad">${over} Repeater über 10 % Duty Cycle</span>` : ''}</div>
    <table><tr><th>Repeater</th><th class="num">Sendungen</th><th class="num">Duty</th><th class="num">Kanal belegt</th><th class="num">Rx-Verlust</th></tr>
    ${r.repeaters.map((x) => `<tr><td>${esc(x.name)}</td><td class="num">${x.tx}</td>
      <td class="num ${x.over_limit ? 'bad' : x.duty > 0.05 ? 'warn' : ''}">${fmt(x.duty * 100)} %</td><td class="num">${fmt(x.busy * 100, 0)} %</td>
      <td class="num ${x.rx_lost > 0.2 ? 'bad' : x.rx_lost > 0.05 ? 'warn' : ''}">${fmt(x.rx_lost * 100, 0)} %</td></tr>`).join('')}</table>`;
}

// -------------------------------------------------------------- Parameter --
const FIELDS = {
  radio: [['freq_mhz', 'Frequenz [MHz]', 0.001], ['bw_khz', 'Bandbreite [kHz]', 0.1], ['sf', 'Spreading-Faktor', 1], ['cr', 'Coding Rate 4/x', 1],
    ['preamble', 'Präambel [Symbole]', 1], ['nf_db', 'Rauschzahl Empfänger [dB]', 0.5]],
  node: [['tx_dbm', 'Sendeleistung [dBm]', 0.5], ['gain_dbi', 'Antennengewinn [dBi]', 0.1], ['body_loss_db', 'Körperdämpfung [dB]', 0.5], ['height_m', 'Höhe über Grund [m]', 0.1]],
  env: [['k_factor', 'Erdradiusfaktor k', 0.01], ['veg_gamma_db_m', 'Bewuchs γ [dB/m] (P.833)', 0.01], ['wall_db', 'Außenwand [dB] (P.2040)', 0.5],
    ['interior_db_m', 'Innen [dB/m]', 0.1], ['building_type', 'Gebäudetyp (P.2109)', ['traditional', 'thermally_efficient']],
    ['building_max_db', 'Max. Durchdringung [dB]', 1], ['diffraction', 'Beugungsmodell', ['delta_bullington', 'deygout']],
    ['ground_eps_r', 'Boden εr', 1], ['ground_sigma', 'Boden σ [S/m]', 0.001], ['step_factor', 'Profilschritt × Raster', 0.25],
    ['canyon', 'Straßenschlucht (P.1411)', 'bool'], ['canyon_max_m', 'Schlucht max. Länge [m]', 10]],
  stats: [['sigma_outdoor_db', 'σ außen [dB]', 0.5], ['sigma_indoor_db', 'σ Gebäudeeintritt [dB]', 0.5], ['target_prob', 'Ziel-Ortswahrsch.', 0.01],
    ['calib_offset_db', 'Kalibrier-Offset Verlust [dB]', 0.1]],
};
function renderSettings() {
  for (const [sec, fs] of Object.entries(FIELDS)) {
    $(`#f-${sec}`).innerHTML = fs.map(([k, t, st]) => {
      const v = S.project[sec][k];
      if (Array.isArray(st)) return `<label>${t}<select data-s="${sec}" data-k="${k}">${st.map((o) => `<option ${o === v ? 'selected' : ''}>${o}</option>`).join('')}</select></label>`;
      if (st === 'bool') return `<label class="check"><input type="checkbox" data-s="${sec}" data-k="${k}" ${v ? 'checked' : ''}> ${t}</label>`;
      return `<label>${t}<input type="number" step="${st}" data-s="${sec}" data-k="${k}" value="${v}"></label>`;
    }).join('');
  }
  const d = S.derived;
  $('#radio-derived').textContent = `Empfindlichkeit ${fmt(d.sens_dbm)} dBm · Airtime 90 Byte ${fmt(d.airtime_msg_ms, 0)} ms · BEL (P.2109) ${fmt(d.bel_db)} dB · Am Bewuchs ${fmt(d.veg_am_db)} dB`;
}
$('#btn-save-set').onclick = async () => {
  const data = {};
  document.querySelectorAll('[data-s]').forEach((el) => {
    (data[el.dataset.s] ||= {})[el.dataset.k] = el.type === 'checkbox' ? el.checked : el.tagName === 'SELECT' ? el.value : +el.value;
  });
  for (const [sec, body] of Object.entries(data)) await api(`/api/settings/${sec}`, { method: 'PUT', body });
  await refresh(); alert('Gespeichert. Für geänderte Ausbreitungsparameter die Ausleuchtung neu berechnen.');
};
$('#btn-clear-cache').onclick = async () => {
  if (!confirm('Alle gecachten Repeater-Berechnungen löschen?')) return;
  const r = await api('/api/cache/clear', { method: 'POST' }); alert(`${r.deleted} Dateien gelöscht`);
};

// ---------------------------------------------------------- Kalibrierung --
let calLayer = null;
$('#cal-preset').onchange = (e) => { $('#cal-extra').value = e.target.value; };
function calSetup() {
  return { rx_gain_dbi: +$('#cal-gain').value, rx_extra_loss_db: +$('#cal-extra').value, rx_height_m: +$('#cal-h').value,
    tx_dbm_node: +$('#cal-tx').value, censor_margin_db: +$('#cal-cens').value, snap_outdoor: $('#cal-snap').checked };
}
let calSets = [];
async function loadSets() {
  if (!S.region) return;
  calSets = await api('/api/measurements');
  $('#cal-sets').innerHTML = calSets.map((e) => `<div class="set"><input type="checkbox" data-m="${e.id}" ${e.enabled !== false ? 'checked' : ''}>
    <div><b>${esc(e.name)}</b><div class="meta">${e.rows.toLocaleString('de-DE')} Zeilen · ${esc(e.added)}${e.warnings?.length ? ' · ' + esc(e.warnings[0]) : ''}</div></div>
    <span></span><button class="x" data-del="${e.id}" title="Datensatz entfernen">×</button></div>`).join('');
  document.querySelectorAll('#cal-sets [data-m]').forEach((el) => el.onchange = async () => { await api(`/api/measurements/${el.dataset.m}`, { method: 'PUT', body: { enabled: el.checked } }); await loadSets(); });
  document.querySelectorAll('#cal-sets [data-del]').forEach((el) => el.onclick = async () => { await api(`/api/measurements/${el.dataset.del}`, { method: 'DELETE' }); await loadSets(); });
  $('#btn-cal').disabled = !calSets.some((e) => e.enabled !== false);
}
async function addFiles(files) {
  let ok = 0;
  for (const f of files) {
    $('#cal-msg').textContent = `${f.name} wird gelesen …`;
    try { await api('/api/measurements', { method: 'POST', body: { name: f.name, text: await f.text() } }); ok++; }
    catch (err) { $('#cal-msg').textContent = 'Fehler: ' + err.message; await loadSets(); return; }
  }
  await loadSets();
  $('#cal-msg').textContent = `${ok} Datei(en) hinzugefügt.`;
  if (ok && $('#cal-auto').checked) runCal();
}
$('#cal-file').onchange = (e) => { addFiles([...e.target.files]); e.target.value = ''; };
const drop = $('#cal-drop');
['dragenter', 'dragover'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', (e) => addFiles([...e.dataTransfer.files]));
async function runCal(url = '/api/calibration', label = 'Werte aus') {
  $('#cal-msg').textContent = `${label} … (Fit kann bei vielen Punkten eine Minute dauern)`; $('#btn-cal').disabled = true; $('#btn-cal-mapme').disabled = true;
  try {
    const r = await api(url, { method: 'POST', body: { fit: $('#cal-fit').checked, setup: calSetup(), auto_apply: $('#cal-auto').checked } });
    const cal = r; await refresh(); S.calibration = cal; renderCal(); await loadSets();
    $('#cal-msg').textContent = '';
  } catch (err) { $('#cal-msg').textContent = 'Fehler: ' + err.message; }
  $('#btn-cal').disabled = !calSets.length; $('#btn-cal-mapme').disabled = false;
}
$('#btn-cal').onclick = () => runCal();
$('#cal-show').onchange = drawCalPoints;
$('#btn-cal-mapme').onclick = () => runCal('/api/calibration/mapme', 'Lade mapme.sh-Daten für alle Repeater mit Public Key und werte aus');

function drawCalPoints() {
  if (calLayer) { calLayer.remove(); calLayer = null; }
  const r = S.calibration;
  if (!r || !$('#cal-show').checked) return;
  calLayer = L.layerGroup();
  for (const p of r.points) {
    let col = '#6b7280', tip;
    if (p.meas === null) { col = '#111827'; tip = 'nicht empfangen'; }
    else if (p.pred !== null) {
      const d = p.meas - p.pred; col = d > 6 ? '#2563eb' : d > 2 ? '#60a5fa' : d < -6 ? '#dc2626' : d < -2 ? '#fc790d' : '#0f766e';
      tip = `gemessen ${fmt(p.meas)} dBm · Modell ${fmt(p.pred)} dBm · Δ ${fmt(d)} dB`;
    }
    L.circleMarker([p.lat, p.lon], { radius: 4, color: '#fff', weight: 1, fillColor: col, fillOpacity: p.used ? 0.95 : 0.45 })
      .bindTooltip(`${esc(p.rep)}<br>${tip}${p.used ? '' : '<br><i>nicht im Fit (Zensierung)</i>'}`).addTo(calLayer);
  }
  calLayer.addTo(map);
}

function scatterSvg(points, sens) {
  const pts = points.filter((p) => p.meas !== null && p.pred !== null && p.pred >= sens - 30);
  if (!pts.length) return '';
  const lo = Math.max(sens - 30, Math.min(sens - 5, ...pts.map((p) => Math.min(p.meas, p.pred)))), hi = Math.max(...pts.map((p) => Math.max(p.meas, p.pred))) + 3;
  const W = 340, H = 240, pad = 30;
  const X = (v) => pad + (v - lo) / (hi - lo) * (W - pad - 8), Y = (v) => H - pad - (v - lo) / (hi - lo) * (H - pad - 8);
  let g = `<line x1="${X(lo)}" y1="${Y(lo)}" x2="${X(hi)}" y2="${Y(hi)}" stroke="#4a6464" stroke-dasharray="4 4"/>`;
  g += `<line x1="${X(sens)}" y1="${Y(lo)}" x2="${X(sens)}" y2="${Y(hi)}" stroke="#dc2626" stroke-width=".7"/>`;
  for (const p of pts) g += `<circle cx="${X(p.pred).toFixed(1)}" cy="${Y(p.meas).toFixed(1)}" r="2.2" fill="${p.used ? '#0f766e' : '#9ca3af'}" opacity=".6"/>`;
  for (let t = Math.ceil(lo / 10) * 10; t <= hi; t += 10) g += `<text x="${X(t)}" y="${H - 12}" font-size="9" text-anchor="middle" fill="#4a6464">${t}</text><text x="4" y="${Y(t) + 3}" font-size="9" fill="#4a6464">${t}</text>`;
  g += `<text x="${W / 2}" y="${H - 1}" font-size="10" text-anchor="middle" fill="#4a6464">Modell [dBm]</text><text x="10" y="12" font-size="10" fill="#4a6464">Messung [dBm]</text>`;
  return `<svg id="cal-svg" viewBox="0 0 ${W} ${H}">${g}</svg>`;
}

function renderCal() {
  const r = S.calibration;
  if (!r) { $('#cal-res').hidden = true; drawCalPoints(); return; }
  $('#cal-res').hidden = false;
  if (!r.points) { $('#cal-res').hidden = true; return; }
  const det = r.detection || {};
  const tot = (det.received_predicted || 0) + (det.received_not_predicted || 0) + (det.lost_predicted || 0) + (det.lost_not_predicted || 0);
  let h = `<h3>Abgleich Modell ↔ Messung</h3><div class="kpis">
    <div class="kpi"><b>${r.n_used}</b><span>Messungen im Fit (von ${r.n_matched})</span></div>
    <div class="kpi"><b class="${Math.abs(r.bias_db) > 3 ? 'warn' : ''}">${fmt(r.bias_db)} dB</b><span>Mittelfehler (Mess − Modell)</span></div>
    <div class="kpi"><b>${fmt(r.std_db)} dB</b><span>Streuung (σ)</span></div></div>
    ${r.source ? `<div class="hint">Quelle <b>${esc(r.source.name)}</b>: ${r.source.hexes_total} Waben geladen, ${r.source.dropped_ambiguous} wegen Hash-Kollision (gleiches 1-Byte-Präfix, anderer Repeater näher) und ${r.source.dropped_far} wegen Entfernung &gt; 40 km verworfen.</div>` : ''}
    <div class="hint">Robust: Median ${fmt(r.median_db)} dB · σ(MAD) ${fmt(r.mad_sigma_db)} dB · RMSE ${fmt(r.rmse_db)} dB · MAE ${fmt(r.mae_db)} dB · Empfindlichkeit ${fmt(r.sens_dbm)} dBm${r.weighted ? ' · gewichtet nach Samples' : ''}
    ${Object.keys(r.unmatched || {}).length ? `<br><span class="warn">Nicht zugeordnet:</span> ${esc(Object.entries(r.unmatched).map(([k, n]) => `${k} (${n})`).join(', '))}` : ''}
    ${(r.warnings || []).length ? '<br>' + esc(r.warnings.join(' · ')) : ''}</div>
    ${scatterSvg(r.points, r.sens_dbm)}
    <div class="hint cal-legend">Karte:<span style="background:#2563eb"></span>Modell zu pessimistisch<span style="background:#0f766e"></span>±2 dB<span style="background:#dc2626"></span>zu optimistisch<span style="background:#111827"></span>nicht empfangen</div>`;
  if (tot) h += `<h3>Trefferquote Versorgung</h3><table><tr><th></th><th class="num">Modell: Empfang</th><th class="num">Modell: kein Empfang</th></tr>
    <tr><td>Gemessen: empfangen</td><td class="num good">${det.received_predicted}</td><td class="num warn">${det.received_not_predicted}</td></tr>
    <tr><td>Gemessen: verloren</td><td class="num warn">${det.lost_predicted}</td><td class="num good">${det.lost_not_predicted}</td></tr></table>
    <div class="hint">Übereinstimmung ${fmt(100 * (det.received_predicted + det.lost_not_predicted) / tot, 0)} %</div>`;
  if (r.per_distance?.length) h += `<h3>Nach Entfernung</h3><table><tr><th>Entfernung</th><th class="num">n</th><th class="num">Mittelfehler</th><th class="num">σ</th></tr>` +
    r.per_distance.map((d) => `<tr><td>${fmt(d.from_m / 1000, 1)} – ${d.to_m ? fmt(d.to_m / 1000, 1) : '…'} km</td><td class="num">${d.n}</td><td class="num">${fmt(d.bias_db)}</td><td class="num">${fmt(d.std_db)}</td></tr>`).join('') + '</table>';
  if (r.per_repeater?.length) h += `<h3>Nach Repeater</h3><table><tr><th>Repeater</th><th class="num">n</th><th class="num">Mittelfehler</th><th class="num">σ</th></tr>` +
    r.per_repeater.map((d) => `<tr><td>${esc(d.name)}</td><td class="num">${d.n}</td><td class="num ${Math.abs(d.bias_db) > 6 ? 'warn' : ''}">${fmt(d.bias_db)}</td><td class="num">${fmt(d.std_db)}</td></tr>`).join('') + '</table>' +
    '<div class="hint">Ein einzelner Repeater mit großem Mittelfehler deutet meist auf falsche Antennenhöhe, Leistung oder Kabelverlust hin – erst dort korrigieren.</div>';
  const f = r.fit;
  if (r.datasets?.length) h = h.replace('<h3>Abgleich Modell ↔ Messung</h3>', `<h3>Abgleich Modell ↔ Messung</h3><div class="hint">Datensätze: ${esc(r.datasets.join(', '))}</div>`);
  if (f?.error) h += `<div class="hint warn">${esc(f.error)}</div>`;
  else if (f?.best) {
    const sg = f.suggest;
    const top = [...f.candidates].sort((a, b) => a.rmse_db - b.rmse_db).slice(0, 5);
    h += `<h3>Fit-Ergebnis</h3><table><tr><th>Beugung</th><th class="num">γ [dB/m]</th><th class="num">Bias</th><th class="num">RMSE</th></tr>` +
      top.map((c, i) => `<tr><td>${i === 0 ? '<b>' : ''}${c.diffraction === 'deygout' ? 'Deygout' : 'Delta-Bullington'}${i === 0 ? '</b>' : ''}</td><td class="num">${fmt(c.veg_gamma_db_m, 2)}</td><td class="num">${fmt(c.bias_db)}</td><td class="num">${fmt(c.rmse_db)}</td></tr>`).join('') + '</table>' +
      `<div class="hint">Vorschlag: Offset ${fmt(sg.calib_offset_db)} dB (aktuell ${fmt(r.current.offset_db)}) · γ ${fmt(sg.veg_gamma_db_m, 2)} dB/m · ${sg.diffraction} · σ außen ${fmt(sg.sigma_outdoor_db)} dB</div>
      ${r.auto_applied ? `<div class="hint good">✓ Automatisch übernommen (Offset ${fmt(sg.calib_offset_db)} dB, γ ${fmt(sg.veg_gamma_db_m, 2)}, ${sg.diffraction}, σ ${fmt(sg.sigma_outdoor_db)} dB). Die Karte zeigt noch den alten Stand.</div><button class="btn primary" id="btn-cal-recalc">Ausleuchtung neu berechnen</button>`
        : f.reliable === false
        ? `<div class="hint bad">Nicht plausibel: Reststreuung ${fmt(f.best.rmse_db)} dB (realistisch 5–10 dB, Grenze ${fmt(f.max_plausible_rmse_db, 0)} dB). Die Messungen passen nicht zum angenommenen Aufbau – typische Ursachen: falsche Antennenhöhen/Leistung der Repeater (siehe Tabelle „Nach Repeater“), Mapper auf Türmen/Hügeln, falsche Zuordnung. Erst die Repeaterdaten korrigieren, dann erneut auswerten.</div>
           <button class="btn" id="btn-cal-apply" disabled>Vorschlag übernehmen</button>`
        : '<button class="btn primary" id="btn-cal-apply">Vorschlag übernehmen</button>'}`;
  }
  $('#cal-res').innerHTML = h;
  const rc = $('#btn-cal-recalc');
  if (rc) rc.onclick = () => { document.querySelector('.tabs [data-tab=calc]').click(); $('#btn-calc').click(); };
  const b = $('#btn-cal-apply');
  if (b) b.onclick = async () => {
    await api('/api/calibration/apply', { method: 'POST', body: {} });
    await refresh();
    alert('Übernommen. Offset und σ wirken sofort nach erneutem „Ausleuchtung berechnen“ (aus Cache); geänderte γ/Beugung rechnen die Repeater neu.');
  };
  drawCalPoints();
}

// ------------------------------------------------------------------- Info --
function renderInfo() {
  $('#foot').innerHTML = `Funkmesh Repeater-Simulator ${esc(S.version)} · Open Source (MIT) · <a href="https://funkmesh.de" target="_blank">funkmesh.de</a>`;
  if (!S.region) { $('#info').innerHTML = '<p>Noch kein Gebiet – im Reiter „Gebiet“ anlegen.</p>'; return; }
  const src = S.region.sources;
  const li = (k, t) => src[k] ? `<li>${t}: ${esc(src[k])}</li>` : '';
  $('#info').innerHTML = `<h4>Datengrundlage (${esc(S.region.title)}, Raster ${fmt(S.region.res, 0)} m)</h4><ul>
    ${li('terrain', 'Gelände')}${li('buildings', 'Gebäude')}${li('vegetation', 'Bewuchs')}
    ${src.terrain_nrw && src.terrain_nrw !== src.terrain ? `<li>In NRW: ${esc(src.terrain_nrw)}, ${esc(src.buildings_nrw)}, ${esc(src.vegetation_nrw)}</li>` : ''}
    <li>Repeater: map.meshcore.io, mapme.sh · Ortssuche/Grenzen: Nominatim, © OpenStreetMap-Mitwirkende · Karten: basemap.de / BKG, Geobasis NRW</li></ul>
    <h4>Ausbreitungsmodell (je Rasterzelle, Profil im ${fmt(S.region.res, 0)}-m-Raster)</h4><ul>
    <li><b>Beugung:</b> Delta-Bullington nach ITU-R P.1812-7 über Gelände + LoD2-Gebäude + Bewuchs, inkl. sphärischer Erdbeugung (P.452) mit effektiven Antennenhöhen; alternativ Deygout (3 Kanten).</li>
    <li><b>Bewuchs:</b> ITU-R P.833 (γ = ${fmt(S.project.env.veg_gamma_db_m, 2)} dB/m, Am = ${fmt(S.derived.veg_am_db)} dB) – Minimum aus „über die Krone“ und „durch die Krone“.</li>
    <li><b>Gebäude:</b> Durchdringung (Wände je ${fmt(S.project.env.wall_db)} dB + ${fmt(S.project.env.interior_db_m)} dB/m) leistungsaddiert zur Dachkantenbeugung; Innenraum-Empfang mit Gebäudeeintrittsdämpfung ITU-R P.2109 (${fmt(S.derived.bel_db)} dB, Median).</li>
    <li><b>Straßenschluchten:</b> Ausbreitung um Ecken entlang von Straßen nach ITU-R P.1411 (UHF, NLoS) – aktiv, wenn die Repeaterantenne unter Dachniveau liegt.</li>
    <li><b>Antenne:</b> Vertikaldiagramm einer Kollinear-Antenne passend zum Gewinn (Nullstellen steil unter dem Mast).</li>
    <li><b>Statistik:</b> lokaler Mittelwert (3×3 Zellen), Ortsvariabilität σ = ${fmt(S.project.stats.sigma_outdoor_db)} dB (außen) – daraus Versorgungswahrscheinlichkeit und Redundanz.</li>
    <li><b>Kalibrierung:</b> Offset ${fmt(S.project.stats.calib_offset_db)} dB auf den Pfadverlust (aus Messungen, Reiter „Kalibrierung“).</li>
    <li><b>Linkbudget:</b> Downlink und Uplink getrennt (Repeater ${'↔'} Endgerät mit ${fmt(S.project.node.tx_dbm)} dBm, ${fmt(S.project.node.height_m)} m Höhe, ${fmt(S.project.node.body_loss_db)} dB Körperdämpfung); LoRa-Empfindlichkeit aus SF/BW/Rauschzahl.</li></ul>
    <h4>Rechner</h4><p>Backend: <b>${S.backend === 'cuda' ? 'NVIDIA-GPU (CUDA, float32)' : 'CPU (numba, parallel)'}</b>. Alle großen Raster liegen als Memory-Map auf der Platte; RAM/VRAM werden vor jeder Allokation geprüft.</p>
    <h4>Grenzen</h4><p>Keine Messkalibrierung ab Werk – Ergebnisse sind Modellwerte. Mobile Hindernisse, Fensterbeschichtungen, Antennenumgebung am Mast und Interferenzen anderer Funkdienste sind nicht enthalten.
    Außerhalb NRW stammen Gelände und Bewuchs aus 30-m-Satellitendaten (Copernicus/ESA) und Gebäude aus OpenStreetMap – deutlich gröber als die 1-m-Laserdaten in NRW. In großen Gebieten ohne Einzelgebäude wird Bebauung als 8 m hoher „Clutter“ behandelt (keine Innenraumwerte).</p>`;
}

// --------------------------------------------------------------- Allgemein --
document.querySelectorAll('.tabs button').forEach((b) => b.addEventListener('click', () => {
  document.querySelectorAll('.tabs button').forEach((x) => x.classList.toggle('active', x === b));
  document.querySelectorAll('.tab').forEach((t) => { t.hidden = t.id !== `tab-${b.dataset.tab}`; });
}));
$('#layers').innerHTML = LAYERS.map(([k, t]) => `<label><input type="radio" name="lay" value="${k}" ${k === curLayer ? 'checked' : ''}> ${t}</label>`).join('');
document.querySelectorAll('#layers input').forEach((el) => el.addEventListener('change', () => setLayer(el.value)));
$('#opacity').oninput = (e) => resultLayer.setOpacity(+e.target.value);

async function refresh() {
  const cal = S?.calibration;
  const prevArea = S?.region?.name;
  S = await api('/api/state');
  S.calibration = cal;
  renderAreas(); renderInfo();
  const hasArea = !!S.region;
  document.querySelectorAll('.tabs button').forEach((b) => { if (b.dataset.tab !== 'area' && b.dataset.tab !== 'info') b.disabled = !hasArea; });
  $('#area-pill').textContent = hasArea ? `▦ ${S.region.title}` : 'Kein Gebiet – anlegen';
  if (!hasArea) { drawRepeaters(); showTab('area'); return; }
  if (prevArea !== S.region.name) { areaChanged(); }
  const changed = S.result_version !== resultVersion || prevArea !== S.region.name;
  resultVersion = S.result_version;
  drawRepeaters(); renderSettings(); renderSummary(); renderLoad();
  const r0 = S.region.res;
  $('#c-res').innerHTML = [1, 2, 4, 10].map((k) => r0 * k).filter((v) => v <= 200)
    .map((v) => `<option value="${v}">${v} m${v === r0 ? ' (Datenraster)' : ''}</option>`).join('');
  $('#c-res').value = String(S.project.compute.out_res); $('#c-rad').value = S.project.compute.radius_km;
  $('#c-area').textContent = `${S.region.title}, ${S.region.km2.toLocaleString('de-DE')} km²`;
  const big = S.region.km2 > 20000;
  $('#c-warn').hidden = !big;
  $('#c-warn').textContent = big ? 'Großes Gebiet: Die Berechnung kann je nach Repeaterzahl Stunden dauern. Vor dem Start erscheint eine Abschätzung.' : '';
  $('#sub').textContent = `${S.project.radio.freq_mhz} MHz · SF${S.project.radio.sf}/BW${S.project.radio.bw_khz} · ${S.backend === 'cuda' ? 'GPU' : 'CPU'}`;
  if (changed) { setLayer(curLayer); drawLinks(); }
}

let boundaryLayer = null;
async function areaChanged() {
  const [w, s, e, n] = S.region.bounds;
  map.fitBounds([[s, w], [n, e]]);
  if (boundaryLayer) boundaryLayer.remove();
  boundaryLayer = L.geoJSON(S.region.boundary, { style: { color: '#0f766e', weight: 2, dashArray: '6 6', fill: false }, interactive: false }).addTo(map);
  checked.clear(); selected = null; $('#editor').hidden = true; $('#import-box').hidden = true;
  if (calLayer) { calLayer.remove(); calLayer = null; }
  S.calibration = null; $('#cal-res').hidden = true;
  try { S.calibration = await api('/api/calibration'); renderCal(); } catch (e) {}
  loadSets();
  const nd = S.project.node;
  $('#cal-gain').value = nd.gain_dbi; $('#cal-extra').value = nd.body_loss_db; $('#cal-h').value = nd.height_m; $('#cal-tx').value = nd.tx_dbm;
  if (S.job.state === 'running') pollJob();
}

function showTab(name) {
  document.querySelectorAll('.tabs button').forEach((x) => x.classList.toggle('active', x.dataset.tab === name));
  document.querySelectorAll('.tab').forEach((t) => { t.hidden = t.id !== `tab-${name}`; });
}
$('#area-pill').onclick = () => showTab('area');

// ------------------------------------------------------------------ Gebiete --
let areaSel = null;      // {bbox, boundary, title}
let areaRect = null;
const DE = () => S.germany_bbox;
function renderAreas() {
  const act = S.region?.name;
  $('#area-list').innerHTML = S.areas.map((a) => `<div class="area ${a.name === act ? 'act' : ''}">
    <div><div class="nm">${esc(a.title)}${a.name === act ? ' <span class="badge">aktiv</span>' : ''}</div>
    <div class="meta">${fmt(a.res, 0)} m · ${a.km2.toLocaleString('de-DE')} km² · ${fmt(a.disk_gb, 2)} GB${a.complete ? '' : ' · <span class="warn">unvollständig</span>'}</div></div>
    <div class="acts">${a.complete ? (a.name === act ? '' : `<button class="btn small" data-act="${a.name}">Öffnen</button>`) : `<button class="btn small" data-res="${a.name}">Fortsetzen</button>`}
    ${a.name === act ? '' : `<button class="btn small danger" data-del="${a.name}" title="Gebiet mit Geodaten, Repeatern und Ergebnissen löschen">×</button>`}</div></div>`).join('')
    || '<div class="hint">Noch kein Gebiet. Unten einen Kreis suchen, ein Rechteck zeichnen oder ganz Deutschland wählen.</div>';
  document.querySelectorAll('#area-list [data-act]').forEach((b) => b.onclick = async () => {
    try { await api(`/api/areas/${b.dataset.act}/activate`, { method: 'POST' }); await refresh(); showTab('rep'); } catch (e) { alert(e.message); }
  });
  document.querySelectorAll('#area-list [data-res]').forEach((b) => b.onclick = async () => {
    try { await api(`/api/areas/${b.dataset.res}/resume`, { method: 'POST' }); pollArea(); } catch (e) { alert(e.message); }
  });
  document.querySelectorAll('#area-list [data-del]').forEach((b) => b.onclick = async () => {
    const a = S.areas.find((x) => x.name === b.dataset.del);
    if (!confirm(`Gebiet „${a.title}“ endgültig löschen?\n\nGeodaten (${fmt(a.disk_gb, 2)} GB), Repeater, Ergebnisse und Kalibrierung dieses Gebiets werden gelöscht.`)) return;
    try { await api(`/api/areas/${b.dataset.del}`, { method: 'DELETE' }); await refresh(); } catch (e) { alert(e.message); }
  });
  const j = S.area_job;
  if (j && j.state === 'running' && !areaPolling) pollArea();
}

function setAreaSel(sel) {
  areaSel = sel;
  if (areaRect) { areaRect.remove(); areaRect = null; }
  if (sel) {
    const [w, s, e, n] = sel.bbox;
    areaRect = L.featureGroup([L.rectangle([[s, w], [n, e]], { color: '#fc790d', weight: 2, dashArray: '6 4', fillOpacity: 0.05, interactive: false })]);
    if (sel.boundary) areaRect.addLayer(L.geoJSON(sel.boundary, { style: { color: '#fc790d', weight: 2, fill: false }, interactive: false }));
    areaRect.addTo(map);
    if (!$('#ar-title').value || $('#ar-title').dataset.auto === '1') { $('#ar-title').value = sel.title || ''; $('#ar-title').dataset.auto = '1'; }
  }
  $('#ar-go').disabled = !sel;
  estimateArea();
}
$('#ar-title').oninput = () => { $('#ar-title').dataset.auto = '0'; };
function autoRes(sel) {
  const [w, s, e, n] = sel.bbox;
  const km2 = (e - w) * 111 * Math.cos((s + n) / 2 * Math.PI / 180) * (n - s) * 111;
  return km2 > 60000 ? 50 : km2 > 8000 ? 25 : 10;
}
$('#ar-search').onclick = async () => {
  const q = $('#ar-q').value.trim(); if (!q) return;
  $('#ar-results').innerHTML = '<div class="hint">suche …</div>';
  try {
    const r = await api(`/api/places?q=${encodeURIComponent(q)}`);
    $('#ar-results').innerHTML = r.map((x, i) => `<div class="rep" data-i="${i}" style="grid-template-columns:1fr auto"><div><div class="nm">${esc(x.name)}</div><div class="meta">${esc(x.display)}</div></div><span class="meta">${esc(x.type || '')}</span></div>`).join('') || '<div class="hint">Nichts gefunden.</div>';
    document.querySelectorAll('#ar-results .rep').forEach((el) => el.onclick = () => {
      const x = r[+el.dataset.i];
      const sel = { bbox: x.bbox, boundary: x.boundary, title: x.name };
      $('#ar-res').value = String(autoRes(sel));
      setAreaSel(sel); map.fitBounds([[x.bbox[1], x.bbox[0]], [x.bbox[3], x.bbox[2]]]);
      $('#ar-results').innerHTML = '';
    });
  } catch (e) { $('#ar-results').innerHTML = `<div class="hint bad">${esc(e.message)}</div>`; }
};
$('#ar-q').onkeydown = (e) => { if (e.key === 'Enter') $('#ar-search').click(); };
$('#ar-view').onclick = () => {
  const b = map.getBounds();
  const sel = { bbox: [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()], boundary: null, title: 'Kartenausschnitt' };
  $('#ar-res').value = String(autoRes(sel)); setAreaSel(sel);
};
$('#ar-de').onclick = async () => {
  const sel = { bbox: DE(), boundary: null, title: 'Deutschland' };
  $('#ar-res').value = '50'; $('#ar-buf').value = '0'; $('#ar-osm').checked = false;
  setAreaSel(sel); map.fitBounds([[DE()[1], DE()[0]], [DE()[3], DE()[2]]]);
  try { const r = await api('/api/places?q=Deutschland&kind=country'); if (r[0] && areaSel === sel) { sel.boundary = r[0].boundary; setAreaSel(sel); } } catch (e) {}
};
$('#ar-clear').onclick = () => { setAreaSel(null); $('#ar-est').innerHTML = ''; };
// Rechteck ziehen
let drawing = false, drawStart = null, drawRect = null;
$('#ar-draw').onclick = () => setDrawing(!drawing);
function setDrawing(on) {
  drawing = on; document.body.classList.toggle('drawing', on); $('#ar-draw').classList.toggle('on', on); $('#ar-draw-hint').hidden = !on;
  if (on) map.dragging.disable(); else map.dragging.enable();
}
map.on('mousedown', (e) => { if (!drawing) return; drawStart = e.latlng; drawRect = L.rectangle([e.latlng, e.latlng], { color: '#fc790d', weight: 2 }).addTo(map); });
map.on('mousemove', (e) => { if (drawing && drawStart && drawRect) drawRect.setBounds([drawStart, e.latlng]); });
map.on('mouseup', (e) => {
  if (!drawing || !drawStart) return;
  const b = L.latLngBounds(drawStart, e.latlng); drawRect.remove(); drawRect = null; drawStart = null; setDrawing(false); justDrew = Date.now();
  if (Math.abs(b.getEast() - b.getWest()) < 0.005 || Math.abs(b.getNorth() - b.getSouth()) < 0.003) return;
  const sel = { bbox: [b.getWest(), b.getSouth(), b.getEast(), b.getNorth()], boundary: null, title: 'Eigenes Gebiet' };
  $('#ar-res').value = String(autoRes(sel)); setAreaSel(sel);
});
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && drawing) { setDrawing(false); if (drawRect) { drawRect.remove(); drawRect = null; } drawStart = null; } });
['#ar-res', '#ar-buf', '#ar-nrw', '#ar-osm', '#ar-bh'].forEach((id) => $(id).addEventListener('change', estimateArea));

function areaBody() {
  return { title: $('#ar-title').value.trim() || areaSel.title || 'Gebiet', bbox: areaSel.bbox, boundary: areaSel.boundary, res: +$('#ar-res').value,
    buffer_m: +$('#ar-buf').value * 1000, use_nrw: $('#ar-nrw').checked, use_osm: $('#ar-osm').checked, default_building_m: +$('#ar-bh').value };
}
let estSeq = 0, lastEst = null;
async function estimateArea() {
  if (!areaSel) { $('#ar-est').innerHTML = ''; return; }
  const seq = ++estSeq;
  $('#ar-est').innerHTML = 'Abschätzung …';
  try {
    const e = await api('/api/areas/estimate', { method: 'POST', body: areaBody() });
    if (seq !== estSeq) return;
    lastEst = e;
    const src = e.use_nrw ? (e.nrw_tiles >= e.km2 ? 'NRW 1 m (DGM1/DOM1/LoD2)' : `NRW 1 m (${e.nrw_tiles.toLocaleString('de-DE')} km²) + Copernicus/WorldCover${e.use_osm ? ' + OSM-Gebäude' : ''}`)
      : `Copernicus GLO-30 + ESA WorldCover${e.use_osm ? ' + OSM-Gebäude' : ' (Bebauung als Clutter)'}`;
    const c = e.compute;
    const huge = e.km2 > 20000 || (c.repeaters_known || 0) > 300;
    $('#ar-est').innerHTML = `<table>
      <tr><td>Fläche</td><td><b>${e.km2.toLocaleString('de-DE')} km²</b> (inkl. Rand) · ${(e.cells / 1e6).toLocaleString('de-DE', { maximumFractionDigits: 0 })} Mio. Zellen à ${e.res} m</td></tr>
      <tr><td>Daten</td><td>${src}</td></tr>
      <tr><td>Platte</td><td>${fmt(e.disk_gb, e.disk_gb < 1 ? 2 : 1)} GB Geodaten (+ Ergebnisse) · frei ${fmt(e.disk_free_gb, 0)} GB</td></tr>
      <tr><td>Download</td><td>ca. ${e.download_mb > 1000 ? fmt(e.download_mb / 1000, 1) + ' GB' : e.download_mb + ' MB'}</td></tr>
      <tr><td>Vorbereitung</td><td>ca. ${dur(e.prep_s)} (einmalig, fortsetzbar)</td></tr>
      <tr><td>Berechnung</td><td>${c.repeaters_known ?? '?'} bekannte Repeater × ${fmt(c.sec_per_repeater, 1)} s ≈ <b>${c.total_s != null ? dur(c.total_s) : '?'}</b> (${S.backend === 'cuda' ? 'GPU' : 'CPU'}, 15 km Radius)</td></tr></table>
      ${e.res <= 5 && !e.use_nrw ? '<div class="warnbox">5 m bringt außerhalb NRW nichts (Quelldaten 10–30 m) – 10 m wählen.</div>' : ''}
      ${e.exists ? '<div class="warnbox">Ein Gebiet mit diesem Namen existiert bereits – anderen Namen wählen oder das unvollständige Gebiet fortsetzen.</div>' : ''}
      ${huge ? `<div class="warnbox"><b>Achtung – großes Gebiet.</b> Vorbereitung und vor allem die Ausleuchtungsberechnung dauern lange (Stunden) und lasten GPU/CPU, Netz und Platte stark aus. Genauigkeit bei ${e.res} m: Überblick, keine Straßenzüge. Für Detailplanung lieber einzelne Kreise als eigene Gebiete anlegen.</div>` : ''}
      ${e.disk_gb * 1.6 + 5 > e.disk_free_gb ? '<div class="warnbox">Nicht genug freier Plattenplatz.</div>' : ''}`;
  } catch (err) { if (seq === estSeq) $('#ar-est').innerHTML = `<span class="bad">${esc(err.message)}</span>`; }
}
$('#ar-go').onclick = async () => {
  if (!areaSel) return;
  const e = lastEst;
  if (e && (e.km2 > 20000 || (e.compute?.repeaters_known || 0) > 300)) {
    if (!confirm(`„${$('#ar-title').value}“ ist ein sehr großes Gebiet (${e.km2.toLocaleString('de-DE')} km²).\n\nVorbereitung: ca. ${dur(e.prep_s)}, Download ca. ${fmt(e.download_mb / 1000, 1)} GB, ${fmt(e.disk_gb, 1)} GB Platte.\nAusleuchtung danach: ca. ${e.compute.total_s != null ? dur(e.compute.total_s) : '?'} für alle ${e.compute.repeaters_known ?? '?'} Repeater.\n\nTrotzdem vorbereiten?`)) return;
  }
  try { await api('/api/areas', { method: 'POST', body: areaBody() }); setAreaSel(null); $('#ar-est').innerHTML = ''; pollArea(); }
  catch (err) { alert(err.message); }
};
$('#ar-stop').onclick = () => api('/api/areas/job/cancel', { method: 'POST' });
let areaPolling = false;
async function pollArea() {
  areaPolling = true;
  const { area_job: j } = await api('/api/job');
  $('#ar-job').hidden = j.state === 'idle';
  $('#ar-job-title').textContent = `Gebiet „${j.title}“ ${j.state === 'running' ? 'wird vorbereitet' : j.state === 'done' ? 'ist fertig' : j.state === 'error' ? ': Fehler' : 'angehalten'}`;
  $('#ar-prog').style.width = `${Math.round(j.progress * 100)}%`;
  $('#ar-job-msg').textContent = `${j.message} · ${dur(j.elapsed)}`;
  $('#ar-stop').hidden = j.state !== 'running';
  if (j.state === 'running') { setTimeout(pollArea, 1500); return; }
  areaPolling = false;
  await refresh();
}

// ---------------------------------------------------------------- Reset --
$('#btn-reset').onclick = async () => {
  if (!confirm(`Gebiet „${S.region.title}“ komplett zurücksetzen?\n\nGelöscht werden: ${S.project.repeaters.length} Repeater, alle Parameter (→ Standard), Ergebnisse, Auslastung, Kalibrierung und Messdatensätze.\nDie Geodaten bleiben erhalten. Das lässt sich nicht rückgängig machen.`)) return;
  try {
    await api('/api/reset', { method: 'POST', body: { cache: $('#reset-cache').checked } });
    S.calibration = null; S.load = null; resultVersion = -1; checked.clear(); selected = null; $('#editor').hidden = true;
    if (calLayer) { calLayer.remove(); calLayer = null; }
    if (linkLayer) { linkLayer.remove(); linkLayer = null; }
    $('#cal-res').hidden = true; $('#load-res').hidden = true; $('#job-log').textContent = ''; $('#job-msg').textContent = ''; $('#prog').style.width = '0';
    await refresh(); await loadSets(); setLayer('terrain'); document.querySelector('#layers input[value="terrain"]').checked = true;
    alert('Zurückgesetzt.');
  } catch (e) { alert(e.message); }
};

(async function init() {
  await refresh();
  if (!S.region) { map.fitBounds([[DE()[1], DE()[0]], [DE()[3], DE()[2]]]); return; }
  setLayer(S.result ? 'margin' : 'terrain');
  document.querySelector(`#layers input[value="${curLayer}"]`).checked = true;
})();
