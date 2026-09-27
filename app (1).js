/* Mobi·Agents Bordeaux — prototype d'IA agentique pour la recommandation de trajets.
 * Trois agents : Préférences (LLM), Environnement (données ouvertes), Décision (arbitrage multicritère).
 * Toutes les sources ont un repli explicite : le prototype reste utilisable hors ligne et le signale.
 */
"use strict";

/* ------------------------------------------------------------------ */
/* Constantes                                                          */
/* ------------------------------------------------------------------ */
const CENTER = [44.8378, -0.5792];
const CRITERIA = [
  { k: "temps", label: "Temps" },
  { k: "securite", label: "Sécurité" },
  { k: "confort", label: "Confort" },
  { k: "cout", label: "Coût" },
  { k: "environnement", label: "Environnement" },
  { k: "effort", label: "Effort évité" },
];
const DEFAULT_W = { temps: 0.3, securite: 0.3, confort: 0.3, cout: 0.3, environnement: 0.3, effort: 0.3 };
const GATE_C = 0.35, GATE_K = 20;       // facteur de sécurité (cf. OTI : c et k)
const HYSTERESIS = 0.04;                // écart minimal pour changer de recommandation
const LEARNING_RATE = 0.35;             // mise à jour du profil sur retour usager
const V3_RADIUS_KM = 0.6;               // distance max à pied jusqu'à une station V³

// Lieux de référence (repli du géocodage et reconnaissance rapide)
const PLACES = [
  { n: "Gare Saint-Jean", a: ["gare saint-jean", "gare saint jean", "saint-jean", "saint jean", "la gare"], lat: 44.8256, lon: -0.5563 },
  { n: "Place de la Victoire", a: ["victoire"], lat: 44.8309, lon: -0.5727 },
  { n: "Place de la Bourse", a: ["bourse", "miroir d'eau"], lat: 44.8413, lon: -0.5695 },
  { n: "Quinconces", a: ["quinconces"], lat: 44.8446, lon: -0.5746 },
  { n: "Pey-Berland", a: ["pey-berland", "pey berland", "hôtel de ville", "hotel de ville", "cathédrale"], lat: 44.8378, lon: -0.5779 },
  { n: "Place Gambetta", a: ["gambetta"], lat: 44.8410, lon: -0.5810 },
  { n: "Mériadeck", a: ["mériadeck", "meriadeck"], lat: 44.8380, lon: -0.5860 },
  { n: "Capucins", a: ["capucins"], lat: 44.8305, lon: -0.5680 },
  { n: "Chartrons", a: ["chartrons"], lat: 44.8530, lon: -0.5700 },
  { n: "Jardin public", a: ["jardin public"], lat: 44.8490, lon: -0.5780 },
  { n: "Cité du Vin", a: ["cité du vin", "cite du vin"], lat: 44.8625, lon: -0.5503 },
  { n: "Bassins à flot", a: ["bassins à flot", "bassins a flot", "bassins"], lat: 44.8660, lon: -0.5580 },
  { n: "Darwin (La Bastide)", a: ["darwin", "bastide"], lat: 44.8499, lon: -0.5590 },
  { n: "LaBRI (Talence)", a: ["labri", "campus", "université", "universite", "talence"], lat: 44.8083, lon: -0.5966 },
  { n: "ENSEIRB-MATMECA (Talence)", a: ["enseirb", "matmeca"], lat: 44.8066, lon: -0.6056 },
  { n: "Hôpital Pellegrin", a: ["pellegrin", "chu", "hôpital", "hopital"], lat: 44.8290, lon: -0.6050 },
  { n: "Stade Matmut Atlantique", a: ["matmut", "stade"], lat: 44.8973, lon: -0.5614 },
  { n: "Aéroport de Mérignac", a: ["aéroport", "aeroport"], lat: 44.8283, lon: -0.7156 },
  { n: "Pont de Pierre", a: ["pont de pierre"], lat: 44.8390, lon: -0.5640 },
];

const EXAMPLES = [
  "Je vais de la Gare Saint-Jean au LaBRI, il pleut un peu et je préfère éviter les grands axes à vélo, mais je ne veux pas arriver trop tard.",
  "Je suis en fauteuil roulant, je dois aller de Pey-Berland à la Cité du Vin.",
  "Je suis pressée : Place de la Victoire → Quinconces, le moins cher possible.",
  "J'ai une grosse valise, je vais du Jardin public à la Gare Saint-Jean.",
  "Avec mon vélo, des Chartrons à Darwin, je veux le trajet le plus écolo et le plus sûr.",
];

// Modes candidats : paramètres posés à la main (à apprendre dans la thèse)
const MODES = {
  marche:      { label: "Marche", color: "#5d6b7c", speed: 4.8, co2: 0,  safe: 0.90, comfort: 0.80, exposed: true },
  velo:        { label: "Vélo personnel", color: "#0e8f80", speed: 15, co2: 0,  safe: 0.62, comfort: 0.72, exposed: true, bike: true },
  velo_calme:  { label: "Vélo personnel — itinéraire calme", color: "#15a38f", speed: 14, co2: 0, safe: 0.84, comfort: 0.78, exposed: true, bike: true, detour: 1.18 },
  v3:          { label: "V³ (vélo partagé)", color: "#1f7ab8", speed: 14, co2: 5, safe: 0.62, comfort: 0.66, exposed: true, bike: true, shared: true },
  v3_calme:    { label: "V³ — itinéraire calme", color: "#3b95d1", speed: 13, co2: 5, safe: 0.84, comfort: 0.72, exposed: true, bike: true, shared: true, detour: 1.18 },
  trottinette: { label: "Trottinette partagée", color: "#b7791f", speed: 17, co2: 35, safe: 0.46, comfort: 0.60, exposed: true, scooter: true },
  tc:          { label: "Tram / bus (TBM)", color: "#6b4fbb", speed: 18, co2: 25, safe: 0.95, comfort: 0.72, exposed: false },
};

/* ------------------------------------------------------------------ */
/* État                                                                */
/* ------------------------------------------------------------------ */
const state = {
  llm: null,                 // true / false / null (inconnu)
  profile: { ...DEFAULT_W }, // profil appris (session)
  feedbacks: 0,
  request: null,             // sortie de l'Agent Préférences
  env: null,                 // sortie de l'Agent Environnement
  events: new Set(),
  result: null,              // sortie de l'Agent Décision
  currentRec: null,          // id de l'option recommandée (pour l'hystérésis)
  selected: null,
  stationsCache: null,
};

/* ------------------------------------------------------------------ */
/* Utilitaires                                                         */
/* ------------------------------------------------------------------ */
const $ = (s) => document.querySelector(s);
const clamp = (x, a = 0, b = 1) => Math.max(a, Math.min(b, x));
const fmt = (x, d = 2) => Number(x).toFixed(d).replace(".", ",");
const norm = (s) => s.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");
function haversine(a, b) {
  const R = 6371, toR = Math.PI / 180;
  const dLat = (b.lat - a.lat) * toR, dLon = (b.lon - a.lon) * toR;
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(a.lat * toR) * Math.cos(b.lat * toR) * Math.sin(dLon / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(h));
}
function normalizeW(w) {
  const out = {}; let s = 0;
  for (const c of CRITERIA) { out[c.k] = Math.max(0.02, Number(w?.[c.k]) || 0); s += out[c.k]; }
  for (const c of CRITERIA) out[c.k] /= s;
  return out;
}
async function fetchJSON(url, opts = {}, timeout = 8000) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), timeout);
  try {
    const r = await fetch(url, { ...opts, signal: ctl.signal });
    if (!r.ok) throw new Error("HTTP " + r.status);
    return await r.json();
  } finally { clearTimeout(t); }
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/* Journal */
function log(agent, msg, cls = "") {
  const el = document.createElement("div");
  el.className = agent;
  const now = new Date();
  const names = { pref: "Préférences", env: "Environnement", dec: "Décision", sys: "Système" };
  el.innerHTML = `<time>${now.toLocaleTimeString("fr-FR")}</time><span class="${cls}"><em>${names[agent]}</em> · ${msg}</span>`;
  $("#log").prepend(el);
}
function agentState(id, st, out) {
  const el = $("#ag-" + id);
  el.classList.remove("work", "done", "err");
  if (st) el.classList.add(st);
  el.querySelector(".st").textContent = { work: "en cours…", done: "terminé", err: "repli" }[st] || "en attente";
  if (out !== undefined) el.querySelector(".out").textContent = out;
}
function chip(id, cls, text) {
  const el = $("#st-" + id);
  el.className = "chip " + cls;
  el.innerHTML = `<i></i>${text}`;
}

/* ------------------------------------------------------------------ */
/* Carte                                                               */
/* ------------------------------------------------------------------ */
const map = L.map("map", { zoomControl: true }).setView(CENTER, 13);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 19, attribution: "© contributeurs OpenStreetMap",
}).addTo(map);
const layers = L.layerGroup().addTo(map);

function drawResult() {
  layers.clearLayers();
  const r = state.result; if (!r) return;
  const { o, d } = state.env;
  const opt = r.options.find((x) => x.id === (state.selected || r.recId));
  if (opt) {
    for (const seg of opt.segments) {
      L.polyline(seg.coords, {
        color: MODES[opt.id].color, weight: seg.kind === "walk" ? 4 : 6,
        dashArray: seg.kind === "walk" ? "4 8" : seg.estimated ? "12 8" : null, opacity: 0.9,
      }).addTo(layers);
    }
    for (const st of opt.stations || []) {
      L.circleMarker([st.lat, st.lon], { radius: 7, color: "#1f7ab8", fillColor: "#fff", fillOpacity: 1, weight: 3 })
        .bindTooltip(`Station V³ ${st.name} : ${st.bikes} vélos, ${st.docks} places`).addTo(layers);
    }
  }
  const mk = (p, c, t) => L.circleMarker([p.lat, p.lon], { radius: 9, color: "#fff", weight: 3, fillColor: c, fillOpacity: 1 }).bindTooltip(t).addTo(layers);
  mk(o, "#0f2a4a", "Départ : " + o.name);
  mk(d, "#0e8f80", "Arrivée : " + d.name);
  map.fitBounds(L.latLngBounds([[o.lat, o.lon], [d.lat, d.lon]]).pad(0.35));
}

/* ------------------------------------------------------------------ */
/* Agent 1 — Préférences                                               */
/* ------------------------------------------------------------------ */
async function llm(task, body) {
  const ctl = new AbortController();
  const t = setTimeout(() => ctl.abort(), 20000);
  let r, res;
  try {
    r = await fetch("/api/agent", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, ...body }), signal: ctl.signal,
    });
    res = await r.json().catch(() => ({}));
  } finally { clearTimeout(t); }
  if (!r.ok || !res.ok) throw new Error(`HTTP ${r.status}${res.error ? " — " + res.error : ""}`);
  return res;
}

function matchPlaces(text) {
  const t = norm(text), found = [];
  for (const p of PLACES) {
    for (const a of p.a) {
      const i = t.indexOf(norm(a));
      if (i >= 0) { found.push({ place: p, i, len: a.length }); break; }
    }
  }
  // supprime les correspondances incluses dans une plus longue (ex. « gare » dans « gare saint-jean »)
  return found.filter((f) => !found.some((g) => g !== f && g.place !== f.place && g.i <= f.i && g.i + g.len >= f.i + f.len && g.len > f.len))
    .sort((a, b) => a.i - b.i);
}

function ruleParse(text) {
  const t = norm(text);
  const has = (...w) => w.some((x) => t.includes(norm(x)));
  const w = { ...DEFAULT_W };
  if (has("presse", "en retard", "vite", "rapide", "trop tard", "urgent")) w.temps = 0.9;
  if (has("secur", "grands axes", "calme", "tranquille", "danger")) w.securite = 0.9;
  if (has("pluie", "pleut", "confort", "fatigue")) w.confort = 0.75;
  if (has("pas cher", "moins cher", "budget", "econom", "gratuit")) w.cout = 0.9;
  if (has("ecolo", "co2", "planete", "vert", "environnement")) w.environnement = 0.9;
  if (has("fatigue", "valise", "bagage", "pas envie de pedaler", "sans effort", "fauteuil")) w.effort = 0.8;
  const c = {
    pas_de_velo: has("pas de velo", "sans velo", "pas a velo", "pas en velo"),
    pmr: has("fauteuil", "mobilite reduite", "pmr", "poussette", "handicap"),
    eviter_grands_axes: has("grands axes", "grand axe", "calme"),
    bagages: has("valise", "bagage", "courses", "sac lourd"),
    velo_perso: has("mon velo", "avec mon velo", "mon propre velo"),
  };
  // origine / destination à partir des lieux reconnus et des prépositions
  const m = matchPlaces(text);
  let origine = null, destination = null;
  for (const f of m) {
    const before = t.slice(Math.max(0, f.i - 14), f.i);
    if (/(depuis|\bde\b|\bdu\b|\bdes\b|d'|part)\s*(la |le |l')?$/.test(before) && !origine) origine = f.place.n;
    else if (/(\ba\b|\bau\b|vers|jusqu|aller|→|->)\s*(la |le |l')?$/.test(before) && !destination) destination = f.place.n;
  }
  const rest = m.map((f) => f.place.n).filter((n) => n !== origine && n !== destination);
  if (!origine && !destination && rest.length >= 2) { origine = rest[0]; destination = rest[1]; }
  else if (!destination && rest.length) destination = rest[rest.length - 1];
  else if (!origine && rest.length) origine = rest[0];
  const h = t.match(/\b(\d{1,2})\s*h\s*(\d{2})?\b/);
  return {
    origine, destination,
    heure: h ? `${h[1].padStart(2, "0")}:${h[2] || "00"}` : null,
    poids: w, contraintes: c,
    justification: "Interprétation par règles lexicales (mode sans LLM).",
  };
}

async function agentPreferences(text) {
  agentState("pref", "work", "Lecture de la demande…");
  let out, source;
  if (state.llm !== false) {
    try {
      const res = await llm("parse", { text });
      out = res.data; source = res.model;
      state.llm = true; chip("llm", "ok", "LLM : " + res.model);
    } catch (e) {
      state.llm = false; chip("llm", "warn", "LLM indisponible : mode règles");
      log("pref", "LLM indisponible (" + e.message + ") : bascule sur l'analyse par règles.", "warn");
    }
  }
  if (!out) { out = ruleParse(text); source = "règles"; }
  // Complète avec les règles si le LLM a omis un lieu
  if (!out.origine || !out.destination) {
    const r = ruleParse(text);
    out.origine = out.origine || r.origine;
    out.destination = out.destination || r.destination;
  }
  out.poids = { ...DEFAULT_W, ...(out.poids || {}) };
  out.contraintes = { pas_de_velo: false, pmr: false, eviter_grands_axes: false, bagages: false, velo_perso: false, ...(out.contraintes || {}) };
  out.source = source;
  const cs = Object.entries(out.contraintes).filter(([, v]) => v).map(([k]) => k.replace(/_/g, " "));
  const top = Object.entries(normalizeW(out.poids)).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([k]) => labelOf(k).toLowerCase());
  agentState("pref", "done", `${out.origine || "?"} → ${out.destination || "?"} · priorités : ${top.join(", ")}${cs.length ? " · contraintes : " + cs.join(", ") : ""}`);
  log("pref", `(${source}) ${out.justification || ""}`);
  return out;
}
const labelOf = (k) => CRITERIA.find((c) => c.k === k)?.label || k;

/* ------------------------------------------------------------------ */
/* Agent 2 — Environnement                                             */
/* ------------------------------------------------------------------ */
async function geocode(q) {
  if (!q) return null;
  const m = matchPlaces(q);
  if (m.length) return { name: m[0].place.n, lat: m[0].place.lat, lon: m[0].place.lon, src: "référentiel" };
  const urls = [
    `https://data.geopf.fr/geocodage/search?q=${encodeURIComponent(q)}&limit=1&lat=${CENTER[0]}&lon=${CENTER[1]}`,
    `https://api-adresse.data.gouv.fr/search/?q=${encodeURIComponent(q)}&limit=1&lat=${CENTER[0]}&lon=${CENTER[1]}`,
  ];
  for (const u of urls) {
    try {
      const j = await fetchJSON(u, {}, 6000);
      const f = j.features?.[0];
      if (!f) continue;
      const p = { name: f.properties.label, lat: f.geometry.coordinates[1], lon: f.geometry.coordinates[0], src: "Géoplateforme" };
      if (haversine(p, { lat: CENTER[0], lon: CENTER[1] }) < 30) return p;
    } catch { /* source suivante */ }
  }
  return null;
}

async function getWeather() {
  try {
    const j = await fetchJSON(
      `https://api.open-meteo.com/v1/forecast?latitude=${CENTER[0]}&longitude=${CENTER[1]}` +
      `&current=temperature_2m,precipitation,rain,wind_speed_10m,is_day,weather_code` +
      `&hourly=precipitation_probability&forecast_hours=3&timezone=Europe%2FParis`, {}, 7000);
    const c = j.current;
    const pp = Math.max(...(j.hourly?.precipitation_probability || [0]));
    chip("meteo", "ok", `Météo : ${fmt(c.temperature_2m, 0)} °C, ${fmt(c.precipitation, 1)} mm/h`);
    return { temp: c.temperature_2m, rain: c.precipitation, wind: c.wind_speed_10m, isDay: !!c.is_day, rainProb: pp, live: true };
  } catch (e) {
    chip("meteo", "warn", "Météo : valeurs par défaut");
    return { temp: 17, rain: 0, wind: 12, isDay: true, rainProb: 20, live: false };
  }
}

function pick(obj, keys) { for (const k of keys) if (obj[k] !== undefined && obj[k] !== null) return obj[k]; }
async function getStations() {
  if (state.stationsCache && Date.now() - state.stationsCache.t < 60000) return state.stationsCache.v;
  try {
    const j = await fetchJSON("https://datahub.bordeaux-metropole.fr/api/explore/v2.1/catalog/datasets/ci_vcub_p/exports/json", {}, 15000);
    const arr = (Array.isArray(j) ? j : j.results || []).map((r) => {
      const g = r.geo_point_2d || r.geom?.geometry?.coordinates && { lon: r.geom.geometry.coordinates[0], lat: r.geom.geometry.coordinates[1] } || {};
      return {
        name: pick(r, ["nom", "name", "libelle"]) || "station",
        lat: Number(g.lat), lon: Number(g.lon),
        bikes: Number(pick(r, ["nbvelos", "nb_velos", "velos", "nbvelo"]) ?? 0),
        docks: Number(pick(r, ["nbplaces", "nb_places", "places"]) ?? 0),
        ok: !/(deconnect|maintenance|ferm)/i.test(String(pick(r, ["etat", "state"]) || "")),
      };
    }).filter((s) => isFinite(s.lat) && isFinite(s.lon) && s.lat > 40);
    if (!arr.length) throw new Error("format inattendu");
    const v = { list: arr, live: true };
    state.stationsCache = { t: Date.now(), v };
    chip("v3", "ok", `V³ temps réel : ${arr.length} stations`);
    return v;
  } catch (e) {
    chip("v3", "warn", "V³ : indisponible, hypothèses");
    return { list: [], live: false };
  }
}

async function osrm(profile, a, b) {
  const u = `https://routing.openstreetmap.de/routed-${profile}/route/v1/driving/${a.lon},${a.lat};${b.lon},${b.lat}?overview=full&geometries=geojson`;
  const j = await fetchJSON(u, {}, 7000);
  const r = j.routes?.[0]; if (!r) throw new Error("pas d'itinéraire");
  return { km: r.distance / 1000, coords: r.geometry.coordinates.map(([x, y]) => [y, x]) };
}

async function agentEnvironment(req) {
  agentState("env", "work", "Géocodage, météo, V³, routage…");
  const origQ = req.origine || "Place de la Victoire";
  if (!req.origine) log("env", "Départ non précisé : Place de la Victoire retenue par défaut.", "warn");
  const [o, d, weather, stations] = await Promise.all([geocode(origQ), geocode(req.destination), getWeather(), getStations()]);
  if (!o || !d) {
    agentState("env", "err", "Lieu introuvable");
    throw new Error(`Lieu non reconnu : ${!o ? origQ : req.destination || "(destination absente)"}`);
  }
  const [walk, bike] = await Promise.all([
    osrm("foot", o, d).catch(() => null),
    osrm("bike", o, d).catch(() => null),
  ]);
  chip("route", walk || bike ? "ok" : "warn", walk || bike ? "Routage : OSM" : "Routage : estimation");
  const env = { o, d, weather, stations, walk, bike, crow: haversine(o, d) };
  log("env", `${o.name} → ${d.name} : ${fmt(env.crow, 1)} km à vol d'oiseau. Météo ${weather.live ? "temps réel" : "par défaut"}` +
    ` (${fmt(weather.rain, 1)} mm/h, vent ${fmt(weather.wind, 0)} km/h). V³ ${stations.live ? stations.list.length + " stations" : "indisponible"}.` +
    ` Routage ${walk || bike ? "OSM" : "estimé"}.`);
  agentState("env", "done", `${fmt(env.crow, 1)} km · ${fmt(weather.temp, 0)} °C · pluie ${fmt(weather.rain, 1)} mm/h · ${stations.live ? stations.list.length + " stations V³" : "V³ hors ligne"}`);
  return env;
}

/* Contexte effectif = données observées + événements simulés */
function effectiveContext(req) {
  const w = { ...state.env.weather };
  if (state.events.has("rain")) { w.rain = Math.max(w.rain, 4); w.rainProb = 100; }
  let hour = new Date().getHours();
  if (req.heure) hour = parseInt(req.heure, 10);
  if (state.events.has("night")) hour = 22;
  const night = hour >= 21 || hour < 6;
  const peak = (hour >= 7 && hour < 9) || (hour >= 17 && hour < 19);
  return { ...w, hour, night, peak, tramDisrupted: state.events.has("tram"), v3Empty: state.events.has("v3empty") };
}

/* ------------------------------------------------------------------ */
/* Agent 3 — Décision et coordination                                  */
/* ------------------------------------------------------------------ */
function contextWeights(base, ctx, c) {
  const w = { ...base }, why = [];
  if (ctx.rain >= 0.3) { w.securite *= 1.3; w.confort *= 1.4; why.push("pluie : sécurité et confort renforcés"); }
  if (ctx.night) { w.securite *= 1.3; why.push("nuit : sécurité renforcée"); }
  if (ctx.wind >= 35) { w.securite *= 1.2; why.push("vent fort : sécurité renforcée"); }
  if (ctx.temp >= 30) { w.effort *= 1.3; why.push("chaleur : effort pénalisé"); }
  if (ctx.peak) { w.temps *= 1.2; why.push("heure de pointe : temps renforcé"); }
  if (c.eviter_grands_axes) { w.securite *= 1.2; why.push("éviter les grands axes"); }
  return { w: normalizeW(w), why };
}

function nearestStation(list, p, need) {
  let best = null;
  for (const s of list) {
    if (!s.ok || s[need] < 1) continue;
    const dist = haversine(p, s);
    if (dist <= V3_RADIUS_KM && (!best || dist < best.dist)) best = { ...s, dist };
  }
  return best;
}

function buildOptions(req, ctx) {
  const { o, d, walk, bike, stations, crow } = state.env;
  const c = req.contraintes;
  const walkKm = walk?.km ?? crow * 1.3;
  const bikeKm = bike?.km ?? crow * 1.3;
  const line = (a, b) => [[a.lat, a.lon], [b.lat, b.lon]];
  const opts = [];

  // Marche
  opts.push({ id: "marche", time: walkKm / MODES.marche.speed * 60, km: walkKm, walkKm, bikeKm: 0,
    segments: [{ kind: "walk", coords: walk?.coords || line(o, d), estimated: !walk }], conf: walk ? 0.95 : 0.7, notes: [] });

  // Vélo personnel ou V³, direct et calme
  const bikeIds = c.velo_perso ? ["velo", "velo_calme"] : ["v3", "v3_calme"];
  for (const id of bikeIds) {
    const m = MODES[id];
    const km = bikeKm * (m.detour || 1);
    const o2 = { id, km, bikeKm: km, walkKm: 0, segments: [], conf: bike ? 0.9 : 0.7, notes: [], stations: [] };
    if (m.shared) {
      let s1 = null, s2 = null;
      if (stations.live) {
        s1 = nearestStation(stations.list, o, "bikes");
        s2 = nearestStation(stations.list, d, "docks");
        if (s1 && ctx.v3Empty) { o2.notes.push(`station ${s1.name} vidée (événement)`); s1 = null; }
        if (!s1) { o2.excluded = "aucune station V³ avec un vélo disponible à moins de 600 m du départ"; }
        else if (!s2) { o2.excluded = "aucune station V³ avec une place libre à moins de 600 m de l'arrivée"; }
      } else {
        o2.conf = 0.45; o2.notes.push("disponibilité V³ non vérifiée");
        if (ctx.v3Empty) o2.excluded = "station V³ de départ vidée (événement simulé)";
      }
      const a1 = s1 ? s1.dist * 1.25 : 0.25, a2 = s2 ? s2.dist * 1.25 : 0.25;
      o2.walkKm = a1 + a2;
      o2.time = o2.walkKm / 4.8 * 60 + km / m.speed * 60 + 2;
      if (s1 && s2) {
        o2.stations = [s1, s2];
        o2.segments = [
          { kind: "walk", coords: line(o, s1) },
          { kind: "ride", coords: m.detour ? line(s1, s2) : (bike?.coords || line(s1, s2)), estimated: !!m.detour || !bike },
          { kind: "walk", coords: line(s2, d) },
        ];
        o2.notes.push(`${s1.bikes} vélos à ${s1.name}, ${s2.docks} places à ${s2.name}`);
      } else {
        o2.segments = [{ kind: "ride", coords: bike?.coords || line(o, d), estimated: true }];
      }
    } else {
      o2.time = km / m.speed * 60 + 1;
      o2.segments = [{ kind: "ride", coords: m.detour ? line(o, d) : (bike?.coords || line(o, d)), estimated: !!m.detour || !bike }];
    }
    if (m.detour) o2.notes.push("détour estimé +18 % par des voies apaisées");
    opts.push(o2);
  }

  // Trottinette
  opts.push({ id: "trottinette", km: bikeKm, walkKm: 0.3, bikeKm, time: 0.3 / 4.8 * 60 + bikeKm / MODES.trottinette.speed * 60 + 1,
    segments: [{ kind: "ride", coords: bike?.coords || line(o, d), estimated: !bike }], conf: 0.55,
    notes: ["disponibilité des trottinettes non connectée (hypothèse : un engin à 150 m)"] });

  // Transport en commun (estimation)
  const inVeh = crow * 1.35 / MODES.tc.speed * 60;
  let tcTime = 10 + (ctx.peak ? 5 : 7) + inVeh;
  const tcNotes = ["estimation : 2 × 400 m à pied, attente et vitesse commerciale moyenne (GTFS non intégré)"];
  if (ctx.tramDisrupted) { tcTime += 15; tcNotes.push("perturbation : +15 min"); }
  opts.push({ id: "tc", km: crow * 1.35, walkKm: 0.8, bikeKm: 0, time: tcTime,
    segments: [{ kind: "ride", coords: line(o, d), estimated: true }], conf: ctx.tramDisrupted ? 0.4 : 0.6, notes: tcNotes });

  // Contraintes dures
  for (const x of opts) {
    const m = MODES[x.id];
    if (c.pmr && (m.bike || m.scooter)) x.excluded = x.excluded || "incompatible avec une mobilité réduite";
    if (c.pmr && x.id === "marche" && walkKm > 1) x.excluded = x.excluded || "plus de 1 km à pied avec une mobilité réduite";
    if (c.pas_de_velo && m.bike) x.excluded = x.excluded || "vous avez exclu le vélo";
    if (x.id === "marche" && walkKm > 8) x.excluded = x.excluded || "plus de 8 km à pied";
    if (x.id === "tc" && crow < 0.6) x.excluded = x.excluded || "trajet trop court pour les transports en commun";
  }
  return opts;
}

function criteriaScores(x, ctx, c, tmin) {
  const m = MODES[x.id];
  const rainF = clamp(ctx.rain / 3);
  let safe = m.safe;
  if (m.bike) safe -= 0.22 * rainF + (ctx.wind >= 40 ? 0.15 : 0);
  if (m.scooter) safe -= 0.32 * rainF + (ctx.wind >= 40 ? 0.15 : 0);
  if (ctx.night) safe -= m.scooter ? 0.15 : m.exposed ? 0.1 : 0.05;
  if (c.eviter_grands_axes && (m.bike || m.scooter) && !m.detour) safe -= 0.08;

  let comfort = m.comfort;
  if (m.exposed) comfort -= 0.45 * rainF + (ctx.temp < 5 ? 0.1 : 0);
  if (x.id === "marche") comfort -= Math.max(0, x.walkKm - 1.5) * 0.12;
  if (x.id === "tc" && ctx.peak) comfort -= 0.25;
  if (x.id === "tc" && ctx.tramDisrupted) comfort -= 0.2;
  if (c.bagages) comfort -= m.bike || m.scooter ? 0.4 : x.id === "marche" ? 0.2 : 0.05;
  if ((m.bike || m.scooter) && ctx.wind >= 30) comfort -= 0.15;

  const cost = { marche: 1, velo: 1, velo_calme: 1, v3: 0.75, v3_calme: 0.72, tc: 0.6, trottinette: Math.max(0.1, 0.5 - 0.015 * x.time) }[x.id];
  const envi = 1 - m.co2 / 60;
  let effort = 1 - Math.min(1, x.walkKm / 3) * 0.9 - (m.bike ? Math.min(0.5, x.bikeKm / 12) + 0.05 : 0);
  if (m.exposed && ctx.temp >= 28) effort -= 0.15;
  if (m.scooter) effort = 0.85;

  return { temps: clamp(tmin / x.time), securite: clamp(safe), confort: clamp(comfort), cout: clamp(cost), environnement: clamp(envi), effort: clamp(effort) };
}

function decide(req, isReplan) {
  const ctx = effectiveContext(req);
  const c = req.contraintes;
  // Profil : poids de la requête, combinés au profil appris si l'usager a déjà donné des retours
  const qW = normalizeW(req.poids);
  const base = state.feedbacks ? normalizeW(Object.fromEntries(CRITERIA.map(({ k }) => [k, 0.5 * qW[k] + 0.5 * state.profile[k]]))) : qW;
  const { w, why } = contextWeights(base, ctx, c);

  const opts = buildOptions(req, ctx);
  const feasible = opts.filter((x) => !x.excluded);
  const tmin = Math.min(...feasible.map((x) => x.time));
  for (const x of opts) {
    x.s = criteriaScores(x, ctx, c, isFinite(tmin) ? tmin : x.time);
    x.raw = CRITERIA.reduce((acc, { k }) => acc + w[k] * x.s[k], 0);
    x.gate = 1 / (1 + Math.exp(-GATE_K * (x.s.securite - GATE_C)));
    x.score = x.raw * x.gate * (0.9 + 0.1 * x.conf);
    if (x.gate < 0.5 && !x.excluded) x.gated = true;
  }
  const ranked = opts.filter((x) => !x.excluded).sort((a, b) => b.score - a.score);
  const excluded = opts.filter((x) => x.excluded);
  if (!ranked.length) throw new Error("Aucune option ne respecte vos contraintes.");
  let rec = ranked[0];
  let hyst = null;
  if (isReplan && state.currentRec) {
    const cur = ranked.find((x) => x.id === state.currentRec);
    if (cur && cur !== rec) {
      const gap = rec.score - cur.score;
      if (gap < HYSTERESIS) { hyst = { kept: cur.id, gap }; rec = cur; }
      else hyst = { switched: true, from: state.currentRec, gap };
    } else if (!cur) hyst = { lost: state.currentRec };
  }
  const fastest = ranked.slice().sort((a, b) => a.time - b.time)[0];
  return { ctx, w, qW, why, options: [...ranked, ...excluded], recId: rec.id, fastestId: fastest.id, hyst };
}

async function agentDecision(req, isReplan = false) {
  agentState("dec", "work", "Arbitrage multicritère…");
  await sleep(250);
  const r = decide(req, isReplan);
  state.result = r; state.selected = null;
  const rec = r.options.find((x) => x.id === r.recId);
  if (r.why.length) log("dec", "Poids adaptés au contexte : " + r.why.join(" ; ") + ".");
  const exc = r.options.filter((x) => x.excluded);
  if (exc.length) log("dec", "Options écartées (contraintes non compensables) : " + exc.map((x) => `${MODES[x.id].label} (${x.excluded})`).join(" ; ") + ".");
  const gated = r.options.filter((x) => x.gated);
  if (gated.length) log("dec", "Facteur de sécurité : " + gated.map((x) => `${MODES[x.id].label} (S = ${fmt(x.s.securite)})`).join(", ") + " fortement pénalisée(s).", "warn");
  if (r.hyst?.kept) log("dec", `Hystérésis : meilleure alternative à +${fmt(r.hyst.gap, 3)} < ${fmt(HYSTERESIS)} → maintien de « ${MODES[r.hyst.kept].label} » pour éviter une bascule instable.`);
  if (r.hyst?.switched) log("dec", `Re-planification : bascule de « ${MODES[r.hyst.from].label} » vers « ${MODES[rec.id].label} » (écart ${fmt(r.hyst.gap, 3)} ≥ ${fmt(HYSTERESIS)}).`);
  if (r.hyst?.lost) log("dec", `Re-planification : « ${MODES[r.hyst.lost].label} » n'est plus réalisable, nouvelle recommandation.`, "warn");
  log("dec", `Recommandation : ${MODES[rec.id].label}, ${Math.round(rec.time)} min, score ${fmt(rec.score)}.`);
  state.currentRec = rec.id;
  agentState("dec", "done", `${MODES[rec.id].label} · ${Math.round(rec.time)} min · score ${fmt(rec.score)}`);
  renderAll();
  explain(req, r);
}

/* Explication : LLM si disponible, sinon gabarit déterministe */
function templateExplain(req, r) {
  const rec = r.options.find((x) => x.id === r.recId);
  const fast = r.options.find((x) => x.id === r.fastestId);
  const top = CRITERIA.map(({ k }) => [k, r.w[k]]).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([k]) => labelOf(k).toLowerCase());
  let s = `Je recommande « ${MODES[rec.id].label} » (${Math.round(rec.time)} min), car vos priorités retenues sont ${top.join(" et ")}`;
  s += r.why.length ? ` (poids ajustés au contexte : ${r.why.join(" ; ")}).` : ".";
  if (fast.id !== rec.id) {
    const dt = Math.round(rec.time - fast.time);
    const gains = CRITERIA.filter(({ k }) => rec.s[k] - fast.s[k] > 0.08).map(({ label }) => label.toLowerCase());
    s += ` Le trajet le plus rapide (« ${MODES[fast.id].label} », ${Math.round(fast.time)} min) est ${dt > 0 ? dt + " min plus court" : "équivalent en temps"}, mais moins bon en ${gains.join(", ") || "score global"}.`;
  } else s += " C'est aussi l'option la plus rapide.";
  const exc = r.options.filter((x) => x.excluded);
  if (exc.length) s += ` Écartées : ${exc.map((x) => `${MODES[x.id].label.toLowerCase()} (${x.excluded})`).join(" ; ")}.`;
  return s;
}

async function explain(req, r) {
  const box = $("#explain"); box.hidden = false;
  const rec = r.options.find((x) => x.id === r.recId);
  const fast = r.options.find((x) => x.id === r.fastestId);
  $("#baseline").innerHTML =
    `<div>Agent : <b>${MODES[rec.id].label}</b> · ${Math.round(rec.time)} min · score ${fmt(rec.score)}</div>` +
    `<div>Référence « plus rapide » : <b>${MODES[fast.id].label}</b> · ${Math.round(fast.time)} min · score ${fmt(fast.score)}</div>`;
  $("#explain-text").textContent = templateExplain(req, r);
  $("#explain-src").textContent = "Explication générée par gabarit à partir des scores.";
  if (state.llm) {
    try {
      const payload = {
        demande: $("#q").value,
        contexte: { pluie_mm_h: r.ctx.rain, vent_kmh: r.ctx.wind, temperature: r.ctx.temp, nuit: r.ctx.night, heure_de_pointe: r.ctx.peak, evenements: [...state.events] },
        poids: Object.fromEntries(Object.entries(r.w).map(([k, v]) => [k, +v.toFixed(2)])),
        recommandation: MODES[rec.id].label,
        plus_rapide: MODES[fast.id].label,
        options: r.options.map((x) => ({ mode: MODES[x.id].label, minutes: Math.round(x.time), score: +x.score.toFixed(2),
          securite: +x.s.securite.toFixed(2), confort: +x.s.confort.toFixed(2), exclue: x.excluded || null })),
      };
      const res = await llm("explain", { payload });
      if (state.result === r) {
        $("#explain-text").textContent = res.text;
        $("#explain-src").textContent = `Explication rédigée par l'agent de décision (${res.model}) à partir des scores calculés par le système.`;
      }
    } catch { /* on garde le gabarit */ }
  }
}

/* ------------------------------------------------------------------ */
/* Apprentissage des préférences (retour usager)                       */
/* ------------------------------------------------------------------ */
function learnFrom(chosenId) {
  const r = state.result;
  const ch = r.options.find((x) => x.id === chosenId), rec = r.options.find((x) => x.id === r.recId);
  const before = { ...state.profile };
  const p = { ...state.profile };
  for (const { k } of CRITERIA) p[k] = Math.max(0.02, p[k] + LEARNING_RATE * (ch.s[k] - rec.s[k]));
  state.profile = normalizeW(p);
  state.feedbacks++;
  const moved = CRITERIA.map(({ k, label }) => [label, state.profile[k] - normalizeW(before)[k]])
    .filter(([, d]) => Math.abs(d) > 0.01).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 3)
    .map(([l, d]) => `${l} ${d > 0 ? "+" : "−"}${fmt(Math.abs(d))}`);
  log("pref", `Retour usager : « ${MODES[ch.id].label} » préférée à « ${MODES[rec.id].label} ». Profil mis à jour (${moved.join(", ") || "variation faible"}).`);
  state.currentRec = null; // la préférence explicite prime sur l'hystérésis
  agentDecision(state.request, false);
}

/* ------------------------------------------------------------------ */
/* Rendu                                                               */
/* ------------------------------------------------------------------ */
function renderWeights() {
  const r = state.result;
  const w = r ? r.w : normalizeW(state.profile);
  const q = r ? r.qW : null;
  $("#weights").innerHTML = CRITERIA.map(({ k, label }) =>
    `<span>${label}</span><div class="bar"><b style="width:${(w[k] * 100 * 2.5).toFixed(1)}%"></b>${q ? `<s style="left:${Math.min(99, q[k] * 250).toFixed(1)}%"></s>` : ""}</div><span class="v">${fmt(w[k])}</span>`
  ).join("");
  $("#fbcount").textContent = `${state.feedbacks} retour${state.feedbacks > 1 ? "s" : ""} appris`;
}

function confTag(c) {
  return c >= 0.8 ? '<span class="tag conf-h">confiance haute</span>' : c >= 0.55 ? '<span class="tag conf-m">confiance moyenne</span>' : '<span class="tag conf-l">confiance faible</span>';
}

function renderOptions() {
  const r = state.result;
  const box = $("#opts");
  if (!r) return;
  box.innerHTML = "";
  for (const x of r.options) {
    const m = MODES[x.id];
    const el = document.createElement("div");
    el.className = "opt" + (x.id === r.recId ? " rec" : "") + (x.excluded ? " excl" : "") + (state.selected === x.id ? " sel" : "");
    const tags = [
      x.id === r.recId ? '<span class="tag rec">recommandé</span>' : "",
      x.id === r.fastestId && !x.excluded ? '<span class="tag fast">plus rapide</span>' : "",
      x.excluded ? '<span class="tag x">écartée</span>' : confTag(x.conf),
    ].join("");
    el.innerHTML = `
      <div class="top">
        <div>
          <div class="name"><span class="dot" style="background:${m.color}"></span>${m.label} ${tags}</div>
          <div class="meta">${Math.round(x.time)} min · ${fmt(x.km, 1)} km${x.walkKm && x.id !== "marche" ? ` · dont ${Math.round(x.walkKm * 1000)} m à pied` : ""}</div>
        </div>
        <div class="score">${x.excluded ? "<b>—</b>" : `<b>${fmt(x.score)}</b>`}<small>${x.excluded ? "non éligible" : `f(S) = ${fmt(x.gate)}`}</small></div>
      </div>
      <div class="crit">${CRITERIA.map(({ k, label }) => `<div>${label}<div class="b"><i style="width:${(x.s[k] * 100).toFixed(0)}%"></i></div></div>`).join("")}</div>
      <div class="why">${x.excluded ? `<span class="gate">Contrainte non compensable : ${x.excluded}.</span>` : ""}${x.gated ? `<span class="gate">Sécurité S = ${fmt(x.s.securite)} sous le seuil : score effondré. </span>` : ""}${x.notes.length ? x.notes.join(" · ") : ""}</div>
      ${!x.excluded && x.id !== r.recId ? `<button class="btn ghost prefer" data-id="${x.id}">Je préfère celle-ci</button>` : ""}`;
    if (!x.excluded) el.addEventListener("click", (e) => {
      if (e.target.closest(".prefer")) return;
      state.selected = x.id; renderOptions(); drawResult();
    });
    const b = el.querySelector(".prefer");
    if (b) b.addEventListener("click", () => learnFrom(x.id));
    box.appendChild(el);
  }
}

function renderAll() { renderWeights(); renderOptions(); drawResult(); }

/* ------------------------------------------------------------------ */
/* Orchestration                                                       */
/* ------------------------------------------------------------------ */
async function run() {
  const text = $("#q").value.trim();
  if (!text) return;
  const btn = $("#go"); btn.disabled = true;
  state.events.clear(); document.querySelectorAll(".ev").forEach((b) => b.classList.remove("on"));
  state.currentRec = null;
  ["pref", "env", "dec"].forEach((a) => agentState(a, null, ""));
  const t0 = performance.now();
  log("sys", "Nouvelle demande reçue.");
  try {
    state.request = await agentPreferences(text);
    state.env = await agentEnvironment(state.request);
    await agentDecision(state.request, false);
    document.querySelectorAll(".ev").forEach((b) => (b.disabled = false));
    $("#runinfo").textContent = `${fmt((performance.now() - t0) / 1000, 1)} s`;
  } catch (e) {
    log("sys", e.message, "warn");
    $("#opts").innerHTML = `<div class="empty">${e.message}<br>Lieux reconnus : ${PLACES.map((p) => p.n).join(", ")} — ou une adresse de Bordeaux Métropole.</div>`;
    ["pref", "env", "dec"].forEach((a) => { if ($("#ag-" + a).classList.contains("work")) agentState(a, "err"); });
  } finally { btn.disabled = false; }
}

async function toggleEvent(ev, btn) {
  if (!state.request) return;
  if (state.events.has(ev)) { state.events.delete(ev); btn.classList.remove("on"); }
  else { state.events.add(ev); btn.classList.add("on"); }
  const labels = { rain: "averse soudaine", v3empty: "station V³ de départ vidée", tram: "perturbation tram (+15 min)", night: "départ à 22 h" };
  log("env", `Événement ${state.events.has(ev) ? "détecté" : "terminé"} : ${labels[ev]}. Contexte mis à jour.`, "warn");
  agentState("env", "done", `Événements actifs : ${[...state.events].map((e) => labels[e]).join(", ") || "aucun"}`);
  await agentDecision(state.request, true);
}

/* ------------------------------------------------------------------ */
/* Initialisation                                                      */
/* ------------------------------------------------------------------ */
$("#examples").innerHTML = EXAMPLES.map((e, i) => `<button class="ex" data-i="${i}">${e.length > 46 ? e.slice(0, 44) + "…" : e}</button>`).join("");
$("#examples").addEventListener("click", (e) => {
  const b = e.target.closest(".ex"); if (!b) return;
  $("#q").value = EXAMPLES[+b.dataset.i]; run();
});
$("#go").addEventListener("click", run);
$("#q").addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run(); });
document.querySelectorAll(".ev").forEach((b) => b.addEventListener("click", () => toggleEvent(b.dataset.ev, b)));
$("#reset").addEventListener("click", () => {
  state.profile = { ...DEFAULT_W }; state.feedbacks = 0;
  log("pref", "Profil appris réinitialisé.");
  if (state.request) agentDecision(state.request, false); else renderWeights();
});
renderWeights();
log("sys", "Prototype prêt. Les agents interrogent des données ouvertes réelles ; chaque source a un repli explicite.");
run();
