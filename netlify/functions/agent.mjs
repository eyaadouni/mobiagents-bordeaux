// Fonction serverless Netlify : relais vers un LLM compatible OpenAI (Groq par défaut).
// La clé reste côté serveur (variable d'environnement LLM_API_KEY), jamais dans le navigateur.
//
// Variables d'environnement :
//   LLM_API_KEY   (obligatoire)  clé Groq (gratuite) ou autre fournisseur compatible OpenAI
//   LLM_BASE_URL  (optionnel)    défaut : https://api.groq.com/openai/v1
//   LLM_MODEL     (optionnel)    défaut : llama-3.3-70b-versatile

const BASE_URL = process.env.LLM_BASE_URL || "https://api.groq.com/openai/v1";
const MODEL = process.env.LLM_MODEL || "llama-3.3-70b-versatile";

const PARSE_PROMPT = `Tu es l'« Agent Préférences » d'un système multi-agents de recommandation de trajets à Bordeaux.
À partir de la demande d'un usager (en langage naturel), tu extrais ses préférences sous forme structurée.
Réponds UNIQUEMENT avec un objet JSON, sans texte autour, au format exact :
{
  "origine": string | null,          // lieu de départ tel qu'écrit (ex. "Gare Saint-Jean"), null si absent
  "destination": string | null,      // lieu d'arrivée tel qu'écrit
  "heure": string | null,            // "HH:MM" si l'usager indique une heure de départ, sinon null
  "poids": {                         // importance relative de chaque critère, entre 0 et 1
    "temps": number, "securite": number, "confort": number,
    "cout": number, "environnement": number, "effort": number
  },
  "contraintes": {                   // contraintes dures (non compensables)
    "pas_de_velo": boolean,          // refuse explicitement le vélo
    "pmr": boolean,                  // fauteuil roulant, mobilité réduite, poussette
    "eviter_grands_axes": boolean,
    "bagages": boolean,              // valise, sac lourd, courses
    "velo_perso": boolean            // possède / utilise son propre vélo
  },
  "justification": string            // une phrase en français expliquant l'interprétation
}
Règles : un critère non évoqué reçoit 0.3 ; un critère explicitement demandé reçoit 0.7 à 1 ;
« pressé », « en retard » => temps élevé ; « sécurité », « calme », « éviter les grands axes » => securite élevé ;
« pluie », « fatigué(e) », « tranquille » => confort ; « pas cher », « budget » => cout ;
« écolo », « CO2 » => environnement ; « fatigué(e) », « valise », « pas envie de pédaler » => effort élevé
(effort = importance d'éviter l'effort physique). N'invente pas de lieu.`;

const EXPLAIN_PROMPT = `Tu es l'« Agent de décision et de coordination » d'un système multi-agents de mobilité à Bordeaux.
On te donne : la demande de l'usager, le contexte (météo, données temps réel, événements), les poids retenus,
et le classement des options calculé par un score multicritère avec facteur de sécurité non compensatoire.
Rédige en français une explication claire de 2 à 4 phrases, adressée à l'usager :
pourquoi l'option recommandée, ce qu'elle coûte par rapport au trajet le plus rapide, et quelles options ont été écartées et pourquoi.
N'invente AUCUN chiffre : utilise uniquement ceux fournis. Pas de liste, pas de markdown.`;

const FALLBACK_MODELS = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"];
let workingModel = null;

function apiKey() {
  const k = (globalThis.Netlify?.env?.get?.("LLM_API_KEY") ?? process.env.LLM_API_KEY ?? "").trim();
  return k;
}

async function callOnce(model, system, user, json) {
  const res = await fetch(`${BASE_URL}/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${apiKey()}` },
    body: JSON.stringify({
      model,
      temperature: 0.2,
      max_tokens: json ? 500 : 300,
      ...(json ? { response_format: { type: "json_object" } } : {}),
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
    }),
  });
  if (!res.ok) {
    const err = new Error(`LLM HTTP ${res.status}: ${(await res.text()).slice(0, 200)}`);
    err.status = res.status;
    throw err;
  }
  const data = await res.json();
  return data.choices?.[0]?.message?.content ?? "";
}

// Liste des modèles réellement accessibles avec cette clé (GET /models)
async function availableModels() {
  try {
    const r = await fetch(`${BASE_URL}/models`, { headers: { Authorization: `Bearer ${apiKey()}` } });
    if (!r.ok) return { error: `HTTP ${r.status}: ${(await r.text()).slice(0, 150)}`, ids: [] };
    const j = await r.json();
    const ids = (j.data || []).map((m) => m.id)
      .filter((id) => !/whisper|tts|orpheus|guard|safeguard|playai|embed|vision/i.test(id));
    const rank = (id) => (/gpt-oss-120b/.test(id) ? 0 : /70b/.test(id) ? 1 : /qwen|gpt-oss/.test(id) ? 2 : /llama/.test(id) ? 3 : 4);
    return { ids: ids.sort((a, b) => rank(a) - rank(b)) };
  } catch (e) { return { error: String(e.message || e), ids: [] }; }
}

// Essaie le modèle configuré, des modèles de secours, puis ceux que la clé peut réellement utiliser
async function callLLM(system, user, json) {
  let models = [...new Set([workingModel, MODEL, ...FALLBACK_MODELS].filter(Boolean))];
  if (!workingModel) {
    const av = await availableModels();
    if (av.ids.length) models = [...new Set([MODEL, ...av.ids.filter((id) => FALLBACK_MODELS.includes(id)), ...av.ids])].filter((m) => av.ids.includes(m)).slice(0, 5);
  }
  let last;
  for (const m of models) {
    try {
      const out = await callOnce(m, system, user, json);
      workingModel = m;
      return { out, model: m };
    } catch (e) {
      last = e;
      if (e.status === 401 || e.status === 403 || e.status === 429) break; // clé invalide ou quota : inutile d'essayer d'autres modèles
    }
  }
  throw last;
}

export default async (req) => {
  // Diagnostic : GET /api/agent indique si la clé est configurée (sans jamais l'afficher)
  if (req.method === "GET") {
    const k = apiKey();
    const diag = { ok: true, keyConfigured: !!k, keyLooksValid: /^gsk_/.test(k) || (k.length > 20 && !BASE_URL.includes("groq")), baseUrl: BASE_URL, model: MODEL };
    if (k) { const av = await availableModels(); diag.availableModels = av.ids; if (av.error) diag.modelsError = av.error; }
    if (k && new URL(req.url).searchParams.has("test")) {
      try { const r = await callLLM("Réponds uniquement : OK", "test", false); diag.llmTest = "réussi"; diag.model = r.model; }
      catch (e) { diag.llmTest = String(e.message || e).slice(0, 250); }
    }
    return Response.json(diag);
  }
  if (req.method !== "POST") return Response.json({ ok: false, error: "POST uniquement" }, { status: 405 });
  if (!apiKey()) return Response.json({ ok: false, error: "LLM_API_KEY non configurée" }, { status: 503 });

  let body;
  try { body = await req.json(); } catch { return Response.json({ ok: false, error: "JSON invalide" }, { status: 400 }); }

  try {
    if (body.task === "parse") {
      const text = String(body.text || "").slice(0, 600);
      const { out, model } = await callLLM(PARSE_PROMPT, text, true);
      const clean = out.replace(/^```(?:json)?\s*|\s*```$/g, "");
      return Response.json({ ok: true, model, data: JSON.parse(clean) });
    }
    if (body.task === "explain") {
      const payload = JSON.stringify(body.payload || {}).slice(0, 4000);
      const { out, model } = await callLLM(EXPLAIN_PROMPT, payload, false);
      return Response.json({ ok: true, model, text: out.trim() });
    }
    return Response.json({ ok: false, error: "tâche inconnue" }, { status: 400 });
  } catch (e) {
    return Response.json({ ok: false, error: String(e.message || e) }, { status: 502 });
  }
};

export const config = { path: "/api/agent" };
