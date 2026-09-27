# Mobi·Agents Bordeaux

Prototype exploratoire d'**IA agentique pour la recommandation personnalisée de trajets multimodaux**, réalisé en lien avec le sujet de thèse *« IA agentique pour la mobilité intelligente : recommandation personnalisée et adaptative de trajets multimodaux dans des environnements dynamiques et incertains »* (LaBRI – Chaire MTI, université de Bordeaux).

**Démo en ligne :** https://VOTRE-SITE.netlify.app

## Idée

Un usager décrit son besoin en langage naturel (*« Je vais de la Gare Saint-Jean au LaBRI, il pleut un peu, je préfère éviter les grands axes à vélo »*). Trois agents coopèrent pour y répondre :

| Agent | Rôle | Implémentation |
|---|---|---|
| **Préférences** | Transformer la demande en poids sur 6 critères et en contraintes dures ; apprendre des retours de l'usager | LLM (Llama 3.3 70B via Groq) avec sortie JSON contrainte, et analyse par règles en repli |
| **Environnement** | Observer l'état réel de la ville et qualifier la confiance de chaque source | V³ temps réel (Bordeaux Métropole), Open-Meteo, routage OSM, géocodage Géoplateforme/BAN |
| **Décision et coordination** | Arbitrer entre les options, écarter celles qui sont infaisables, expliquer le choix, re-planifier en cas d'imprévu | Score multicritère contextuel, facteur de sécurité non compensatoire, hystérésis, explication générée |

## Score de décision

```
Score = [ Σ_k w_k(contexte) · s_k ] × f(S) × (0,9 + 0,1 · confiance)
f(S)  = 1 / (1 + e^(−20 (S − 0,35)))
```

- `k` ∈ {temps, sécurité, confort, coût, environnement, effort}.
- Les poids `w_k` viennent de la demande et du profil appris, puis sont modulés par le contexte (pluie, nuit, vent, chaleur, heure de pointe).
- `f(S)` est un facteur de sécurité non compensatoire : une option dangereuse s'effondre quel que soit le reste.
- Lors d'une re-planification, la recommandation ne change que si l'écart dépasse 0,04 (hystérésis), pour éviter les bascules instables.
- Quand l'usager dit « Je préfère celle-ci », le profil est mis à jour : `w ← w + η (s_choisie − s_recommandée)`.
- Chaque recommandation est comparée à une référence classique, le « trajet le plus rapide ».

Ces mécanismes transposent à la mobilité l'**indice de confiance opérationnelle (OTI)** que j'ai développé pendant mon mémoire de master (sélection de drones-relais pour le V2X, LaBRI 2026) : poids dépendant du contexte, facteur non compensatoire, hystérésis, explicabilité.

## Limites assumées

- Les temps en transport en commun sont **estimés** : les horaires GTFS de TBM ne sont pas encore intégrés.
- Les coefficients de sécurité et de confort par mode sont posés à la main ; il faudrait les apprendre sur des données réelles.
- Le profil appris ne dure que le temps de la session (pas d'historique, pas encore d'apprentissage par renforcement).
- La disponibilité des trottinettes n'est pas connectée.

Pistes pour la suite : intégration GTFS/GTFS-RT de TBM, apprentissage des préférences sur des historiques de déplacements, agent de prédiction de la disponibilité V³ (séries temporelles), comparaison avec des approches par apprentissage par renforcement profond.

## Structure

```
index.html                  interface (carte Leaflet, agents, journal, options)
app.js                      logique des trois agents
netlify/functions/agent.mjs relais serveur vers le LLM (la clé n'est jamais exposée)
netlify.toml                configuration Netlify
```

## Déploiement

1. Créer une clé API gratuite sur https://console.groq.com (menu *API Keys*).
2. Pousser ce dossier sur GitHub, puis sur Netlify : *Add new site → Import an existing project → GitHub*.
3. Dans *Site configuration → Environment variables*, ajouter `LLM_API_KEY` = la clé Groq, puis relancer le déploiement.
4. Facultatif : `LLM_MODEL` (défaut `llama-3.3-70b-versatile`) et `LLM_BASE_URL` (tout fournisseur compatible OpenAI).

Sans clé, le site fonctionne quand même : l'Agent Préférences passe en analyse par règles, et l'interface le signale.

---
Eya Adouni — septembre 2026 · [portfolio](https://eya-adouni.netlify.app/) · données : Bordeaux Métropole, Open-Meteo, © contributeurs OpenStreetMap, IGN Géoplateforme.
