# Organism CRISTAL -- Spec technique

**Version 0.8.2** | Derniere mise a jour : 2026-02-27

---

## C'est quoi en une phrase ?

L'Organism est un **cerveau artificiel a 3 agents** (A, B, C) qui pensent en boucle, sans qu'on leur pose de question. Ils explorent, debattent, construisent -- et le user peut intervenir a tout moment.

---

## Le probleme que ca resout

CRISTAL classique fonctionne en **question/reponse** :
```
User: "Salut"  -->  Pipeline  -->  "Bonjour !"
```

C'est reactif. L'IA ne fait rien quand personne ne parle.

L'Organism fait l'inverse : il tourne **en continu**, comme un cerveau au repos qui reflechit. Le user peut ecouter les agents penser, ou intervenir.

---

## Les 3 agents

| Agent | Nom | Role | Modele cloud | Temperature |
|-------|-----|------|--------------|-------------|
| **A** | Explorer | Genere des idees, explore (creatif/audacieux) | glm-4.6:cloud (357B MoE) | 0.9 (creatif) |
| **B** | Critic | Analyse, critique, cherche les failles (analyste/incisif) | deepseek-v3.1:671b-cloud (671B) | 0.3 (precis) |
| **C** | Builder | Implemente, construit, synthetise (pragmatique) | qwen3-coder:480b-cloud (480B) | 0.5 (equilibre) |

Chaque agent est un **vrai LLM cloud** appele via Ollama. Ils ne partagent pas de contexte directement -- ils communiquent via une memoire commune.

---

## Comment ca tourne : le Tick

Toutes les ~15 secondes (ou immediatement apres un message user), un **tick** se produit :

```
TICK #42
  1. Le Scheduler regarde les signaux et choisit un MODE
     (Idle, Explore, Debate, Implement, Consolidate, Recover)

  2. Il assigne des ROLES aux agents :
     - Lead   = celui qui parle en premier, qui guide
     - Support = celui qui aide, complete
     - Oppose  = celui qui critique, peut mettre un VETO

  3. Chaque agent recoit :
     - Une instruction de mode en francais naturel ("Explore une nouvelle piste.")
     - Son status si pertinent ("Tu menes la reflexion." pour Lead)
     - Un pack d'evidences (les faits recents de la memoire, dedupliques)
     - Les faits du World Model

  4. Chaque agent repond (appel LLM cloud, think=True)
     Les agents NE SAVENT PAS qu'ils sont dans un systeme multi-agent.
     Pas de [ROLE], pas de [MODE], pas de meta-raisonnement.

  5. Les reponses sont enregistrees dans la memoire
     et envoyees au chat via WebSocket

  TICK #43 commence apres 15s d'attente (ou immediatement si message user)
```

---

## Les couches (P0 a P5)

L'Organism est construit en 6 couches empilees :

### P0 : Reality Memory (Mr)
**Fichier** : `mr.py`

Un journal d'evenements qui ne peut jamais etre modifie. Chaque evenement est lie au precedent par un hash SHA-256 (comme une mini-blockchain). Ca garantit qu'aucun souvenir ne peut etre falsifie.

```
Event #1: Agent A dit "Et si on explorait les fractales ?"  [hash: a3f2...]
Event #2: Agent B dit "Pas tres utile sans objectif"         [hash: 7b1c... <- depend de a3f2]
Event #3: User dit "Parlez-moi de musique"                   [hash: e9d4... <- depend de 7b1c]
```

### P1 : L0R Ring (memoire de travail)
**Fichier** : `l0r.py`

Un buffer circulaire de 64 "slots". Chaque slot pointe vers un evenement dans Mr. Les slots ont une **salience** (importance) et un **TTL** (duree de vie).

C'est la "memoire de travail" : ce que l'organisme a en tete en ce moment. Les vieux souvenirs sans importance tombent, les souvenirs importants restent.

Quand un agent doit parler, on lui construit un **Evidence Pack** : les N souvenirs les plus importants qui tiennent dans son budget de tokens.

### P2 : Scheduler
**Fichier** : `scheduler.py`

Le chef d'orchestre. Il decide :
- **Quel mode** activer (Explore quand c'est calme, Debate quand il y a un conflit, etc.)
- **Quel role** donner a chaque agent (Lead, Support, Oppose)
- **Combien de tokens** chaque agent peut utiliser

Les transitions entre modes sont basees sur 7 signaux de controle :
- **energy** : reserve energetique (diminue avec les tokens consommes)
- **novelty** : est-ce qu'on decouvre des choses nouvelles ?
- **conflict** : est-ce que les agents ne sont pas d'accord ?
- **cohesion** : est-ce que tout est coherent ?
- **impl_pressure** : est-ce qu'il faut agir maintenant ?
- **cost_pressure** : pression cout (tokens/min, latence)
- **prediction_error** : ecart entre projections et observations

Les signaux sont estimes par **heuristiques textuelles francaises** (mots-cles comme "hypothese", "cependant", "d'accord", "concretement") — pas d'auto-evaluation par les LLMs.

### P3 : World Model (Mw)
**Fichier** : `world_model.py`

Un graphe de "claims" (affirmations). Chaque claim a :
- Un texte ("La musique baroque utilise le contrepoint")
- Une provenance (quel agent l'a dit, a quel tick)
- Un statut (HYPOTHESIS, SUPPORTED, CONTRADICTED, RETRACTED)
- Des liens vers d'autres claims

Ca permet a l'organisme de savoir ce qu'il "croit" et de changer d'avis.

### P4 : Orchestrator
**Fichier** : `orchestrator.py`

La boucle principale. C'est lui qui :
1. Demande au Scheduler les roles
2. Construit le prompt pour chaque agent
3. Appelle chaque agent
4. Enregistre les resultats dans Mr et L0R
5. Met a jour le World Model
6. Retourne un `TickResult`

### P5 : Agent Wrapper
**Fichier** : `agent_wrapper.py`

Le pont entre l'Orchestrator et les vrais LLMs. Pour chaque agent :
1. Prend le prompt construit par l'Orchestrator
2. Appelle ollama.chat() avec le bon modele cloud (think=True)
3. Extrait le contenu (prefere content > 50 chars, sinon fusionne thinking+content)
4. Estime les signaux cognitifs par heuristiques textuelles francaises
5. Detecte les boucles de repetition et tronque si necessaire (max 3000 chars)
6. Retourne un AgentTurn propre

System prompts par agent : chaque agent a un prompt unique qui definit sa personnalite
(creatif / rigoureux / pragmatique) sans jamais mentionner CRISTAL, Organisme, ou Agent A/B/C.
Instruction explicite : "Reponds UNIQUEMENT en francais. Jamais d'anglais."

---

## L'interface utilisateur

### Le toggle

Dans le panneau de configuration (a droite), tout en haut :

```
[x] Organism
    Boucle interne autonome -- les agents pensent librement.
```

- **Coche** = l'organisme demarre. Les agents commencent a penser.
- **Decoche** = l'organisme s'arrete. Retour au mode classique.

### Ce qu'on voit dans le chat

Quand l'organisme tourne, les messages des agents apparaissent avec un prefixe colore :

```
[Explorer . Idle] Support -- Explorons le concept de memoire collective...
[Critic . Explore] Oppose -- Attention, cette hypothese manque de fondement.
[Builder . Debate] Lead -- Je propose de synthetiser les deux positions...
```

Les couleurs :
- Explorer (Agent A) = **cyan** (#00E5FF)
- Critic (Agent B) = **rouge** (#FF6B6B)
- Builder (Agent C) = **vert** (#7CFC00)

### Ligne d'audit (tick)

A chaque tick, une ligne discrete apparait dans le chat :

```
── tick #42 | Explore [NEW] | 3 agents | 1200tok | 4500ms ──
```

Ca permet de suivre en temps reel : quel tick, quel mode, combien d'agents ont parle, combien de tokens, et la latence. `[NEW]` apparait quand le mode vient de changer.

### Envoyer un message

Quand l'organisme est actif et que le user tape un message :
1. Le message **n'entre PAS** dans le pipeline classique
2. Il est **injecte** dans la boucle organism via `inject_message()`
3. Les signaux sont **boostes immediatement** (novelty → 0.7, energy → 0.8) pour forcer une transition de mode
4. Au prochain tick, le message est place **en tete du prompt** de chaque agent, avec un marqueur visible `>>>`
5. Les agents reagissent en priorite au message du user

---

## Fichiers modifies/crees

| Fichier | Action | Ce qu'il fait |
|---------|--------|---------------|
| `organism/mr.py` | Existait | Memoire d'evenements (journal + hash chain) |
| `organism/l0r.py` | Existait | Buffer circulaire (memoire de travail) |
| `organism/scheduler.py` | Existait | Choix des modes et roles |
| `organism/world_model.py` | Existait | Graphe de croyances |
| `organism/orchestrator.py` | Existait | Boucle de ticks |
| `organism/agent_wrapper.py` | **Cree** | Connexion aux LLMs Ollama cloud |
| `organism/organism_loop.py` | **Cree** | Thread autonome + emission WebSocket |
| `organism/config.py` | **Modifie** | Ajout config agents (modeles, temperatures) |
| `server_state.py` | **Modifie** | Ajout `organism_active: bool` |
| `websocket_handlers.py` | **Modifie** | Handler toggle_organism |
| `message_routes.py` | **Modifie** | Routing vers organism quand actif |
| `templates/chat.html` | **Modifie** | Toggle dans le panneau config |
| `static/js/cristal.js` | **Modifie** | Logique toggle + affichage des turns |
| `static/css/cristal.css` | **Modifie** | Styles des messages organism |
| `config/cristal.json` | **Modifie** | Section agents avec modeles cloud |

---

## Bugs corriges (v0.6.0)

Lors des premiers tests avec les modeles cloud, 4 problemes critiques ont ete identifies et corriges :

### 1. Bloque en mode Idle

**Le probleme** : Quand l'organisme demarre, tous les signaux sont a zero (novelty=0, conflict=0, etc.). Le score du mode Idle est calcule comme `0.5 * (1 - energy) + 0.2 * (1 - novelty) + ...`, donc il vaut ~0.9 au demarrage. Avec le bonus d'hysteresis (+0.15) pour le mode courant, Idle gagnait **toujours** le softmax. Meme quand le user envoyait un message, les signaux ne changeaient pas assez pour sortir d'Idle.

**La solution** : Quand un message user est injecte (`inject_user_message`), les signaux sont **boostes immediatement** :
- novelty → 0.7 (y'a du nouveau !)
- energy → 0.8 (on a de l'energie)
- prediction_error → 0.3 (surprise)

Ca force le Scheduler a quitter Idle au tick suivant.

### 2. Chambre d'echo (agents qui repetent en boucle)

**Le probleme** : Quand Agent A dit "La surface de la Terre est de 510 millions de km2", cette reponse est enregistree dans la memoire (L0R). Au tick suivant, cette reponse revient dans l'evidence pack de TOUS les agents (y compris A lui-meme). Resultat : les 3 agents voient la meme phrase 15 fois et la repetent chacun a leur tour. C'est une boucle : plus ils la repetent, plus elle a de salience, plus elle revient.

**La solution** : **Deduplication de l'evidence pack**. Avant d'envoyer le pack a un agent, on filtre :
- Si un message d'agent a les memes 80 premiers caracteres qu'un precedent, on le saute
- Les messages du **user** ne sont JAMAIS dedupliques (ils sont sacres)
- Chaque evidence est prefixee par son score de salience pour que l'agent sache ce qui est important

### 3. Messages user ignores

**Le probleme** : Dans le prompt envoye aux agents, le message du user etait place a la **fin** (section 7 sur 8). Les LLMs donnent plus d'importance au debut du prompt. Resultat : les agents parlaient de leur sujet sans meme voir la question du user.

**La solution** : Le message user est maintenant place en **section 1** (tout en haut du prompt, avant le mode), avec un marqueur fort :
```
Reponds a cette question :
>>> Parlez-moi de musique
```
De plus, `inject_message()` reveille la boucle immediatement (`_wake.set()`) au lieu d'attendre le prochain tick.

### 4. Pas auditable

**Le probleme** : Impossible de savoir ce qui se passait. Quel mode ? Combien de tokens ? Quelle latence ? Juste des murs de texte dans le chat.

**La solution** :
- **Dans le chat** : une ligne discrete a chaque tick `── tick #42 | Explore [NEW] | 3 agents | 1200tok | 4500ms ──`
- **Dans le terminal** : log detaille de chaque agent `[Agent A / glm-4.6:cloud] Support — calling (temp=0.9, max_tok=400)` + `→ 180 tok out, 2300ms, 420 chars`
- **Prefixe enrichi** : chaque message montre le status `[Explorer·Explore·Lead]`

---

## Protections anti-garbage

Les petits LLMs locaux produisaient du texte repetitif infini. Meme avec les modeles cloud, on garde les protections :

1. **num_predict** : limite le nombre de tokens en sortie (1500-2000 selon l'agent)
2. **repeat_penalty** : 1.2 pour penaliser les repetitions dans la generation
3. **Detection de boucle** : si un pattern de 20+ caracteres se repete 3+ fois, on coupe
4. **Troncature** : max 3000 chars cote wrapper, max 2000 chars cote WebSocket
5. **Logging** : chaque appel LLM affiche dans le terminal quel agent, quel modele, combien de tokens, quelle latence
6. **Content > thinking** : si le content LLM est > 50 chars, le thinking (souvent meta-analyse en anglais) est ignore

---

## Evaluation et benchmark (v0.7.0)

### Pipeline d'evaluation

L'Organism dispose d'un pipeline d'evaluation automatique qui mesure les metriques a chaque tick et produit des rapports comparatifs.

| Fichier | Role |
|---------|------|
| `organism/evaluator.py` | Observateur — calcule les metriques par tick, ecrit JSONL + summary.json |
| `scripts/run_bench.py` | Lance 3 conditions (Organism / RoundRobin / Monologue) avec meme budget |
| `scripts/report_metrics.py` | Lit les JSONL, produit CSV + stats p50/p95 + courbes |
| `docs/EVALUATION.md` | Definitions metriques, protocole, seuils |

### Les 3 conditions du benchmark

| Condition | Scheduler | Roles | Agents | WorldModel |
|-----------|-----------|-------|--------|------------|
| **Organism** | Actif (6 modes, softmax, hysteresis) | Dynamiques (Lead/Support/Oppose) | 3 (A, B, C) | Actif |
| **RoundRobin** | Desactive (mode fixe Explore) | Cycliques (A→Lead, B→Lead, C→Lead) | 3 (A, B, C) | Desactive |
| **Monologue** | Desactive | Pas de roles | 1 (A seul, budget 3x) | Desactive |

### Metriques clefs

- **repetition_3gram** : proportion de trigrammes repetes (rumination)
- **hashvec_novelty** : changement de sujet entre ticks (hashing trick + cosine)
- **agent_balance_entropy** : equilibre de parole entre agents (Shannon)
- **mode_entropy_w20** : diversite des modes sur 20 ticks
- **user_response_latency** : ticks entre injection user et reaction
- **signal_mode_corr** : correlation Spearman signaux ↔ transitions

Voir `docs/EVALUATION.md` pour les formules et seuils.

### Lancement

```bash
# Mode dry (test structure, pas de LLM)
python scripts/run_bench.py --ticks 10 --dry

# Mode reel (300 ticks, vrais LLMs cloud)
python scripts/run_bench.py --ticks 300

# Rapport
python scripts/report_metrics.py runs/<dossier_bench>/
```

Le bench detecte automatiquement WSL et configure OLLAMA_HOST vers le host Windows. Il abort apres 5 ticks consecutifs sans reponse LLM (fail-fast).

---

## Reflexion sur la publicabilite

### Ce qui est nouveau (contribution potentielle)

1. **Architecture a boucle autonome** : les agents pensent en continu sans stimulus externe. C'est fondamentalement different du multi-agent reactif (AutoGen, CrewAI, CAMEL) ou l'execution est declenchee par une tache.

2. **Scheduler a modes cognitifs** : transitions entre 6 modes (Idle/Explore/Debate/Implement/Consolidate/Recover) guidees par des signaux auto-evalues, avec hysteresis et dwell time. Ce n'est pas un FSM classique — c'est un softmax continu sur des scores composites.

3. **Roles dynamiques contextuels** : Lead/Support/Oppose ne sont pas fixes mais assignes en fonction du mode. L'agent qui explore en mode Explore devient celui qui critique en mode Debate.

4. **Memoire a integrite cryptographique** : la Reality Memory (Mr) est un journal hash-chaine. Aucun souvenir ne peut etre falsifie retroactivement.

5. **Injection humaine non-bloquante** : le user intervient dans un flux autonome, pas dans un dialogue. L'injection booste les signaux pour forcer une reorientation, ce qui est une forme de "steering" continu.

### Ce qui pose probleme pour publier

1. **Heuristiques textuelles limitees** : les signaux cognitifs (novelty, conflict, cohesion) sont estimes par comptage de mots-cles francais. C'est mieux que l'auto-evaluation LLM (qui copiait les exemples), mais ca reste superficiel. Le juge 8B (Phase 1) les remplacera.

2. **Baselines trop faibles** : RoundRobin et Monologue sont des versions degradees du meme systeme. Ce n'est pas une comparaison avec l'etat de l'art (AutoGen, CrewAI, MetaGPT). Les reviewers diront "vous avez construit les baselines pour perdre".

3. **Metriques purement lexicales** : repetition_3gram et hashvec_novelty ne captent que la surface. Un agent qui dit la meme chose avec des mots differents a une bonne novelty mais zero progres reel.

4. **N=1, pas de puissance statistique** : un seul run de 300 ticks. Les LLMs cloud ne sont pas deterministes (pas de seed). Impossible de faire un test t ou un Mann-Whitney.

5. **Confounding : modeles heterogenes** : les 3 agents utilisent 3 modeles differents (GLM-4.6, DeepSeek, Qwen). Toute difference entre conditions pourrait venir des modeles, pas de l'architecture.

6. **WorldModel inoperant** : dans les premiers benchs, wm_stats reste a zero (aucune claim parsee). Le graphe de croyances est la en theorie mais pas en pratique.

### Angles d'attaque possibles

**Option A : Systems/Demo paper (AAAI, NeurIPS workshops, AAMAS)**
- Focus sur l'architecture, pas sur les resultats numeriques
- "Voici un systeme qui fait X, voici comment il marche, voici ce qu'on observe"
- Les metriques sont illustratives, pas probantes
- Avantage : bar plus basse, contribution = le systeme lui-meme
- Cible : Workshop on LLM Agents, Multi-Agent Systems, Human-AI Interaction

**Option B : Ablation study (plus solide)**
- Au lieu de Organism vs RoundRobin, faire :
  - Organism complet
  - Organism sans scheduler (mode fixe, mais roles dynamiques)
  - Organism sans roles (scheduler actif, mais roles fixes)
  - Organism sans WorldModel (scheduler + roles, mais pas de claims)
- Ca isole l'effet de chaque composant
- Plus convaincant qu'une baseline externe

**Option C : Etude qualitative + mixed methods**
- 300 ticks quantitatifs + annotation humaine de 20 fenetres
- Un juge humain note : coherence, pertinence, creativite, progression
- Correlation entre annotations humaines et metriques automatiques
- Ca valide (ou invalide) les metriques lexicales

**Option D : Focus sur le steering (HCI angle)**
- Le user injecte un message → combien de ticks pour reorienter les agents ?
- Comparer : injection avec signal boost vs injection sans boost vs pas d'injection
- Ca teste une hypothese precise et falsifiable
- Cible : CHI, IUI, UIST workshops

### Recommendation

L'angle le plus honnete et defensible est probablement **A + B** : un systems paper avec ablation. On presente l'architecture comme contribution principale, et l'ablation montre que chaque composant (scheduler, roles, worldmodel) a un effet mesurable — sans pretendre battre l'etat de l'art.

---

## Tests

244 tests unitaires couvrent l'ensemble organism (+ evaluator + bench + theories) :

```bash
# Tous les tests (exclure companion qui n'est plus actif)
python -m pytest tests/ --ignore=tests/test_behavior_modulator.py \
  --ignore=tests/test_companion_spec.py --ignore=tests/test_intimacy_tracker.py -v

# Tests theories uniquement (25 tests)
python -m pytest tests/test_theories.py -v

# Bench dry (verifie la structure de sortie)
python scripts/run_bench.py --ticks 5 --dry
```

---

## Pour tester

```bash
cd /home/thom315/cristal_work/CRISTAL_v12_UI_V3
source .venv/bin/activate
python3 server.py
```

1. Ouvrir http://localhost:5000
2. Cliquer sur l'icone config (panneau droit)
3. Cocher "Organism" tout en haut
4. Observer les agents penser dans le chat
5. Envoyer un message -> il est injecte dans la boucle
6. Decocher -> la boucle s'arrete, retour au mode classique

---

## Phase 0 — Changelog (22 fev 2026)

Changements appliques pour "faire penser les agents" au lieu de decrire leur role.

### agent_wrapper.py

- **System prompts par agent** : A=creatif, B=rigoureux, C=pragmatique. Aucune mention de CRISTAL/Organisme/Agent A/B/C.
- **Instruction francais explicite** : "Reponds UNIQUEMENT en francais. Jamais d'anglais." + interdictions listes numerotees, meta-analyse.
- **`<signals>` supprime** : plus d'auto-evaluation. Remplace par `_estimate_signals_from_text()` (heuristiques mots-cles francais).
- **`_extract_content` smart** : si content > 50 chars, ignore le thinking (evite la meta-analyse anglaise de GLM).
- **Dead code deplace** dans `organism/bak/dead_code_phase0.py`.

### orchestrator.py

- **Prompt agent simplifie** : plus de tags `[ROLE]`, `[MODE]`, `[STATUS]`, `[BOOTSTRAP]`, `[FACTS]`, `[FORMAT]`. Remplace par du francais naturel (1 phrase mode + 1 ligne status + contexte + facts).
- **Message user en section 1** (tout en haut du prompt).
- **Evidence pack elargi** : `turn.text[:500]` → `turn.text[:1500]`, dedupe `text[:80]` → `text[:400]`.

### scheduler.py

- **Signaux initiaux non-nuls** : `novelty=0.5, energy=1.0, prediction_error=0.3` (etait tout a 0 → bloque en Idle).
- **Idle dwell_min** : 3 → 1 tick (transition plus rapide).

### config.py

- **num_predict augmente** : A: 1500→2000, B: 500→1500, C: 600→1500.

### organism_loop.py

- **tick_interval** : 2.0s → 15.0s (evite de saturer les LLMs cloud).
- **Wake-up user** : `_wake` event — quand un message est injecte, le prochain tick demarre immediatement.
- **Troncature WebSocket** : 800 → 2000 chars.

---

## Phase 1 — Changelog (26-27 fev 2026)

### Juge cloud (v0.7.1)

- **Judge upgrade** : deepseek-r1:8b (local) → qwen3-vl:235b-cloud (cloud). L'ancien juge choisissait C 95% du temps avec 4 marges distinctes.
- **Summarizer** : gpt-oss:120b-cloud (inchange).
- **`_ollama_chat_smart()`** dans `judge.py` : auto-detection des capacites modele (think+json / json-only / free-text). Cache le resultat apres le premier appel. qwen3-vl utilise "json-only" (think=True vide le champ content).
- **`_MODEL_CAPS`** : cache global `Dict[str, tuple]` → `(use_think, use_format_json)`.
- **Gate P1 PASSED** : 97% JSON valid, Var(margin)=0.0425, winners C=31/B=27/A=22.

### Agent rebalance (v0.7.1)

- **Prompts agents** : A=creatif/audacieux, B=analyste/incisif, C=pragmatique.
- **Judge prompt** : echelle 0-100 entier, detection de dominance, prompt pedagogique.
- **Temperature adaptative du juge** : basee sur variance des marges recentes.

### Fix theories de conscience (v0.8.0)

6 bugs corriges sur 8 theories. Audit complet + 25 tests unitaires.

#### orchestrator.py

- **prediction_error dynamique** : calcule `pe = 0.4 * pe_mode + 0.6 * pe_signals` a chaque tick au lieu de copier la valeur initiale 0.3. Utilise une fenetre glissante de 5 ticks.
- **claims_added_this_tick** : ajoute dans `wm_stats` pour que GWT utilise le churn par tick.

#### consciousness/theories/mdm.py

- **Jaccard 3-grams** : `_jaccard_similarity` sur mots → `_jaccard_3gram` sur word 3-grams. Plus sensible aux reformulations.
- **Poids** : 0.40 competition + 0.30 diversity + 0.30 selection (etait 0.35/0.30/0.25/0.10).
- **diagnostics["degraded"]** quand judge_verdict=None.

#### consciousness/theories/gwt.py

- **wm_churn** : `total_claims / 20` (cumulatif, saturait a 1.0 apres 20 claims) → `claims_added_this_tick / 20` (par tick).
- **Poids redistribues** : broadcast_wm=0.35 (principal), memory=0.20, mode=0.15, agents=0.15, winner=0.15.

#### consciousness/theories/hot.py

- **eval_density** : `len(reason) / 100` → densite de mots evaluatifs (meilleur, faille, critique, superieur, etc.). Score = hits / 3, cap a 1.0.
- **judge_divergence** : `1 - margin_2v3` (le juge hesite entre 2e et 3e = forte reflexion).
- **Fallback sans juge** : utilise veto_present et oppose_count comme proxy meta-cognitif.

#### consciousness/theories/fep.py

- **Fenetre glissante** : `pe_history` (deque maxlen=20). Calcule `error_reduction = max(0, pe(t-1) - pe(t))`.
- **learning_rate** : regression lineaire sur pe_history. Slope negative = le systeme apprend. `learning_rate = max(0, -slope * 5)`, cap a 1.0.
- **active_inference** : 1.0 si Implement/Consolidate, 0.5 si Explore, 0.4 si Debate, 0.2 sinon.
- **NaN au tick 1** (pas assez d'historique).

#### consciousness/theories/iit.py

- **Rewrite complet** : l'ancien code retournait 0.0 quand judge_verdict=None (integration et irreducibility dependaient du juge).
- **integration_agents** : cosinus entre vecteurs de signaux des paires d'agents. Toujours disponible.
- **signal_mode_alignment** : le mode correspond-il au signal dominant ? (Explore↔novelty, Debate↔conflict, etc.)
- **wm_integration** : contradictions / total_claims × 10 + avg_confidence.
- **judge_bonus** : +0.15 a +0.40 quand le juge est present (bonus, pas pre-requis).

#### consciousness/theories/dyn.py

- **NaN si < 3 ticks** : etait 0.25 au tick 1 (incalculable).
- **Pearson sur fenetre glissante** : correlation temporelle entre paires d'agents × 4 dimensions de signaux. Remplace la variance instantanee.
- **agent_history** : `Dict[str, deque(maxlen=20)]` par agent.

#### consciousness/theories/hybrid.py

- **NaN avant 20 ticks** : etait 0.0 constant. Pas assez de donnees pour les poids.
- **Qualite multi-critere** : `0.30*wm_conf + 0.25*judge_conf + 0.25*participation + 0.20*conflict_quality` (etait juste judge.confidence).
- **Exclut les NaN** : les theories qui retournent NaN (FEP tick 1, DYN tick 1-2) sont exclues du calcul pondere.
- **Weight update** : multiplie par `(1 + 0.1 * corr)` puis normalise (etait `lr * (target - current)`).

#### tests/test_theories.py (CREE)

25 tests couvrant :
- Score dans [0,1] avec verdict complet
- Score degrade (pas 0.0) sans verdict
- NaN aux premiers ticks pour DYN/FEP/Hybrid
- GWT insensible a total_claims (utilise churn)
- HOT insensible a la longueur du reason (utilise eval_density)
- IIT > 0 sans juge
- Hybrid poids evoluent apres 40 ticks
- Hybrid exclut les NaN

### Fix pipeline (v0.8.2)

4 bugs corriges dans le pipeline principal. BUG 1 (veto court-circuite le juge) faisait deja partie du fix v0.8.1.

#### judge.py — Parse robuste (BUG 2)

- **`_normalize_agent_id()`** : normalise les variantes du winner ("Agent B", "agent_b", " B ", "AGENT_B") vers l'ID canonique.
- **Ranking fallback** : si le winner est invalide, utilise le premier du ranking qui matche un agent valide.
- **Truncated JSON repair** : dans `_extract_json()`, si le JSON est tronque (braces non fermees), ferme automatiquement les braces avant de retenter le parsing.
- **Ranking normalization** : chaque entree du ranking est normalisee via `_normalize_agent_id()`.
- **Cible** : parse errors < 5% (etait ~20%).

#### scheduler.py — Veto → Debate boost (BUG 3)

- **`register_veto()`** : methode appelee par l'orchestrateur quand un veto est pose. Positionne `_veto_this_tick = True`.
- **Boost dans `_compute_raw_scores()`** : `scores[Mode.DEBATE] += 0.3` quand un veto a ete pose ce tick.
- **Reset apres scoring** : le flag est consomme dans `tick()` apres le calcul des scores.
- **Wire** : `self._scheduler.register_veto()` appele dans orchestrator quand un veto passe le budget Fibonacci.
- **Cible** : Debate > 5% sur 100 ticks (etait 0%).

#### orchestrator.py — Claims fallback (BUG 4)

- **Heuristique d'extraction** : quand le juge ne produit pas de claims (champ vide/absent), extrait les phrases assertives du draft gagnant via `_extract_claims_from_text()`.
- **Max 3 claims** par tick, confidence 0.5, status "hypothesis".
- **Provenance** : lies au chunk_id du draft gagnant.
- **Cible** : >= 1 claim dans >= 50% des ticks avec verdict valide (etait 0%).

#### Bench validation (20 ticks, 27 fev 2026, run 20260227_210455_bench)

| Metrique | Avant | Apres | Cible | Status |
|----------|-------|-------|-------|--------|
| Parse errors juge | ~20% | 0% | < 5% | PASS |
| Debate mode | 0% | 20% | > 5% | PASS |
| Claims par tick | 0% | 100% | >= 50% | PASS |
| Veto rate | 55% | 10% | raisonnable | PASS |
| Winners diversifies | C=95% | B=8/C=8/A=4 | equilibre | PASS |
| Var(margin) | 0.017 | 0.046 | > 0.02 | PASS |
| prediction_error | constant 0.3 | 19 unique | dynamique | PASS |
| 8 theories actives | 4 affichees | 7 actives + Hybrid NaN | toutes | PASS |

Resultats detailles :
- **Judge** : 20/20 verdicts valides (100%), confidence moyenne 0.83
- **Modes** : Implement 75%, Debate 20%, Idle 5%, 2 transitions
- **Claims** : 194 total, 193 supported, 1 contradicted
- **Theories** : HOT=0.74, MDM=0.71, IIT=0.63, GWT=0.62, RPT=0.60, DYN=0.38, FEP=0.29, Hybrid=NaN (need 20+ ticks)
- **Latence** : avg 125s/tick, user response 1.0 tick

### Fix STEM/Debate/Hybrid (v0.8.3)

3 bugs corriges apres le bench 100 ticks v0.8.2 (Debate=0%, Hybrid=0.000, STEM NaN).

#### scheduler.py — Debate decaying boost (BUG 5)

- **Probleme** : le boost single-tick +0.3 etait insuffisant contre Implement + hysteresis sur 100 ticks.
- **Fix** : `_veto_this_tick: bool` remplace par `_veto_boost: float`. Initialise a 0.5, decay x0.6/tick (0.5 → 0.30 → 0.18 → 0.11 → cutoff 0.05). Persiste ~3-4 ticks par veto.
- **Resultat** : Debate passe de 0% a 38% sur 100 ticks.

#### orchestrator.py — Hybrid ordering (BUG 6)

- **Probleme** : les 8 theories etaient calculees dans la meme boucle. Hybrid lisait `state.theory_scores` qui etait vide pendant la boucle → score constant 0.000.
- **Fix** : boucle separee — 7 theories normales d'abord, update `theory_scores`, puis Hybrid.
- **Resultat** : Hybrid avg 0.582 (81 ticks avec donnees, 19 NaN attendus).

#### stem.py — PCA NaN-safe + zero-variance (BUG 7)

- **Probleme** : les scores NaN des theories (DYN tick 1-2, FEP tick 1, Hybrid tick 1-19) propagaient dans la matrice de covariance. Les dimensions a variance zero (colonnes constantes) degeneraient la PCA.
- **Fix** : `math.isfinite()` guard sur les theory scores dans le vecteur d'etat. Filtrage des dimensions a variance < 1e-12 avant covariance. Guard NaN/inf dans velocities.
- **Resultat** : 100 points PCA, effective_dim=2.81, 5 attracteurs, 9 transitions de phase.

#### Bench validation (100 ticks, 28 fev 2026, run 20260228_204414_bench)

| Metrique | v0.8.2 (100t) | v0.8.3 (100t) | Cible | Status |
|----------|---------------|---------------|-------|--------|
| Debate mode | 0% | 38% | > 5% | PASS |
| Hybrid score | 0.000 | 0.582 | non-zero | PASS |
| STEM PCA | NaN possible | 100 pts, dim=2.81 | fonctionnel | PASS |
| JSON valid | 98% | 100% | >= 90% | PASS |
| Var(margin) | 0.039 | 0.056 | > 0.02 | PASS |
| Winners | A=20/B=38/C=42 | A=26/B=39/C=35 | equilibre | PASS |
| Mode entropy | 0.812 | 0.714 | > 0.5 | PASS |
| Gate P1 | PASS | PASS | - | PASS |

Resultats detailles :
- **Judge** : 100/100 valides (100%), confidence avg 0.837
- **Modes** : Implement 36%, Debate 38%, Idle 14%, Explore 12%, 10 transitions
- **Theories** : IIT=0.742, HOT=0.736, MDM=0.694, RPT=0.667, GWT=0.647, Hybrid=0.582, DYN=0.400, FEP=0.188
- **STEM** : 5 attracteurs, 9 phase transitions, variance 3D expliquee = 1.0
- **Latence** : avg 127.6s/tick, p95 185.9s
