"""
Organism Loop — Boucle autonome de l'organisme CRISTAL
======================================================
Thread de ticks continus. Les agents A/B/C dialoguent entre eux,
le scheduler gère les modes, et les turns sont émis via WebSocket.

Usage (depuis server):
    from organism.organism_loop import organism_loop
    organism_loop.init(socketio)
    organism_loop.start()   # démarre la boucle
    organism_loop.stop()    # arrête proprement
    organism_loop.inject_message("Bonjour")  # message user
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from organism.mr import RealityMemory
from organism.l0r import L0RRing
from organism.scheduler import Scheduler
from organism.world_model import WorldModel
from organism.orchestrator import Orchestrator, TickResult, AgentTurn
from organism.agent_wrapper import OllamaAgentFn
from organism.judge import JudgePipeline
from organism.stem import StateEvolutionModel
from organism.config import ORGANISM_CONFIG
from consciousness.theories import ALL_THEORIES

log = logging.getLogger("organism.loop")


class OrganismLoop:
    """
    Boucle autonome de l'organisme.
    Tourne dans un thread daemon, émet les turns via socketio.
    """

    def __init__(self, tick_interval: float = 15.0):
        self._tick_interval = tick_interval
        self._socketio = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._wake = threading.Event()   # Réveille la boucle quand un user parle
        self._orch: Optional[Orchestrator] = None
        self._mr: Optional[RealityMemory] = None
        self._stem: Optional[StateEvolutionModel] = None
        self._evaluator = None

    def init(self, socketio) -> None:
        """Initialise avec la référence socketio du serveur."""
        self._socketio = socketio

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Démarre la boucle de ticks."""
        if self.active:
            log.warning("Organism loop already running")
            return
        if not self._socketio:
            log.error("Organism loop not initialized (call init(socketio) first)")
            return

        # Créer les composants organism
        self._mr = RealityMemory()
        l0r = L0RRing(mr=self._mr)
        scheduler = Scheduler()
        wm = WorldModel(mr=self._mr)
        agent_fn = OllamaAgentFn()
        judge = JudgePipeline()
        self._stem = StateEvolutionModel()

        # Instantiate consciousness theories
        theories = []
        for TheoryCls in ALL_THEORIES:
            try:
                theories.append(TheoryCls())
            except Exception:
                log.warning("Failed to instantiate theory %s", TheoryCls.__name__)

        self._orch = Orchestrator(
            mr=self._mr,
            l0r=l0r,
            scheduler=scheduler,
            world_model=wm,
            agent_fn=agent_fn,
            judge_pipeline=judge,
            theories=theories,
            stem=self._stem,
        )

        # Evaluator (metriques par tick)
        try:
            from organism.evaluator import Evaluator
            run_id = f"org_{int(time.time())}"
            self._evaluator = Evaluator(run_id=run_id, output_dir="runs")
            log.info("Evaluator started (run_id=%s)", run_id)
        except Exception:
            log.warning("Evaluator not available, running without metrics")
            self._evaluator = None

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._tick_loop,
            name="organism-loop",
            daemon=True,
        )
        self._thread.start()
        log.info("Organism loop started (interval=%.1fs)", self._tick_interval)

    def stop(self) -> None:
        """Arrête la boucle proprement."""
        if not self.active:
            return
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._evaluator:
            try:
                self._evaluator.finalize()
                log.info("Evaluator finalized (summary.json written)")
            except Exception:
                log.exception("Error finalizing evaluator")
            self._evaluator = None
        self._thread = None
        self._orch = None
        log.info("Organism loop stopped")

    def inject_message(self, text: str) -> None:
        """Injecte un message utilisateur dans l'organisme.
        Réveille la boucle pour un tick immédiat."""
        if self._orch:
            self._orch.inject_user_message(text)
            if self._evaluator:
                self._evaluator.on_user_injection(text, self._orch.tick_id)
            self._wake.set()  # Réveille la boucle immédiatement
            log.info("User message injected: %.60s...", text)

    def update_config(self, patch: dict) -> None:
        """Patch la config Organism à chaud depuis l'UI.

        Modifie ORGANISM_CONFIG dict (partagé avec orchestrator/judge/agent_wrapper).
        Les changements prennent effet au prochain tick.
        """
        if not patch:
            return

        # Agent models/temperatures
        if "agents" in patch:
            agents_cfg = ORGANISM_CONFIG.get("agents", {})
            for aid, params in patch["agents"].items():
                if aid in agents_cfg:
                    agents_cfg[aid].update(params)
                else:
                    agents_cfg[aid] = params
            ORGANISM_CONFIG["agents"] = agents_cfg

        # Global num_predict / repeat_penalty for all agents
        for key in ("num_predict", "repeat_penalty"):
            if key in patch:
                for aid in ORGANISM_CONFIG.get("agents", {}):
                    ORGANISM_CONFIG["agents"][aid][key] = patch[key]

        # Judge config
        if "judge" in patch:
            judge_cfg = ORGANISM_CONFIG.get("judge", {})
            judge_cfg.update(patch["judge"])
            ORGANISM_CONFIG["judge"] = judge_cfg

        # Summarizer model
        if "summarizer_model" in patch:
            ORGANISM_CONFIG.setdefault("judge", {})["summarizer_model"] = patch["summarizer_model"]

        # Tick interval
        if "tick_interval" in patch:
            self._tick_interval = float(patch["tick_interval"])

        log.info("Config patched: %s", list(patch.keys()))

    # ── Boucle interne ───────────────────────────────────────────

    def _tick_loop(self) -> None:
        """Boucle principale — exécute un tick puis attend.
        Se réveille immédiatement si un message user arrive (_wake)."""
        log.info("Organism tick loop entering")
        while not self._stop.is_set():
            try:
                result = self._orch.run_tick()
                if self._evaluator:
                    self._evaluator.on_tick_end(
                        result, self._orch._scheduler, self._orch._wm,
                    )
                self._log_tick(result)
                self._emit_tick(result)
            except Exception:
                log.exception("Error in organism tick %d", self._orch.tick_id if self._orch else -1)

            # Attente inter-tick : interrompue par _wake (message user) ou _stop
            self._wake.clear()
            self._wake.wait(self._tick_interval)
            if self._stop.is_set():
                break

        log.info("Organism tick loop exiting")

    def _log_tick(self, result: TickResult) -> None:
        """Log détaillé d'un tick pour audit terminal."""
        sigs = result.signals
        sig_str = ""
        if sigs:
            sig_str = (
                f"nov={sigs.novelty:.2f} conf={sigs.conflict:.2f} "
                f"coh={sigs.cohesion:.2f} impl={sigs.impl_pressure:.2f} "
                f"energy={sigs.energy:.2f}"
            )
        agents_str = " | ".join(
            f"{t.agent.value}:{t.status.value}({len(t.text)}c)"
            for t in result.agent_turns
        )
        mode_flag = " [MODE CHANGED]" if result.mode_changed else ""
        judge_str = ""
        if result.judge_verdict:
            v = result.judge_verdict
            m12 = v.competition.margin_1v2 if v.competition else "?"
            win_str = v.winner or "FAILED"
            judge_str = f" | JUDGE: win={win_str} conf={v.confidence:.2f} m12={m12}"
        log.info(
            "TICK #%d | %s%s | %s | %s%s | %dtok %.0fms",
            result.tick_id, result.mode.value, mode_flag,
            agents_str, sig_str, judge_str,
            result.total_tokens, result.elapsed_ms,
        )

    def _emit_tick(self, result: TickResult) -> None:
        """Émet les turns du tick via WebSocket."""
        if not self._socketio:
            return

        for turn in result.agent_turns:
            text = (turn.text or "").strip()
            if not text:
                continue
            # Sécurité : tronquer côté émission aussi
            if len(text) > 2000:
                text = text[:2000].rsplit(" ", 1)[0] + "..."

            self._socketio.emit("organism_turn", {
                "tick_id": result.tick_id,
                "mode": result.mode.value,
                "mode_changed": result.mode_changed,
                "agent": turn.agent.value,
                "status": turn.status.value,
                "text": text,
                "signals": {
                    "novelty": round(turn.novelty, 2),
                    "conflict": round(turn.conflict, 2),
                    "cohesion": round(turn.cohesion, 2),
                    "impl_pressure": round(turn.impl_pressure, 2),
                },
                "veto": turn.veto,
                "veto_reason": turn.veto_reason,
                "tokens": turn.token_in + turn.token_out,
                "latency_ms": round(turn.latency_ms, 0),
            })

        # Émettre le judge verdict si disponible
        if result.judge_verdict:
            v = result.judge_verdict
            verdict_data = {
                "tick_id": result.tick_id,
                "winner": v.winner,
                "reason": v.reason,
                "confidence": round(v.confidence, 3),
                "signals": v.signals,
            }
            if v.competition:
                verdict_data["competition"] = {
                    "ranking": list(v.competition.ranking),
                    "margin_1v2": round(v.competition.margin_1v2, 3),
                    "margin_2v3": round(v.competition.margin_2v3, 3),
                    "counterfactual": v.competition.counterfactual,
                }
            if v.claims:
                verdict_data["claims"] = [
                    dict(c) if isinstance(c, dict) else c
                    for c in v.claims
                ]
            self._socketio.emit("judge_verdict", verdict_data)

        # Émettre theory scores si disponibles
        if result.organism_state and result.organism_state.theory_scores:
            self._socketio.emit("theory_scores", {
                "tick_id": result.tick_id,
                "scores": {
                    k: round(v, 4)
                    for k, v in result.organism_state.theory_scores.items()
                },
            })

        # Émettre STEM snapshot
        if self._stem:
            try:
                snap = self._stem.snapshot()
                if snap.get("ready"):
                    self._socketio.emit("stem_update", snap)
            except Exception:
                log.exception("STEM snapshot failed")

        # Émet aussi un résumé du tick
        self._socketio.emit("organism_tick_end", {
            "tick_id": result.tick_id,
            "mode": result.mode.value,
            "mode_changed": result.mode_changed,
            "agents_called": len(result.agent_turns),
            "vetoed": result.vetoed,
            "total_tokens": result.total_tokens,
            "elapsed_ms": round(result.elapsed_ms, 0),
            "claims_added": result.claims_added,
            "judge_winner": result.judge_verdict.winner if result.judge_verdict else None,
        })


# ── Singleton ────────────────────────────────────────────────────

organism_loop = OrganismLoop()
