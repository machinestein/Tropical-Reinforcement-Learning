"""Goal-conditioned Lean collection with portable prefixes and audited root replay."""

import copy
import re
import time

from .env import LeanEnv
from .state import LeanSnapshot


class LeanTropicEnv(LeanEnv):
    allowed_axioms = frozenset({"propext", "Classical.choice", "Quot.sound"})

    @staticmethod
    def theorem_name(record):
        statement = record["formal_statement"].strip()
        match = re.match(r"(?:theorem|lemma)\s+([A-Za-z_][A-Za-z_0-9'.]*)\s", statement)
        if match is None or not statement.endswith(":= by"):
            raise ValueError("Lean TROPIC requires a named theorem/lemma ending in ':= by'")
        return match[1]

    def reset(self, seed=None, **kwargs):
        self.goals, self.proof_success = [], False
        super().reset(seed=seed, **kwargs)
        self.theorem_name(self.current_theorem)
        result = self._run_lean_query(["skip"])
        if not result.accepted or result.success:
            raise RuntimeError(f"Cannot initialise Lean goals for {self.current_theorem['name']}: "
                               f"{result.message_objects}")
        self.goals = self._goals(result)
        return self.render()

    @staticmethod
    def _goals(result):
        return [str(msg.get("data", "")).strip() for msg in result.message_objects
                if msg.get("severity") == "error"
                and str(msg.get("data", "")).lstrip().startswith("unsolved goals")]

    def _construct_proof(self, steps):
        return super()._construct_proof([]) + "".join(
            "  " + line + "\n" for step in steps for line in step.splitlines())

    def _call_lean_server(self, proof_text):
        name = self.theorem_name(self.current_theorem)
        return super()._call_lean_server(proof_text + f"\n#print axioms {name}\n")

    @staticmethod
    def _is_outage(result):
        return result.status == "server_error" or (result.status == "error" and result.response is None)

    def _run_lean_query(self, candidate_steps):
        retries = max(int(self.config.server_error_retries), 0)
        for attempt in range(retries + 1):
            result = super()._run_lean_query(candidate_steps)
            if not self._is_outage(result):
                break
            if attempt < retries:
                time.sleep(float(self.config.server_error_backoff) * (2 ** attempt))
        if self._is_outage(result):
            if not self._server_healthy():
                raise RuntimeError(f"Lean verification service failed after {retries + 1} attempts: "
                                   f"{result.message_objects}")
            result.status, result.accepted, result.success = "invalid", False, False
            result.message_objects.append({"severity": "error",
                                           "data": "Lean verification request rejected by a healthy server"})
            return result
        if result.status in ("timeout", "error"):
            return result
        if not isinstance(result.response, dict) or not any(
                key in result.response for key in ("env", "messages", "sorries", "tactics")):
            raise RuntimeError("Malformed Lean verification payload")
        if result.accepted and not result.success and not self._goals(result):
            result.accepted = False
        if result.success and not self._axioms_allowed(result):
            result.accepted = result.success = False
            result.message_objects.append({"severity": "error", "data": "Proof failed the axiom audit"})
        if not result.accepted:
            result.status = "invalid"
        return result

    def _server_healthy(self):
        """Distinguish a content-specific rejection from a real outage with a trivial check."""
        probe = "theorem ragen_health_probe : True := by\n  exact True.intro\n"
        imports = (self.current_theorem or {}).get("imports") or self.config.default_imports
        try:
            response = LeanEnv._call_lean_server(self, f"{imports.strip()}\n\n{probe}")
        except Exception:
            return False
        repl = self._extract_repl_response(response)
        payload = getattr(repl, "response", None)
        return isinstance(payload, dict) and getattr(repl, "error", None) is None and not any(
            str(m.get("severity", "")).lower() == "error" for m in (payload.get("messages") or []))

    def _axioms_allowed(self, result):
        name = re.escape(self.theorem_name(self.current_theorem))
        pattern = rf"'{name}' (does not depend on any axioms|depends on axioms: \[([^\]]*)\])"
        audits = [re.fullmatch(pattern, str(msg.get("data", "")).strip())
                  for msg in result.message_objects if msg.get("severity") == "info"]
        audits = [match for match in audits if match is not None]
        if not audits:
            return False
        axioms = {s.strip() for s in (audits[-1][2] or "").split(",") if s.strip()}
        return axioms <= self.allowed_axioms

    def step(self, action):
        _, reward, done, info = super().step(action)
        self.proof_success = bool(info["success"])
        if info["accepted"]:
            self.goals = self._goals(self.last_result)
        return self.render(), reward, done or not info["accepted"], info

    def render(self, mode=None):
        if self.current_theorem is None:
            return "Lean environment not initialised."
        return (self.current_theorem["formal_statement"] + "\n\nCurrent goals:\n"
                + ("\n\n".join(self.goals) or "No goals."))

    def get_state(self):
        theorem = copy.deepcopy(self.current_theorem)
        theorem["imports"] = theorem.get("imports") or self.config.default_imports
        return LeanSnapshot(theorem, list(self.tactic_history), list(self.goals),
                            self.num_env_steps, self.proof_success, self.config.max_steps)

    def set_state(self, state):
        state = LeanSnapshot.from_dict(state) if isinstance(state, dict) else state
        if (state.max_steps != self.config.max_steps or state.num_env_steps != len(state.tactics)
                or not 0 <= state.num_env_steps <= state.max_steps
                or state.success != (not state.goals)):
            raise ValueError("Invalid or incompatible Lean snapshot")
        current = self.get_state().theorem if self.current_theorem is not None else None
        if current != state.theorem:
            candidates = [dict(row, imports=row.get("imports") or self.config.default_imports)
                          for row in self._dataset]
            if state.theorem not in candidates:
                raise ValueError("Lean snapshot theorem is not in this dataset partition")
        self.current_theorem = copy.deepcopy(state.theorem)
        self.tactic_history = list(state.tactics)
        self.goals = list(state.goals)
        self.proof_success = state.success
        self.num_env_steps = state.num_env_steps
        self.proof_log, self.last_feedback, self.last_result = [], "", None
        self._last_structured_feedback = {}
        self._latest_proof_text = self._construct_proof(self.tactic_history)
        return self.render()
