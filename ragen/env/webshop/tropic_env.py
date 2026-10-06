"""State-conditioned WebShop with isolated sessions and root-replayed snapshots."""

import copy

from webshop_minimal.engine import parse_action
from webshop_minimal.env import SimServer

from .env import WebShopEnv
from .state import WebShopSnapshot, canonical_data, canonical_json, fingerprint
from .tropic_config import WebShopTropicEnvConfig


class WebShopTropicEnv(WebShopEnv):
    def __init__(self, config=None):
        config = config or WebShopTropicEnvConfig()
        if config.observation_mode != "text" or config.max_steps < 1:
            raise ValueError("WebShop TROPIC requires text observations and a positive action budget")
        self.num_env_steps = 0
        self._root_seed, self._mode = None, None
        self._actions = []
        self._done = self._success = False
        super().__init__(config)
        if not isinstance(self.server, SimServer):
            raise ValueError("WebShop TROPIC requires the local SimServer backend")
        if self.server.assigned_instruction_text is not None:
            raise ValueError("WebShop TROPIC does not support mutable instruction overrides")
        if not hasattr(self.server, "_ragen_tropic_catalog_id"):
            self.server._ragen_tropic_catalog_id = fingerprint({
                "goals": self.server.goals,
                "products": self.server.product_item_dict,
                "prices": self.server.product_prices,
                "show_attrs": self.server.show_attrs,
            })
        self.catalog_id = self.server._ragen_tropic_catalog_id

    def _release_session(self):
        session = getattr(self, "session", None)
        if session is not None:
            self.server.user_sessions.pop(session, None)

    def reset(self, seed=None, mode="train", **kwargs):
        if seed is None:  
            return None
        if mode not in ("train", "val", "test") or kwargs:
            raise ValueError("WebShop TROPIC resets require a seed and an original split mode")
        if len(self.server.goals) <= 1500:
            raise ValueError("WebShop's original split requires more than 1500 goals")
        self._release_session()
        self._root_seed, self._mode = int(seed), mode
        self.num_env_steps, self._actions = 0, []
        self._done = self._success = False
        super().reset(seed=int(seed), mode=mode)
        session = self.server.user_sessions[self.session]
        session["goal"] = copy.deepcopy(session["goal"])
        return self.render()

    def step(self, action):
        if self._root_seed is None or self._done or self.num_env_steps >= self.config.max_steps:
            raise RuntimeError("WebShop TROPIC cannot act before reset or after episode termination")
        name, argument = parse_action(action)
        random_search = (name == "search" and argument and argument.lower().split()[:1] == ["<r>"])
        malformed = argument is None or not argument.strip() or action != f"{name}[{argument}]"
        if random_search or malformed:
            reward, done = 0.0, True
            info = {"raw_reward": 0.0, "action_is_valid": False, "action_is_effective": False,
                    "success": False, "success_purchase": False, "success_find": False}
        else:
            _, reward, done, info = super().step(action)
        self.num_env_steps += 1
        self._actions.append(action)
        self._success = bool(info["success"])
        self._done = bool(done or not info["action_is_valid"] or self.num_env_steps >= self.config.max_steps)
        return self.render(), reward, self._done, info

    def prepare_render_cache(self, observation):
        self.render_cache = observation

    def render(self, mode=None):
        if self._root_seed is None:
            return "WebShop environment not initialised."
        if self._done:
            return f"Shopping episode finished. Success: {int(self._success)}."
        options = self.server.user_sessions[self.session]["options"]
        actions = self.get_available_actions()
        page = self.server.get_page_name(self.browser.current_url)
        remaining = self.config.max_steps - self.num_env_steps
        guidance = ["Choose exactly ONE of the available actions for this response."]
        if not any(action.startswith("search[") for action in actions):
            guidance.append("There is no search bar on this page.")
        if "click[buy now]" in actions:
            guidance.append("Selected options are already applied; do not select them again. "
                            "If several requested options are missing, select just ONE missing option now "
                            "and wait for the next page. "
                            "If the requested options are selected, buy now.")
        return (f"Current page: {page}\nActions remaining: {remaining}\n"
                f"Selected options: {canonical_json(options)}\n\n"
                + super().render() + "\n\nAvailable actions:\n"
                + "\n".join(actions) + "\n\n" + " ".join(guidance))

    def get_state(self):
        if self._root_seed is None:
            raise RuntimeError("Reset WebShop before taking a snapshot")
        state = canonical_data({
            "session": self.server.user_sessions[self.session],
            "page": self.server.get_page_name(self.browser.current_url),
            "instruction": self.instruction_text,
            "observation": self.render(),
            "available_actions": [] if self._done else self.get_available_actions(),
            "done": self._done,
            "success": self._success,
        })
        return WebShopSnapshot(self.catalog_id, self._root_seed, self._mode, list(self._actions),
                               state, self.num_env_steps, self.config.max_steps)

    def set_state(self, snapshot):
        snapshot = WebShopSnapshot.from_dict(snapshot) if isinstance(snapshot, dict) else snapshot
        if (snapshot.catalog_id != self.catalog_id or snapshot.max_steps != self.config.max_steps
                or snapshot.num_env_steps != len(snapshot.actions)
                or not 0 <= snapshot.num_env_steps <= snapshot.max_steps
                or snapshot.mode not in ("train", "val", "test")
                or (self._mode is not None and snapshot.mode != self._mode)
                or any(not isinstance(action, str) for action in snapshot.actions)):
            raise ValueError("Invalid or incompatible WebShop snapshot")
        self.reset(seed=snapshot.root_seed, mode=snapshot.mode)
        for action in snapshot.actions:
            self.step(action)
        if self.get_state().state != snapshot.state:
            raise ValueError("WebShop snapshot failed root replay; catalog or transition changed")
        return self.render()

    def close(self):
        self._release_session()
        super().close()
