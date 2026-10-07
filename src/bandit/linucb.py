"""The LinUCB learners of the bandit policy.

The reported runs use `WarmLinUCB` with `discount_mode="pass"` and no warm start. Each action keeps
discounted sums of the outer products of its contexts and of reward times context, and once per
maintenance pass `end_of_pass` discounts every action by 0.99, which is D-LinUCB with the pass as
the round. The ridge term stays at the identity. `LinUCBAgent` is an earlier learner that discounts
an action only when that action is updated, and the reported runs do not use it.
"""

from __future__ import annotations

import numpy as np


class LinUCBAgent:
    """Disjoint LinUCB with discounted updates, D-LinUCB of Russac, Vernade and Cappe, 2019.

    Not used by the reported runs. For each action a, theta_a is V_a inverse times b_a, and the
    confidence width is alpha times the square root of x V_a^-1 Vt_a V_a^-1 x, where Vt_a is
    discounted at gamma squared. With a discount below 1 this is the width D-LinUCB bounds, while
    the plain width of stationary LinUCB would overstate the variance. alpha = 1.0 is the
    practical choice of Li et al., 2010, not a theoretical value. Unlike the paper, an action is
    discounted only when it is updated, so an action that is rarely chosen forgets more slowly.
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        alpha: float = 1.0,
        discount: float = 0.99,
    ) -> None:
        """Create a learner with `n_arms` actions, each starting at the identity ridge term."""
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.discount = discount
        self.A: list[np.ndarray] = [np.eye(context_dim, dtype=np.float64) for _ in range(n_arms)]
        self.A_tilde: list[np.ndarray] = [
            np.eye(context_dim, dtype=np.float64) for _ in range(n_arms)]
        self.b: list[np.ndarray] = [np.zeros(context_dim, dtype=np.float64) for _ in range(n_arms)]
        self._update_count: int = 0

    def scores(self, context: np.ndarray) -> np.ndarray:
        """Return the upper confidence score of every action for `context`."""
        x = context.astype(np.float64)
        eye = np.eye(self.context_dim, dtype=np.float64)
        out = np.empty(self.n_arms, dtype=np.float64)
        for a in range(self.n_arms):
            a_inv = np.linalg.solve(self.A[a], eye)
            theta = a_inv @ self.b[a]
            y = a_inv @ x
            # never below zero in exact arithmetic, but rounding can make it slightly negative
            width = max(0.0, float(y @ self.A_tilde[a] @ y))
            out[a] = float(theta @ x) + self.alpha * np.sqrt(width)
        return out

    def select_arm(self, context: np.ndarray, available: list[int] | None = None) -> int:
        """Return the action with the highest score, chosen among `available` if it is given.

        As in Li et al., Algorithm 1, the choice is over the actions allowed in the current round,
        so the learner is never charged for, or taught by, an action that could not be taken.
        """
        s = self.scores(context)
        if available is not None:
            if not available:
                raise ValueError("no arm is available")
            return int(max(available, key=lambda a: s[a]))
        return int(np.argmax(s))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """Learn the reward of one action, discounting that action's past evidence first.

        The ridge term must stay at the identity, as in Li et al., equation 5. Discounting the
        whole matrix would shrink it over a run, so (1 - g) times the identity is added back after
        each discount, which keeps it exactly at the identity while the data still fades at the
        chosen rate. Vt gets the same treatment at g squared.
        """
        x = context.astype(np.float64)
        eye = np.eye(self.context_dim, dtype=np.float64)
        xx = np.outer(x, x)
        g = self.discount
        self.A[arm] = g * self.A[arm] + xx + (1.0 - g) * eye
        self.A_tilde[arm] = g * g * self.A_tilde[arm] + xx + (1.0 - g * g) * eye
        self.b[arm] = g * self.b[arm] + reward * x
        self._update_count += 1

    @property
    def total_updates(self) -> int:
        """The number of updates made so far."""
        return self._update_count

    def theta(self, arm: int) -> np.ndarray:
        """Return the current estimate of an action's weight vector."""
        a_inv = np.linalg.solve(self.A[arm], np.eye(self.context_dim, dtype=np.float64))
        return a_inv @ self.b[arm]


# the weights over warm start and real feedback tried by WarmLinUCB, following ARROW-CB of Zhang
# et al., 2019, Algorithm 1, which picks a weight online from a set that includes 0 and 1. their
# spacing belongs to epsilon greedy and has no counterpart in LinUCB, so this spacing is chosen
# here, eight values like theirs, closer together toward the warm start end
ARROW_LAMBDAS: tuple[float, ...] = (0.0, 0.01, 0.03, 0.1, 0.3, 0.5, 0.7, 1.0)


class WarmLinUCB:
    """Disjoint LinUCB with an optional warm start, weighted as in ARROW-CB, and a choice of clock.

    Each action keeps real feedback in the sums G, G2 and h and warm start estimates in S, S2 and
    hs, so the model for a weight lambda is V = I + (1 - lambda) S + lambda G, Vt = I +
    (1 - lambda)^2 S2 + lambda^2 G2 and b = (1 - lambda) hs + lambda h, with lambda weighting the
    real feedback as in ARROW-CB equation 3. Before each real outcome is learned, the model of
    every lambda predicts it, the squared errors add up, and the lambda with the least error
    drives the next decision. The reported runs use no warm start and a single lambda of 1.

    With `discount_mode` "arm" an action's evidence is discounted when that action is updated.
    With "pass", which the reported runs use, every action is discounted once per maintenance
    pass whether it was chosen or not, as D-LinUCB discounts every round, so an action left
    unchosen regains uncertainty and is tried again. The memory is then measured in stream time,
    not in the number of candidates a pass looks at.
    """

    def __init__(
        self,
        n_arms: int,
        context_dim: int,
        alpha: float = 1.0,
        discount: float = 0.99,
        discount_mode: str = "pass",
        lambdas: tuple[float, ...] = (1.0,),
    ) -> None:
        """Create a learner with `n_arms` actions and no evidence yet."""
        if discount_mode not in ("arm", "pass"):
            raise ValueError(discount_mode)
        self.n_arms = n_arms
        self.context_dim = context_dim
        self.alpha = alpha
        self.discount = discount
        self.discount_mode = discount_mode
        self.lambdas = tuple(sorted(lambdas))
        d = context_dim
        z = lambda: [np.zeros((d, d)) for _ in range(n_arms)]  # noqa: E731
        v = lambda: [np.zeros(d) for _ in range(n_arms)]  # noqa: E731
        self.G, self.G2, self.h = z(), z(), v()
        self.S, self.S2, self.hs = z(), z(), v()
        self.err = np.zeros(len(self.lambdas))
        # before any real outcome every lambda has zero error and the tie goes to the smallest,
        # so a warm started learner begins on its warm start
        self.sel = 0
        self._update_count = 0
        self.warm_samples = 0

    @property
    def lam(self) -> float:
        """The weight on real feedback currently in use."""
        return self.lambdas[self.sel]

    @property
    def total_updates(self) -> int:
        """The number of real outcomes learned so far."""
        return self._update_count

    def add_warm(self, arm: int, contexts: np.ndarray, rewards: np.ndarray) -> None:
        """Add warm start examples for an action, estimated rewards rather than observed ones."""
        x = np.asarray(contexts, dtype=np.float64)
        r = np.asarray(rewards, dtype=np.float64)
        if len(x) == 0:
            return
        xx = x.T @ x
        self.S[arm] += xx
        self.S2[arm] += xx
        self.hs[arm] += x.T @ r
        self.warm_samples += len(x)

    def _model(self, arm: int, lam: float):
        """Return V, Vt and b of an action for the weight `lam`."""
        eye = np.eye(self.context_dim)
        v = eye + (1.0 - lam) * self.S[arm] + lam * self.G[arm]
        vt = eye + (1.0 - lam) ** 2 * self.S2[arm] + lam ** 2 * self.G2[arm]
        b = (1.0 - lam) * self.hs[arm] + lam * self.h[arm]
        return v, vt, b

    def theta(self, arm: int, lam: float | None = None) -> np.ndarray:
        """Return the weight vector estimate of an action, for `lam` or the current weight."""
        v, _, b = self._model(arm, self.lam if lam is None else lam)
        return np.linalg.solve(v, b)

    def scores(self, context: np.ndarray) -> np.ndarray:
        """Return the upper confidence score of every action for `context`."""
        x = context.astype(np.float64)
        out = np.empty(self.n_arms)
        for a in range(self.n_arms):
            v, vt, b = self._model(a, self.lam)
            theta = np.linalg.solve(v, b)
            y = np.linalg.solve(v, x)
            out[a] = float(theta @ x) + self.alpha * np.sqrt(max(0.0, float(y @ vt @ y)))
        return out

    def select_arm(self, context: np.ndarray, available: list[int] | None = None) -> int:
        """Return the action with the highest score, chosen among `available` if it is given."""
        s = self.scores(context)
        if available is not None:
            if not available:
                raise ValueError("no arm is available")
            return int(max(available, key=lambda a: s[a]))
        return int(np.argmax(s))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """Learn the observed reward of one action in one context."""
        x = context.astype(np.float64)
        if len(self.lambdas) > 1:
            # each weight predicts the outcome before it is learned
            for i, lam in enumerate(self.lambdas):
                self.err[i] += (float(self.theta(arm, lam) @ x) - reward) ** 2
            self.sel = int(np.argmin(self.err))
        g = self.discount
        if self.discount_mode == "arm":
            self.G[arm] *= g
            self.G2[arm] *= g * g
            self.h[arm] *= g
            self.S[arm] *= g
            self.S2[arm] *= g * g
            self.hs[arm] *= g
        xx = np.outer(x, x)
        self.G[arm] += xx
        self.G2[arm] += xx
        self.h[arm] += reward * x
        self._update_count += 1

    def end_of_pass(self) -> None:
        """Discount the evidence of every action once, at the end of a maintenance pass.

        Only in "pass" mode. The record of which weight predicts best is discounted too, since the
        best weight can change as the index changes.
        """
        if self.discount_mode != "pass":
            return
        g = self.discount
        for a in range(self.n_arms):
            self.G[a] *= g
            self.G2[a] *= g * g
            self.h[a] *= g
            self.S[a] *= g
            self.S2[a] *= g * g
            self.hs[a] *= g
        self.err *= g
