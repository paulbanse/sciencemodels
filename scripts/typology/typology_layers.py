"""Path D typology layers (P-0005, 2026-11-15): driver-side subclass of the
baseline ABM implementing the destruction and counterforce channels.

Write boundary of the Path D round: ``sciencemodels/*.py`` is NOT touched.
Both channels are implemented here, in the driver tree, as a subclass of
``Scientist`` and ``MyModel``; the commission's bit-identity proof at
gamma=0, kappa=0 is the round's identity gate (see ``identity_proof``).

Channels (formulas as commissioned, P-0005 / task header 2026-11-15):

* Destruction (gamma, ``AP_F -| G``): when ``m_t(pos)`` agents occupy the
  same tile, the effective perception noise applied to that tile's novelty
  read scales as ``eps_eff = eps * m_t(pos)**gamma``.  Crowding in hot
  fields degrades signal-to-noise.  Implemented at the single epsilon
  dynamics site (``sciencemodels/agents.py`` step, the neighbour-novelty
  line ``neighbor_rw_novelty * (1 + self.epsilon * noise)``).
* Counterforce (kappa, ``P -> G``): a newcomer's perception noise is
  reduced by its supervisor's prestige, ``eps_i = eps_0 / (1 + kappa *
  P_sup)``.  ``P_sup`` is pinned as the supervisor's entry-stock prestige
  (``model.entry_prestige_stock``, pinned to "prestige" in this round) AT
  THE NEWCOMER'S BIRTH STEP, stored once on the agent; the divisor is then
  fixed for the agent's life.  Founders have no supervisor and are kappa-
  inert.  At g_entry=0.0 no newcomers exist, so kappa is inert by
  construction there (used as the round's built-in control arm).

Declared pins (disclosed, not silently chosen):
- ``m_t(pos)`` counts all scientists whose current position equals the
  CANDIDATE tile, evaluated at the focal agent's evaluation time (others
  do not move during one agent's step, so one per-step occupancy map is
  exact).  The focal agent is never on a candidate tile (candidates are
  von Neumann neighbours, centre excluded).
- Literal formula: at gamma=1 an EMPTY candidate tile (m=0) receives
  eps_eff = 0 (perfect read).  This is the commissioned formula's direct
  consequence - crowding amplifies noise, absence of crowding removes the
  amplification - and is disclosed rather than "corrected" to a floor.
- The current tile's own reward is computed WITHOUT noise in the parent
  (unchanged); the channels touch candidate evaluation only.

Name-mangling warning (the reason this module exists in its current shape):
``MyModel.end_loop_update`` calls ``self.__generate_new_agents()``, which
Python compiles to ``self._MyModel__generate_new_agents``.  A normally
named subclass override would be silently invisible.  This module
therefore defines the mangled name explicitly; the runtime identity proof
in ``identity_proof`` is the guard that this interception really reproduces
the parent's newborn construction (roulette order included).
"""
from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from sciencemodels.agents import Scientist
from sciencemodels.model import MyModel

REPO = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Divergence canary: the source markers the subclass relies on.  If the
# shared tree moves these lines (owner rounds are live), the round fails
# loudly here instead of silently diverging (ADR-0001 in-run canary).
# ---------------------------------------------------------------------------

AGENTS_MARKERS = (
    "neighbor_rw_novelty * (1 + self.epsilon * noise)",  # the single epsilon site
    "for neighbor in neighbors_nodes:",
    "if best_reward - total_current_reward > 0:",
    "self.model.farm(self.pos, self)",
)
MODEL_MARKERS = (
    "def _init_agents(self, agent_generation_function):",
    "def __generate_new_agents(self):",
    "a = Scientist(self, w, self.epsilon)",
    "a.is_newcomer = True",
    "self.datacollector.collect(self)",
)


def source_canary():
    """Assert the parent source still carries the lines this subclass copies
    or intercepts.  Returns the sha256 of both files for design.json."""
    import hashlib

    agents_src = (REPO / "sciencemodels" / "agents.py").read_text()
    model_src = (REPO / "sciencemodels" / "model.py").read_text()
    for marker in AGENTS_MARKERS:
        assert marker in agents_src, "agents.py moved: %r" % marker
    for marker in MODEL_MARKERS:
        assert marker in model_src, "model.py moved: %r" % marker
    # the epsilon site must be UNIQUE: a second site would need the channel too
    assert agents_src.count("self.epsilon * noise") == 1, (
        "epsilon now enters the dynamics at more than one site - re-derive"
    )
    # exactly two Scientist construction sites (founders + newcomers)
    assert model_src.count("a = Scientist(self, w, self.epsilon)") == 2, (
        "Scientist construction sites changed - re-derive the subclass"
    )
    hashes = {}
    for name in ("model.py", "agents.py", "metrics.py", "io.py", "landscapes.py"):
        p = REPO / "sciencemodels" / name
        hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return hashes


def _numeric(v):
    return (
        isinstance(v, (int, float, np.integer, np.floating))
        and not isinstance(v, bool)
    )


def compare_rows(r_base, r_sub):
    """Type-robust row comparison for the gates: build_report mixes scalars
    with dict/list values (generation_params, teams, ...).  Numeric keys are
    compared by abs diff; every other type by equality.  Returns
    (worst_numeric_diff, worst_key, mismatched_non_numeric_keys)."""
    worst, worst_key, mismatches = 0.0, None, []
    for k in r_base:
        vb, vs = r_base[k], r_sub[k]
        if _numeric(vb) and _numeric(vs):
            d = abs(float(vb) - float(vs))
            if d > worst:
                worst, worst_key = d, k
        elif vb != vs:
            mismatches.append(k)
    return worst, worst_key, mismatches


# ---------------------------------------------------------------------------
# The two channels
# ---------------------------------------------------------------------------


class TypologyScientist(Scientist):
    """Scientist with the destruction (gamma) and counterforce (kappa)
    perception-noise channels.  Dynamics-neutral at gamma=kappa=0: the
    helpers short-circuit to the parent's exact float operations."""

    def __init__(self, model, curiosity, epsilon, prestige_sup=None):
        super().__init__(model, curiosity, epsilon)
        # supervisor's entry-stock prestige at this agent's birth step;
        # None for founders (kappa inert for them)
        self.typology_prestige_sup = (
            None if prestige_sup is None else float(prestige_sup)
        )

    def _eps_base(self):
        """Counterforce divisor.  Short-circuits at kappa=0 to the parent's
        own attribute (no arithmetic at all -> bit-identity)."""
        kappa = self.model.typology_kappa
        if kappa == 0.0 or self.typology_prestige_sup is None:
            return self.epsilon
        return self.epsilon / (1.0 + kappa * self.typology_prestige_sup)

    def _eps_eff(self, m_count):
        """Destruction scaling.  Short-circuits at gamma=0."""
        base = self._eps_base()
        gamma = self.model.typology_gamma
        if gamma == 0.0:
            return base
        return base * (m_count ** gamma)

    def _tile_occupancy(self):
        """Agent count per tile at evaluation time.  Others do not move
        during one focal agent's step, so one map per agent-step is exact."""
        occ = {}
        for a in self.model._scientists:
            occ[a.pos] = occ.get(a.pos, 0) + 1
        return occ

    def step(self):
        """Verbatim copy of ``Scientist.step`` (sciencemodels/agents.py,
        current tree) with ONE change: the neighbour-novelty noise term
        uses the channel-scaled epsilon.  The rng call sequence is
        unchanged; the identity proof verifies the consequence."""
        neighbors_nodes = self.model.grid.get_neighborhood(
            self.pos, moore=False, include_center=False
        )
        # optionpos = self.model.rng.choice(neighbors_nodes)
        current_rw_novelty = self.model.compute_knowledge(self.pos)
        total_current_reward, currentVis = self.compute_all_rewards(
            self.pos, current_rw_novelty
        )
        best_reward = total_current_reward
        optionpos = self.pos  # fallback to current position
        best_vis = currentVis

        occ = self._tile_occupancy()
        for neighbor in neighbors_nodes:
            neighbor_rw_novelty = self.model.compute_knowledge(neighbor)
            noise = self.rng.uniform(-1.0, 1.0)
            m_here = occ.get(neighbor, 0)
            total_reward, visibility = self.compute_all_rewards(
                neighbor,
                neighbor_rw_novelty * (1 + self._eps_eff(m_here) * noise),
            )

            if total_reward > best_reward:
                best_reward = total_reward
                optionpos = neighbor
                best_vis = visibility

        if best_reward - total_current_reward > 0:
            self.model.new_place(self, optionpos)
            self.visibility = best_vis
        else:
            self.visibility = currentVis

        knowledge_before = self.model.compute_knowledge(self.pos)
        self.model.farm(self.pos, self)  # farming also updates AvgAgentKnowledge

        self.age += 1
        self.last_tile_knowledge = self.model.grid.properties["knowledge"].data[
            self.pos
        ]
        if self.model.credit_sharing == "equal_split":
            # Path B Q8: buffer this step's accrual; the model pools and
            # splits it among linked co-located agents in end_loop_update.
            # (Accrual is therefore deferred to end-of-step under this rule
            # only - a mechanics change inherent to pooling, disclosed.)
            if self.model.prestige_accrual == "harvested":
                self._pending_prestige = np.float64(
                    knowledge_before * self.model.harvest
                )
            else:  # "post_tile": legacy rule, accrues the post-harvest tile value
                self._pending_prestige = np.float64(self.last_tile_knowledge)
            self._pending_tile = self.pos
        elif self.model.prestige_accrual == "harvested":
            # accrue the harvested flow (knowledge just before farming * harvest)
            self.increase_prestige(knowledge_before * self.model.harvest)
        else:  # "post_tile": legacy rule, accrues the post-harvest tile value
            self.increase_prestige(self.last_tile_knowledge)


class TypologyModel(MyModel):
    """MyModel with the two typology channels wired through the two agent
    construction sites (founders, newborns).  gamma=kappa=0 is bit-identical
    to MyModel (proven at runtime by ``identity_proof``)."""

    def __init__(self, typology_gamma=0.0, typology_kappa=0.0, **kw):
        if not np.isfinite(typology_gamma) or typology_gamma < 0.0:
            raise ValueError(
                "typology_gamma must be finite and non-negative, got "
                + repr(typology_gamma)
            )
        if not np.isfinite(typology_kappa) or typology_kappa < 0.0:
            raise ValueError(
                "typology_kappa must be finite and non-negative, got "
                + repr(typology_kappa)
            )
        # set BEFORE super().__init__: the parent constructor already runs
        # end_loop_update (newborn generation) at its end
        self.typology_gamma = np.float64(typology_gamma)
        self.typology_kappa = np.float64(typology_kappa)
        super().__init__(**kw)

    def _init_agents(self, agent_generation_function):
        """Verbatim copy of MyModel._init_agents with the class swapped."""
        for _ in range(self.number_agents):
            w = agent_generation_function(
                self.initial_curiosity, self.rng, self.generation_params
            )
            a = TypologyScientist(self, w, self.epsilon)
            coords = (self.rng.integers(0, self.size), self.rng.integers(0, self.size))
            a.age = self.rng.choice(np.arange(23, 50))
            self.grid.place_agent(a, coords)
            self.grid.properties["explored"].data[coords] = True
            a.last_tile_knowledge = self.grid.properties["knowledge"].data[coords]
            a.current_local_merit = a.local_merit(a.pos)

        self._scientists = [
            agent for agent in self.agents if isinstance(agent, Scientist)
        ]

    # Name-mangling interception: MyModel.end_loop_update calls
    # self.__generate_new_agents() == self._MyModel__generate_new_agents().
    # A normally named override would never run.  Defining the mangled name
    # explicitly is the minimal, honest interception; identity_proof guards
    # its fidelity (roulette order and rng consumption included).
    def _MyModel__generate_new_agents(self):
        """Verbatim copy of MyModel.__generate_new_agents with the class
        swapped and the supervisor's entry-stock prestige stamped on the
        newborn (the counterforce's P_sup)."""
        if self.agent_generation_time > 0:
            agents_to_generate: int = np.floor(
                self.steps * self.agent_generation_time
            ) - np.floor((self.steps - 1) * self.agent_generation_time)
            entry_stock = self.entry_prestige_stock
            prestige_agent_tuples = [
                (getattr(a, entry_stock), a) for a in self._scientists
            ]
            totalPrestige = sum([getattr(a, entry_stock) for a in self._scientists])
            supervisors = []
            for _ in range(int(agents_to_generate)):
                val = self.rng.uniform(0, totalPrestige)
                for p, a in prestige_agent_tuples:
                    if val <= p:
                        supervisors.append(a)
                        break
                    val -= p
            for sup in supervisors:
                w = self.agent_generation_function(
                    self.initial_curiosity, self.rng, self.generation_params
                )
                a = TypologyScientist(self, w, self.epsilon)
                # accumulator init: covers the birth-step collection and
                # final-wave newcomers, for whom no updateIdealMerit follows
                # their birth (agents.py is outside this round's edit boundary)
                a.ideal_merit_cum = np.float64(0.0)
                # Path B Q5: permanent newcomer flag (proposal regime)
                a.is_newcomer = True
                # Path D counterforce: supervisor's entry-stock prestige at
                # the birth step, fixed for the newborn's life.  sup is never
                # None under the parent's roulette (val=0 <= first p=0 also
                # selects), but the guard keeps kappa inert if that changes.
                a.typology_prestige_sup = float(getattr(sup, entry_stock))
                coords = sup.pos
                self.grid.place_agent(a, coords)
                self._scientists.append(a)
            for agent in self._scientists:
                self.grid.properties["explored"].data[agent.pos] = True


# ---------------------------------------------------------------------------
# Round gates: identity proof, binding probe
# ---------------------------------------------------------------------------


def _run_extras(m):
    """The driver's per-run extras (final-step correlations etc.), shared by
    the gates so they prove exactly what the sweep writes."""
    dc = m.datacollector.get_model_vars_dataframe()
    last = dc.iloc[-1]
    row = {}
    for src, dst in (
        ("Corr_prestige_ideal_merit", "c_ideal_tm"),
        ("Corr_prestige_local_merit", "c_local_tm"),
    ):
        row[dst] = float(dc[src].mean())
        row[dst.replace("_tm", "_fin")] = float(last.get(src, np.nan))
    row["wedge_fin"] = row["c_ideal_fin"] - row["c_local_fin"]
    row["wedge_tm"] = row["c_ideal_tm"] - row["c_local_tm"]
    row["ideal_loss_cum"] = -float(
        sum(getattr(a, "ideal_merit_cum", 0.0) for a in m._scientists)
    )
    row["agents_final"] = len(m._scientists)
    row["newcomers_final"] = int(sum(1 for a in m._scientists if a.is_newcomer))
    return row


def _build_row(m):
    row = dict(m.get_report())
    row.update(_run_extras(m))
    return row


def identity_proof(base_params, seeds=(0, 1, 2, 3, 4), entry_levels=(0.0, 0.1)):
    """Commission step 2: (gamma=0, kappa=0) must match MyModel bit-for-bit.

    Runs the plain model and the subclass at both entry levels for each
    seed and compares EVERY report key plus the driver extras.  Any nonzero
    difference raises.  Returns a summary dict for design.json/canary.txt.
    """
    from sciencemodels import landscapes

    out = {"seeds": list(seeds), "entry_levels": list(entry_levels), "max_abs_diff": 0.0}
    for g_entry in entry_levels:
        for seed in seeds:
            p = dict(base_params)
            p.update(
                initial_curiosity=0.5,
                cell_init_function=landscapes.multipleGaussians,
                generation_params={"std": 5, "number_gaussians": 2},
                agent_seed=seed,
                agent_generation_time=g_entry,
            )
            m_base = MyModel(**p)
            for _ in range(200):
                m_base.step(True)
            m_sub = TypologyModel(typology_gamma=0.0, typology_kappa=0.0, **p)
            for _ in range(200):
                m_sub.step(True)
            r_base, r_sub = _build_row(m_base), _build_row(m_sub)
            assert set(r_base) == set(r_sub), "schema drift between base and subclass"
            worst, worst_key, mismatches = compare_rows(r_base, r_sub)
            if worst != 0.0 or mismatches:
                raise AssertionError(
                    "IDENTITY PROOF FAILED at g=%s seed=%s: worst numeric key %s"
                    " diff %r; non-numeric mismatches %s"
                    % (g_entry, seed, worst_key, worst, mismatches)
                )
            out["max_abs_diff"] = max(out["max_abs_diff"], worst)
    out["n_comparisons"] = len(seeds) * len(entry_levels)
    out["status"] = "PASS: subclass (gamma=0, kappa=0) bit-identical to MyModel on every report key"
    return out


def binding_probe(base_params, seeds=(0, 1)):
    """Binding + control probes for the two channels.

    Returns a dict of paired contrasts.  Asserts:
    * gamma binds: (1,0) vs (0,0) differs on the report surface at g=0.1;
    * kappa binds at g=0.1 (newcomers exist) and is BIT-IDENTICAL at g=0.0
      (no newcomers -> the channel has no object; the round's built-in
      control arm, quoted in the report).
    """
    from sciencemodels import landscapes

    def run(gamma, kappa, g_entry, seed):
        p = dict(base_params)
        p.update(
            cell_init_function=landscapes.multipleGaussians,
            generation_params={"std": 5, "number_gaussians": 2},
            agent_seed=seed,
            agent_generation_time=g_entry,
            initial_curiosity=0.5,
        )
        m = TypologyModel(typology_gamma=gamma, typology_kappa=kappa, **p)
        for _ in range(200):
            m.step(True)
        return _build_row(m)

    res = {}
    for g_entry in (0.0, 0.1):
        for seed in seeds:
            r00 = run(0.0, 0.0, g_entry, seed)
            r10 = run(1.0, 0.0, g_entry, seed)
            r01 = run(0.0, 1.0, g_entry, seed)
            w_g, _, mm_g = compare_rows(r00, r10)
            w_k, _, mm_k = compare_rows(r00, r01)
            res[(g_entry, seed)] = {
                "gamma_delta_wedge_fin": r10["wedge_fin"] - r00["wedge_fin"],
                "gamma_delta_c_ideal_tm": r10["c_ideal_tm"] - r00["c_ideal_tm"],
                "gamma_delta_ideal_loss_cum": r10["ideal_loss_cum"] - r00["ideal_loss_cum"],
                "kappa_delta_wedge_fin": r01["wedge_fin"] - r00["wedge_fin"],
                "kappa_delta_c_ideal_tm": r01["c_ideal_tm"] - r00["c_ideal_tm"],
                "kappa_delta_ideal_loss_cum": r01["ideal_loss_cum"] - r00["ideal_loss_cum"],
                "gamma_any_key_differs": bool(w_g > 0.0 or mm_g),
                "kappa_any_key_differs": bool(w_k > 0.0 or mm_k),
                "kappa_bitidentical_all_keys": bool(w_k == 0.0 and not mm_k),
            }
    # assertions
    for (g_entry, seed), d in res.items():
        assert d["gamma_any_key_differs"], "gamma INERT at g=%s seed=%s" % (g_entry, seed)
        if g_entry == 0.0:
            assert d["kappa_bitidentical_all_keys"], (
                "kappa must be bit-identical at g=0.0 (no newcomers); control arm broken"
            )
        else:
            assert d["kappa_any_key_differs"], "kappa INERT at g=0.1 - wiring bug"
    return res
