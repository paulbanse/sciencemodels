"""Metric computations that separate cleanly from the model class.

Extracted verbatim from the pre-restructure ``mesa_model.py``:

- ``local_merit``        (was ``Scientist.local_merit``)
- ``update_ideal_merit`` (was ``MyModel.updateIdealMerit``)
- ``correlation_reporter`` (was a closure defined inside ``MyModel.__init__``)
- ``update_step_goals``  (was ``MyModel._update_step_goals``, milestone logic)

Each function takes the model (or the relevant objects) as an argument; the
original methods on ``Scientist`` / ``MyModel`` remain as thin delegates so
the public API is unchanged.

Entanglement notes (methods that could not become pure functions without a
logic change):

- ``update_ideal_merit`` mutates the model's ``previous_knowledge`` property
  layer in place while computing the ideal harvest and writes each agent's
  ``ideal_Merit`` attribute; it is therefore a function *of the model* and is
  driven from ``MyModel.updateIdealMerit``.
- ``update_step_goals`` writes the model's milestone attributes
  (``explored_50_step`` ... ``harvested_25_step``); same treatment.
- ``local_merit`` needs ``model.grid`` and ``model.compute_knowledge``; it is
  a function of the model and a position, with ``Scientist.local_merit``
  delegating to it.
- The core dynamics (``step``, ``farm``, ``end_loop_update``, ``new_place``,
  ``compute_knowledge``, ``update_knowledge``, newcomer generation and new
  questions) remain methods on ``MyModel``: they *are* the model class.
"""
from __future__ import annotations

import numpy as np


def local_merit(model, pos) -> np.float64:
    neighbors = model.grid.get_neighborhood(
        pos, moore=False, include_center=False
    )

    # positions: center + 4-neighbors
    positions = [pos, *neighbors]

    # compute into a NumPy array (float64)
    knowledge = np.fromiter(
        (model.compute_knowledge(p) for p in positions),
        dtype=np.float64,
        count=len(positions),
    )

    return knowledge.mean()


def update_ideal_merit(model):
    """ computes the loss for each agent compared to the best position they could have in the ideal harvest situation"""
    temp_knowledge_map= model.grid.properties["previous_knowledge"].data
    
    # compute the harvest made
    IdealHarvestList = []
    L = [(a.pos, a) for a in model.agents]
    L = sorted(L, key=lambda L: L[0])  #sort for positions so same positions are together
    ActualHarvestList = []
    pastpos, redone = -1,0
    for kpos,k in L:
        know = temp_knowledge_map[kpos]
        if kpos == pastpos:
            redone += 1
        else:
            redone = 0
        know = know * ((1- model.harvest) ** redone)
        pastpos = kpos
        ActualHarvestList.append([know*model.harvest, kpos, k])
    
    #compute ideal harvest: greedy allocation, one slot per agent in the
    #model (under entry the population exceeds number_agents, so the slot
    #count follows len(model.agents))
    for k in range(len(model.agents)):
        pos = np.unravel_index(temp_knowledge_map.argmax(), temp_knowledge_map.shape) 
        IdealHarvestList.append((temp_knowledge_map[pos]*model.harvest,pos))
        temp_knowledge_map[pos] = temp_knowledge_map[pos] * (1- model.harvest)
    
    #compute loss per agent: sorted-to-sorted pairing - the k-th largest
    #actual harvest is matched against the k-th largest greedy-ideal
    #harvest. (Pre-fix this paired every actual harvest against the ideal
    #list's static head, which only advanced on an exact-zero loss, so
    #below-head agents were measured against the single best tile.)
    ActualHarvestList = sorted(ActualHarvestList, key= lambda L: L[0], reverse=True)
    IdealHarvestList = sorted(IdealHarvestList, key= lambda L: L[0], reverse=True)
    for k in range(len(model.agents)):
        loss = ActualHarvestList[k][0] - IdealHarvestList[k][0]
        ActualHarvestList[k] = ActualHarvestList[k] + [loss]
    
    # average loss per position
    decided_pos = []
    for (kharvested, kpos, k, loss) in ActualHarvestList:
        if kpos not in decided_pos:
            agents_on_site = [a[2] for a in ActualHarvestList if a[1]== kpos]
            loss_on_site = sum([a[3] for a in ActualHarvestList if a[1]== kpos])
            for a in agents_on_site:
                a.ideal_Merit = loss_on_site/(len(agents_on_site))
            decided_pos.append(kpos)
    
    # cumulative per-agent ideal-merit loss over the run (welfare-round
    # need); lazy init keeps agents.py outside the edit boundary
    for a in model.agents:
        a.ideal_merit_cum = getattr(a, "ideal_merit_cum", 0.0) + a.ideal_Merit
    
    model.grid.properties["previous_knowledge"].data = model.grid.properties["knowledge"].data.copy()


# ---------------------------------------------------------------------------
# Path B: the dyadic collaborator-choice layer (design menu
# AdmiralsLog/archive/path_b_dyadic_sketch_v1.md; owner rulings Q1-Q8 in the Path B brief).
# compute_dyadic_scores and compute_teams are measures; dyadic_step mutates
# model.links (the declared choice layer) and is driven from
# MyModel.end_loop_update. Everything here is guarded: with
# dyadic_mode="off" no rng is consumed and no dynamic state changes.


def _normalize_minmax(values):
    """Scale to [0, 1]; a degenerate range maps to the neutral 0.5."""
    v = np.asarray(values, dtype=np.float64)
    lo, hi = float(v.min()), float(v.max())
    if hi <= lo:
        return np.full(v.shape, 0.5)
    return (v - lo) / (hi - lo)


def dyadic_visibility_stock(model):
    """Raw visibility stock per agent for the dyadic score (Q4 ruling:
    prestige_visibility by default; 'entry' routes via entry_prestige_stock)."""
    attr = (
        model.entry_prestige_stock
        if model.dyadic_visibility_stock == "entry"
        else model.dyadic_visibility_stock
    )
    return np.array([getattr(a, attr) for a in model.agents], dtype=np.float64)


def compute_dyadic_scores(model):
    """U_ij = (1-lam)*V~_j + lam*sqrt(C~_ij * M~_j).

    V~ (visibility stock) and M~ (current_local_merit, Q3) are min-max
    normalised over the current population; C~ is 1 - d_torus(i,j)/size
    (grid mode) or the cosine overlap of the agents' flow-weighted harvest
    profiles (profile_overlap mode, Q2). Returns (U, ids) with U[i, j] the
    utility of candidate j for proposer i; the diagonal is 0.
    """
    agents = list(model.agents)
    n = len(agents)
    ids = [a.unique_id for a in agents]
    U = np.zeros((n, n), dtype=np.float64)
    if n < 2:
        return U, ids
    vis = _normalize_minmax(dyadic_visibility_stock(model))
    merit = _normalize_minmax([a.current_local_merit for a in agents])
    lam = float(model.dyadic_lambda)
    if model.closeness_mode == "profile_overlap":
        prof = np.stack(
            [np.asarray(a.harvest_profile, dtype=np.float64) for a in agents]
        )
        norms = np.linalg.norm(prof, axis=1)
        C = np.zeros((n, n), dtype=np.float64)
        nz = norms > 0.0
        if nz.any():
            C[np.ix_(nz, nz)] = (prof[nz] @ prof[nz].T) / np.outer(
                norms[nz], norms[nz]
            )
    else:  # grid: Manhattan torus distance, vectorised (identical arithmetic
        # to Scientist.compute_distance: abs, min against the wrap, sum)
        P = np.array([a.pos for a in agents], dtype=np.float64)
        d = np.abs(P[:, None, :] - P[None, :, :])
        d = np.minimum(d, model.size - d)
        C = 1.0 - d.sum(axis=2) / model.size
    np.fill_diagonal(C, 0.0)
    goal = np.sqrt(np.maximum(C * merit[None, :], 0.0))
    U = (1.0 - lam) * vis[None, :] + lam * goal
    np.fill_diagonal(U, 0.0)
    return U, ids


def _dyadic_bars(model, U, ids, index):
    """bar(k) = the score of k's best current link if linked, else the mean
    candidate score (the knob-free relative solo bar)."""
    n = len(ids)
    best = np.full(n, -np.inf)
    has_link = np.zeros(n, dtype=bool)
    for pair in model.links:
        a_uid, b_uid = tuple(pair)
        if a_uid in index and b_uid in index:
            ia, ib = index[a_uid], index[b_uid]
            if U[ia, ib] > best[ia]:
                best[ia] = U[ia, ib]
            if U[ib, ia] > best[ib]:
                best[ib] = U[ib, ia]
            has_link[ia] = True
            has_link[ib] = True
    mean_cand = U.sum(axis=1) / max(n - 1, 1)
    return np.where(has_link, best, mean_cand)


def dyadic_step(model):
    """One dyadic round (pseudocode: AdmiralsLog/archive/path_b_dyadic_sketch_v1.md).

    Proposers (per the regime) score all candidates, form or upgrade links
    under bilateral relative-bar clearance, then strictly weaker ties
    dissolve. Mutates model.links only; consumes model.rng (proposal order,
    logit sampling). Both formulations share this link update.
    """
    agents = list(model.agents)
    if len(agents) < 2:
        return
    U, ids = compute_dyadic_scores(model)
    index = {uid: i for i, uid in enumerate(ids)}
    n = len(ids)

    # --- formation: proposal + bilateral bar clearance
    if model.dyadic_proposal_regime == "newcomer":
        proposers = [a for a in agents if getattr(a, "is_newcomer", False)]
    else:  # universal
        proposers = list(agents)
    order = list(proposers)
    model.rng.shuffle(order)
    for prop in order:
        i = index[prop.unique_id]
        bars = _dyadic_bars(model, U, ids, index)  # sees links formed so far
        if model.dyadic_mode == "logit":
            scores = U[i, :].copy()
            scores[i] = -np.inf  # never propose to oneself
            scores = scores - scores.max()
            w = np.exp(scores / model.epsilon)
            w = w / w.sum()
            j = int(model.rng.choice(n, p=w))
        else:  # threshold: argmax, ties break on candidate order
            j = int(np.argmax(U[i, :]))
        if j == i:
            continue
        if U[i, j] >= bars[i] and U[j, i] >= bars[j]:
            model.links.add(frozenset((ids[i], ids[j])))

    # --- dissolution: strict on either side; ties persist
    bars = _dyadic_bars(model, U, ids, index)
    for pair in list(model.links):
        a_uid, b_uid = tuple(pair)
        ia, ib = index[a_uid], index[b_uid]
        if U[ia, ib] < bars[ia] or U[ib, ia] < bars[ib]:
            model.links.discard(pair)


def compute_teams(model):
    """Co-harvest (production) teams: agents sharing a tile at the end of
    the step. Records model.teams (tuples of unique_ids, size >= 2) and the
    four dyadic statistics the datacollector reporters read. Consumes no
    rng and runs in every dyadic_mode, so the off-mode structural baseline
    (co-location without any choice layer) is measured too."""
    by_pos = {}
    for a in model.agents:
        by_pos.setdefault(a.pos, []).append(a.unique_id)
    model.teams = [
        tuple(sorted(members)) for members in by_pos.values() if len(members) >= 2
    ]
    active = set()
    for members in model.teams:
        for x_i in range(len(members)):
            for y_i in range(x_i + 1, len(members)):
                active.add(frozenset((members[x_i], members[y_i])))
    repeated = len(active & model._ever_active_pairs)
    model._ever_active_pairs |= active
    n_active = len(active)
    share = (repeated / n_active) if n_active else 0.0
    mean_team = float(np.mean([len(t) for t in model.teams])) if model.teams else 0.0
    # prestige assortativity over active co-harvest pairs (Pearson, guarded
    # like correlation_reporter: degenerate variance reads as 0)
    by_id = {a.unique_id: a for a in model.agents}
    xs, ys = [], []
    for pair in active:
        p, q = tuple(pair)
        xs.append(by_id[p].prestige)
        ys.append(by_id[q].prestige)
    if len(xs) >= 2 and np.std(xs) > 0 and np.std(ys) > 0:
        assort = float(np.corrcoef(xs, ys)[0, 1])
    else:
        assort = 0.0
    # choice-layer link statistics (the dyadic decision itself): repetition
    # of declared ties and prestige assortativity over the link set
    n_links = len(model.links)
    link_repeated = len(model.links & model._ever_active_links)
    model._ever_active_links |= model.links
    link_rep_share = (link_repeated / n_links) if n_links else 0.0
    lx, ly = [], []
    for pair in model.links:
        p, q = tuple(pair)
        lx.append(by_id[p].prestige)
        ly.append(by_id[q].prestige)
    if len(lx) >= 2 and np.std(lx) > 0 and np.std(ly) > 0:
        link_assort = float(np.corrcoef(lx, ly)[0, 1])
    else:
        link_assort = 0.0
    model._dyadic_stats = {
        "mean_team_size": np.float64(mean_team),
        "repeated_collab_share": np.float64(share),
        "dyadic_assortativity_prestige": np.float64(assort),
        "n_active_links": n_links,
        "link_repeated_share": np.float64(link_rep_share),
        "link_assortativity_prestige": np.float64(link_assort),
    }


def correlation_reporter(merit_type, prestige_type):
    def reporter(m):
        # BUG-0006: exclude unscored step-t newborns (non-finite ideal_Merit)
        Prest, Mert = _finite_pair_vectors(m, prestige_type, merit_type)
        if len(Prest) >= 2 and np.std(Prest) > 0 and np.std(Mert) > 0:
            return np.corrcoef(Prest, Mert)[0, 1]
        else:
            return 0
    return reporter


def update_step_goals(model) -> None:
    sentinel = model._default_steps_thresholds
    steps = model.steps

    milestones = (
        ("explored_50_step", "explored_percentage", 0.5),
        ("explored_90_step", "explored_percentage", 0.9),
        ("weighted_50_step", "explored_weighted_by_initial_knowledge", 0.5),
        ("weighted_90_step", "explored_weighted_by_initial_knowledge", 0.9),
        ("harvested_10_step", "percentage_knowledge_harvested", 0.10),
        ("harvested_25_step", "percentage_knowledge_harvested", 0.25),
    )

    for step_attr, metric_attr, threshold in milestones:
        if (
            getattr(model, step_attr) == sentinel
            and getattr(model, metric_attr) >= threshold
        ):
            setattr(model, step_attr, steps)


def _finite_pair_vectors(model, prestige_type, merit_type):
    """Extract matched finite (prestige, merit) vectors for correlation_reporter.

    Excludes agents whose merit is non-finite (e.g. newborns spawned on the
    current step after ``update_ideal_merit`` already ran — BUG-0006).
    """
    prest, mert = [], []
    for a in model.agents:
        p = float(getattr(a, prestige_type))
        mt = float(getattr(a, merit_type))
        if np.isfinite(p) and np.isfinite(mt):
            prest.append(p)
            mert.append(mt)
    return prest, mert
