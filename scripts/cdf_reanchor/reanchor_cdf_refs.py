"""Verify the `file:line` anchors of docs/conceptual_degrees_of_freedom.md.

Model-internal documentation tooling: no data, no calibration. Run from the
repo root:

    /opt/hermes/.venv/bin/python scripts/cdf_reanchor/reanchor_cdf_refs.py

For every catalogue row it (a) checks that the Reference cell in the document
matches the audited mapping, (b) opens the live file and asserts the quoted
token is on the cited line, and (c) prints the old -> new table with a status.
Exit status is non-zero if any anchor has drifted, so a later edit that moves
code fails loudly instead of leaving the catalogue silently wrong.

Status vocabulary
-----------------
re-anchored   the module that owned the code still owns it (the old file was
              renamed or moved only: mesa_model.py -> sciencemodels/model.py,
              landscapefunctions.py -> sciencemodels/landscapes.py,
              runner.py -> scripts/runner.py, bulk_runner.py ->
              scripts/bulk_runner.py); only the line numbers changed.
moved-module  the restructure split the concern out into a different module
              (agents.py, metrics.py, io.py, viz.py, plotting.py).
unresolvable  the referenced code/comment no longer exists anywhere; the
              document says so in place with the reason.
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DOC = os.path.join(ROOT, "docs", "conceptual_degrees_of_freedom.md")

# (old reference cell as published in v2, new reference cell as published in v3,
#  status, [(file, 1-based line, substring that must be on that line)])
MAPPING = [
    ('mesa_model.py:305,328', '`sciencemodels/model.py:121,147`', 're-anchored',
     [('sciencemodels/model.py', 121, 'grid_size: int'), ('sciencemodels/model.py', 147, 'self.size = grid_size')]),
    ('mesa_model.py:231', '`sciencemodels/model.py:46`', 're-anchored',
     [('sciencemodels/model.py', 46, 'MultiGrid(self.size, self.size, torus=True)')]),
    ('runner.py:42`, `bulk_runner.py:51', '`scripts/runner.py:51`, `scripts/bulk_runner.py:62`', 're-anchored',
     [('scripts/runner.py', 51, 'grid_size'), ('scripts/bulk_runner.py', 62, 'grid_size')]),
    ('mesa_model.py:228-249', '`sciencemodels/model.py:47-64`', 're-anchored',
     [('sciencemodels/model.py', 47, 'add_property_layer'), ('sciencemodels/model.py', 63, '"explored"')]),
    ('landscapefunctions.py:13-15', '`sciencemodels/landscapes.py:13-15`', 're-anchored',
     [('sciencemodels/landscapes.py', 13, 'def randominit')]),
    ('landscapefunctions.py:18-29', '`sciencemodels/landscapes.py:18-29`', 're-anchored',
     [('sciencemodels/landscapes.py', 18, 'def Gaussian'), ('sciencemodels/landscapes.py', 27, 'vonmises.pdf')]),
    ('landscapefunctions.py:23', '`sciencemodels/landscapes.py:23`', 're-anchored',
     [('sciencemodels/landscapes.py', 23, 'rng_fix = np.random.default_rng(args["seed"])')]),
    ('landscapefunctions.py:19-22', '`sciencemodels/landscapes.py:19-22`', 're-anchored',
     [('sciencemodels/landscapes.py', 22, 'kappa = 1 / (args["std"] / size)')]),
    ('landscapefunctions.py:41-53', '`sciencemodels/landscapes.py:41-53`', 're-anchored',
     [('sciencemodels/landscapes.py', 41, 'def multipleGaussians'), ('sciencemodels/landscapes.py', 52, 'tot += Gaussian')]),
    ('runner.py:10,21,24', '`scripts/runner.py:19-21,31,34`', 're-anchored',
     [('scripts/runner.py', 19, 'chosen_setup'), ('scripts/runner.py', 31, '"std": 2, "number_gaussians": 2'), ('scripts/runner.py', 34, '"std": 5, "number_gaussians": 4')]),
    ('bulk_runner.py:22-26', '`scripts/bulk_runner.py:21-23,34,37`', 're-anchored',
     [('scripts/bulk_runner.py', 21, 'chosen_setup'), ('scripts/bulk_runner.py', 34, '"std": 5, "number_gaussians": 2'), ('scripts/bulk_runner.py', 37, '"std": 5, "number_gaussians": 6')]),
    ('landscapefunctions.py:32-38', '`sciencemodels/landscapes.py:32-38`', 're-anchored',
     [('sciencemodels/landscapes.py', 34, 'prop_random = args["prop_random"]'), ('sciencemodels/landscapes.py', 38, 'a + prop_random * b')]),
    ('runner.py:28`, `bulk_runner.py:30', '`scripts/runner.py:38`, `scripts/bulk_runner.py:41`', 're-anchored',
     [('scripts/runner.py', 38, '"prop_random": 5'), ('scripts/bulk_runner.py', 41, '"prop_random": 5')]),
    ('runner.py:30`, `bulk_runner.py:32', '`scripts/runner.py:40`, `scripts/bulk_runner.py:43`', 're-anchored',
     [('scripts/runner.py', 40, '"std_curiosity": 0.1'), ('scripts/bulk_runner.py', 43, '"std_curiosity": 0.1')]),
    ('landscapefunctions.py:56-64 (comment block)', '**unresolvable** (block deleted)', 'unresolvable',
     []),
    ('mesa_model.py:265-272', '`sciencemodels/model.py:80-87`', 're-anchored',
     [('sciencemodels/model.py', 80, 'if self.knowledge_normalization == "share"'), ('sciencemodels/model.py', 87, '* 0.95 + 0.05')]),
    ('mesa_model.py:270-272', '`sciencemodels/model.py:85-87`', 're-anchored',
     [('sciencemodels/model.py', 87, '* 0.95 + 0.05')]),
    ('mesa_model.py:266-268', '`sciencemodels/model.py:80-83`', 're-anchored',
     [('sciencemodels/model.py', 82, 'val / self.total_initial_knowledge')]),
    ('mesa_model.py:327,265,533', '`sciencemodels/model.py:146,161,343`', 're-anchored',
     [('sciencemodels/model.py', 146, 'self.new_questions = new_questions'), ('sciencemodels/model.py', 161, 'self.knowledge_normalization = knowledge_normalization'), ('sciencemodels/model.py', 343, 'if self.new_questions:')]),
    ('mesa_model.py:281', '`sciencemodels/model.py:96`', 're-anchored',
     [('sciencemodels/model.py', 96, 'self.total_initial_knowledge = np.float64(1.0)')]),
    ('mesa_model.py:274-279', '`sciencemodels/model.py:89-94`', 're-anchored',
     [('sciencemodels/model.py', 89, '"initial_knowledge"'), ('sciencemodels/model.py', 92, '"previous_knowledge"')]),
    ('mesa_model.py:503-507', '`sciencemodels/model.py:303-307`', 're-anchored',
     [('sciencemodels/model.py', 303, 'def farm'), ('sciencemodels/model.py', 306, '* (1 - self.harvest)')]),
    ('runner.py:41`, `bulk_runner.py:50', '`scripts/runner.py:50`, `scripts/bulk_runner.py:61`', 're-anchored',
     [('scripts/runner.py', 50, '"harvest":0.02'), ('scripts/bulk_runner.py', 61, '"harvest": 0.02')]),
    ('mesa_model.py:507', '`sciencemodels/model.py:303-307,342-349`', 're-anchored',
     [('sciencemodels/model.py', 307, 'self.update_knowledge()'), ('sciencemodels/model.py', 342, 'def _generate_new_questions')]),
    ('mesa_model.py:441-444', '`sciencemodels/model.py:283-286`', 're-anchored',
     [('sciencemodels/model.py', 283, 'def update_knowledge'), ('sciencemodels/model.py', 285, 'last_tile_knowledge')]),
    ('mesa_model.py:532-539', '`sciencemodels/model.py:342-349`', 're-anchored',
     [('sciencemodels/model.py', 346, '< 0.025')]),
    ('mesa_model.py:534', '`sciencemodels/model.py:344`', 're-anchored',
     [('sciencemodels/model.py', 344, 'occupied_pos = list(set(')]),
    ('mesa_model.py:59', '`sciencemodels/agents.py:41`', 'moved-module',
     [('sciencemodels/agents.py', 41, 'self.age = 23')]),
    ('mesa_model.py:290', '`sciencemodels/model.py:105`', 're-anchored',
     [('sciencemodels/model.py', 105, 'rng.choice(np.arange(23, 50))')]),
    ('mesa_model.py:206', '`sciencemodels/agents.py:175`', 'moved-module',
     [('sciencemodels/agents.py', 175, 'self.age += 1')]),
    ('mesa_model.py:45,206 (whole class)', '`sciencemodels/agents.py:41,175` (class 21-183)', 'moved-module',
     [('sciencemodels/agents.py', 41, 'self.age = 23'), ('sciencemodels/agents.py', 175, 'self.age += 1')]),
    ('mesa_model.py:289', '`sciencemodels/model.py:104`', 're-anchored',
     [('sciencemodels/model.py', 104, 'self.rng.integers(0, self.size)')]),
    ('mesa_model.py:288', '`sciencemodels/model.py:103`', 're-anchored',
     [('sciencemodels/model.py', 103, 'Scientist(self, w, self.epsilon)')]),
    ('mesa_model.py:56-58', '`sciencemodels/agents.py:38-40`', 'moved-module',
     [('sciencemodels/agents.py', 38, 'self.prestige = np.float64(0.0)'), ('sciencemodels/agents.py', 40, 'self.prestige_vanishing = np.float64(0.0)')]),
    ('mesa_model.py:313-315', '`sciencemodels/model.py:129-131`', 're-anchored',
     [('sciencemodels/model.py', 131, 'lambda cur, rng, params: cur')]),
    ('landscapefunctions.py:57-64', '`sciencemodels/landscapes.py:57-64`', 're-anchored',
     [('sciencemodels/landscapes.py', 57, 'def beta'), ('sciencemodels/landscapes.py', 59, 'params.get("std_curiosity", 0.1)')]),
    ('landscapefunctions.py:67-68', '`sciencemodels/landscapes.py:67-68`', 're-anchored',
     [('sciencemodels/landscapes.py', 67, 'def uniform'), ('sciencemodels/landscapes.py', 68, 'rng.uniform(0.0, 1.0)')]),
    ('bulk_runner.py:38-43', '`scripts/bulk_runner.py:49-54`', 're-anchored',
     [('scripts/bulk_runner.py', 50, '[k / 10 for k in range(11)]'), ('scripts/bulk_runner.py', 54, 'landscapes.uniform')]),
    ('runner.py:37', '`scripts/runner.py:46`', 're-anchored',
     [('scripts/runner.py', 46, '# minimum 3')]),
    ('bulk_runner.py:46', '`scripts/bulk_runner.py:57`', 're-anchored',
     [('scripts/bulk_runner.py', 57, '"n_agents": 20')]),
    ('mesa_model.py:174-176', '`sciencemodels/agents.py:142-144`', 'moved-module',
     [('sciencemodels/agents.py', 142, 'get_neighborhood'), ('sciencemodels/agents.py', 143, 'moore=False, include_center=False')]),
    ('mesa_model.py:182-198', '`sciencemodels/agents.py:150-170`', 'moved-module',
     [('sciencemodels/agents.py', 150, 'best_reward = total_current_reward'), ('sciencemodels/agents.py', 166, 'if best_reward - total_current_reward > 0:')]),
    ('mesa_model.py:188-190', '`sciencemodels/agents.py:156-159`', 'moved-module',
     [('sciencemodels/agents.py', 156, 'noise = self.rng.uniform(-1.0, 1.0)'), ('sciencemodels/agents.py', 158, '* (1 + self.epsilon * noise)')]),
    ('runner.py:40`, `bulk_runner.py:49', '`scripts/runner.py:49`, `scripts/bulk_runner.py:60`', 're-anchored',
     [('scripts/runner.py', 49, '"epsilon": 0.1'), ('scripts/bulk_runner.py', 60, '"epsilon": 0.1')]),
    ('mesa_model.py:178-181', '`sciencemodels/agents.py:146-149`', 'moved-module',
     [('sciencemodels/agents.py', 146, 'current_rw_novelty = self.model.compute_knowledge(self.pos)'), ('sciencemodels/agents.py', 147, 'total_current_reward, currentVis')]),
    ('mesa_model.py:177', '`sciencemodels/agents.py:145`', 'moved-module',
     [('sciencemodels/agents.py', 145, '# optionpos = self.model.rng.choice(neighbors_nodes)')]),
    ('mesa_model.py:152', '`sciencemodels/agents.py:134`', 'moved-module',
     [('sciencemodels/agents.py', 134, '#    reward = self.curiosity * Novelty/')]),
    ('mesa_model.py:147-150', '`sciencemodels/agents.py:129-132`', 'moved-module',
     [('sciencemodels/agents.py', 130, '(1) the above'), ('sciencemodels/agents.py', 132, 'weighted by prestige')]),
    ('mesa_model.py:599-601', '`sciencemodels/model.py:390-394`', 're-anchored',
     [('sciencemodels/model.py', 394, 'temp.shuffle_do("step")')]),
    ('mesa_model.py:107-108', '`sciencemodels/agents.py:89-90`', 'moved-module',
     [('sciencemodels/agents.py', 89, 'This part is just a test')]),
    ('mesa_model.py:143-145', '`sciencemodels/agents.py:125-127`', 'moved-module',
     [('sciencemodels/agents.py', 125, 'reward = (novelty / self.model.avg_current_agent_knowledge) ** ('), ('sciencemodels/agents.py', 127, '* visibility_used ** (1 - self.curiosity)')]),
    ('mesa_model.py:111,143', '`sciencemodels/agents.py:93,125`', 'moved-module',
     [('sciencemodels/agents.py', 93, 'self.curiosity * novelty / self.model.avg_current_agent_knowledge'), ('sciencemodels/agents.py', 125, 'avg_current_agent_knowledge')]),
    ('mesa_model.py:109-113', '`sciencemodels/agents.py:91-95`', 'moved-module',
     [('sciencemodels/agents.py', 91, 'if not self.model.use_distance:')]),
    ('mesa_model.py:101-102', '`sciencemodels/agents.py:83-84`', 'moved-module',
     [('sciencemodels/agents.py', 84, '+ 1.0) / (avg_agent_distance + 1.0)')]),
    ('mesa_model.py:576-585', '`sciencemodels/model.py:369-378`', 're-anchored',
     [('sciencemodels/model.py', 369, 'self.agent_avg_distance = np.mean(')]),
    ('mesa_model.py:66-80', '`sciencemodels/agents.py:48-62`', 'moved-module',
     [('sciencemodels/agents.py', 48, 'def compute_distance'), ('sciencemodels/agents.py', 57, 'dx = min(dx, n - dx)')]),
    ('mesa_model.py:136-139,317', '`sciencemodels/agents.py:117-121`, `sciencemodels/model.py:154`', 'moved-module',
     [('sciencemodels/agents.py', 118, 'if self.model.use_visibility_reward:'), ('sciencemodels/model.py', 154, 'self.use_visibility_reward = use_visibility_reward')]),
    ('mesa_model.py:121-133', '`sciencemodels/agents.py:103-115`', 'moved-module',
     [('sciencemodels/agents.py', 104, 'a.prestige_vanishing'), ('sciencemodels/agents.py', 111, 'if prest.sum() > 0.0:'), ('sciencemodels/agents.py', 112, 'np.average(')]),
    ('mesa_model.py:91|prestige_visibility', '`sciencemodels/agents.py:73`', 'moved-module',
     [('sciencemodels/agents.py', 73, 'we could also use individual distances')]),
    ('mesa_model.py:141-142', '`sciencemodels/agents.py:123-124`', 'moved-module',
     [('sciencemodels/agents.py', 123, 'visibility of a field vs. visibility of an agent?')]),
    ('mesa_model.py:27-36', '**unresolvable** (helper deleted)', 'unresolvable',
     []),
    ('mesa_model.py:355-363', '`sciencemodels/model.py:195-203`', 're-anchored',
     [('sciencemodels/model.py', 195, 'if not ('), ('sciencemodels/model.py', 202, 'Agents will NOT use distance information')]),
    ('mesa_model.py:204,207-210,82-83', '`sciencemodels/agents.py:172-183`', 'moved-module',
     [('sciencemodels/agents.py', 173, 'self.model.farm(self.pos)'), ('sciencemodels/agents.py', 181, 'increase_prestige(knowledge_before * self.model.harvest)')]),
    ('mesa_model.py:84-90', '`sciencemodels/agents.py:66-72`', 'moved-module',
     [('sciencemodels/agents.py', 66, 'self.prestige_visibility += np.mean('), ('sciencemodels/agents.py', 68, 'value ** (a.curiosity) * self.visibility')]),
    ('mesa_model.py:91|prestige_vanishing', '`sciencemodels/agents.py:73`', 'moved-module',
     [('sciencemodels/agents.py', 73, 'individual distances')]),
    ('mesa_model.py:92-99,316', '`sciencemodels/agents.py:74-81`, `sciencemodels/model.py:132`', 'moved-module',
     [('sciencemodels/agents.py', 74, 'x = self.model.vanishing_factor_for_prestige'), ('sciencemodels/model.py', 132, 'vanishing_factor_for_prestige: np.float64 = np.float64(0.9)')]),
    ('runner.py:50', '`scripts/runner.py:59`', 're-anchored',
     [('scripts/runner.py', 59, '"vanishing_factor_for_prestige": 0.9')]),
    ('bulk_runner.py:59', '`scripts/bulk_runner.py:70`', 're-anchored',
     [('scripts/bulk_runner.py', 70, '"vanishing_factor_for_prestige": 0.9')]),
    ('mesa_model.py:453-497', '`sciencemodels/metrics.py:53-97`', 'moved-module',
     [('sciencemodels/metrics.py', 53, 'def update_ideal_merit')]),
    ('mesa_model.py:469,476-477', '`sciencemodels/metrics.py:69,78-79`', 'moved-module',
     [('sciencemodels/metrics.py', 69, '((1- model.harvest) ** redone)'), ('sciencemodels/metrics.py', 79, 'temp_knowledge_map[pos] = temp_knowledge_map[pos] * (1- model.harvest)')]),
    ('mesa_model.py:455,497', '`sciencemodels/metrics.py:55,107`', 'moved-module',
     [('sciencemodels/metrics.py', 55, 'model.grid.properties["previous_knowledge"].data'), ('sciencemodels/metrics.py', 107, '.data.copy()')]),
    ('mesa_model.py:509-513', '`sciencemodels/model.py:309-313`', 're-anchored',
     [('sciencemodels/model.py', 309, 'def __generate_new_agents'), ('sciencemodels/model.py', 312, 'self.steps * self.agent_generation_time')]),
    ('runner.py:45`, `bulk_runner.py:54', '`scripts/runner.py:54`, `scripts/bulk_runner.py:65`', 're-anchored',
     [('scripts/runner.py', 54, '"agent_generation_time": 0'), ('scripts/bulk_runner.py', 65, '"agent_generation_time": 0')]),
    ('mesa_model.py:514-523', '`sciencemodels/model.py:315-326`', 're-anchored',
     [('sciencemodels/model.py', 315, 'prestige_agent_tuples'), ('sciencemodels/model.py', 323, 'if val <= p:')]),
    ('mesa_model.py:515-523', '`sciencemodels/model.py:321-326`', 're-anchored',
     [('sciencemodels/model.py', 321, 'self.rng.uniform(0, totalPrestige)')]),
    ('mesa_model.py:525', '`sciencemodels/model.py:328-331`', 're-anchored',
     [('sciencemodels/model.py', 328, 'w = self.agent_generation_function(')]),
    ('mesa_model.py:526-527', '`sciencemodels/model.py:336-337`', 're-anchored',
     [('sciencemodels/model.py', 336, 'coords = sup.pos')]),
    ('mesa_model.py:59,290', '`sciencemodels/agents.py:41`, `sciencemodels/model.py:105`', 'moved-module',
     [('sciencemodels/agents.py', 41, 'self.age = 23'), ('sciencemodels/model.py', 105, 'rng.choice(np.arange(23, 50))')]),
    ('mesa_model.py:288,525', '`sciencemodels/model.py:103,331`', 're-anchored',
     [('sciencemodels/model.py', 103, 'self.epsilon'), ('sciencemodels/model.py', 331, 'Scientist(self, w, self.epsilon)')]),
    ('runner.py:45|mislabel', '`scripts/runner.py:54`', 're-anchored',
     [('scripts/runner.py', 54, 'if 0.1, 10 new agent per generation')]),
    ('mesa_model.py:155-170', '`sciencemodels/metrics.py:35-50`', 'moved-module',
     [('sciencemodels/metrics.py', 35, 'def local_merit'), ('sciencemodels/metrics.py', 41, 'positions = [pos, *neighbors]')]),
    ('mesa_model.py:453-497|ideal', '`sciencemodels/metrics.py:53-97`', 'moved-module',
     [('sciencemodels/metrics.py', 53, 'def update_ideal_merit')]),
    ('mesa_model.py:367-375', '`sciencemodels/metrics.py:110-118`', 'moved-module',
     [('sciencemodels/metrics.py', 110, 'def correlation_reporter'), ('sciencemodels/metrics.py', 115, 'np.corrcoef')]),
    ('mesa_model.py:423-428', '`sciencemodels/model.py:261-266`', 're-anchored',
     [('sciencemodels/model.py', 261, 'Corr_prestige_local_merit'), ('sciencemodels/model.py', 266, 'Corr_prestige_vanishing_ideal_merit')]),
    ('mesa_model.py:545-552', '`sciencemodels/metrics.py:125-132`', 'moved-module',
     [('sciencemodels/metrics.py', 130, '"harvested_10_step"'), ('sciencemodels/metrics.py', 131, '"harvested_25_step"')]),
    ('mesa_model.py:344', '`sciencemodels/model.py:184`', 're-anchored',
     [('sciencemodels/model.py', 184, '_default_steps_thresholds = step_limit * (1.25)')]),
    ('mesa_model.py:312', '`sciencemodels/model.py:128`', 're-anchored',
     [('sciencemodels/model.py', 128, 'step_limit: int = 400')]),
    ('runner.py:48', '`scripts/runner.py:57`', 're-anchored',
     [('scripts/runner.py', 57, '"step_limit": 200')]),
    ('bulk_runner.py:57', '`scripts/bulk_runner.py:68`', 're-anchored',
     [('scripts/bulk_runner.py', 68, '"step_limit": 400')]),
    ('mesa_model.py:387-412', '`sciencemodels/model.py:225-250`', 're-anchored',
     [('sciencemodels/model.py', 225, 'top1pct_prestige_share'), ('sciencemodels/model.py', 238, 'top10pct_prestige_share')]),
    ('mesa_model.py:566-575', '`sciencemodels/model.py:359-368`', 're-anchored',
     [('sciencemodels/model.py', 359, 'self.explored_percentage = np.sum('), ('sciencemodels/model.py', 362, 'explored_weighted_by_initial_knowledge')]),
    ('mesa_model.py:280,451,291', '`sciencemodels/model.py:107,293`', 're-anchored',
     [('sciencemodels/model.py', 107, 'self.grid.properties["explored"].data[coords] = True'), ('sciencemodels/model.py', 293, 'self.grid.properties["explored"].data[coords] = True')]),
    ('mesa_model.py:586-589', '`sciencemodels/model.py:379-382`', 're-anchored',
     [('sciencemodels/model.py', 379, 'self.percentage_knowledge_harvested = (')]),
    ('mesa_model.py:645-664', '`sciencemodels/io.py:25-44`', 'moved-module',
     [('sciencemodels/io.py', 25, 'def build_report'), ('sciencemodels/io.py', 29, 'if k[0].islower()'), ('sciencemodels/io.py', 40, '"Mean_of_"+k')]),
    ('mesa_model.py:856-864', '`sciencemodels/model.py:506-514`', 're-anchored',
     [('sciencemodels/model.py', 506, 'if needs_header or intention'), ('sciencemodels/model.py', 510, 'all_params["step_limit"] = 1')]),
    ('mesa_model.py:881-886', '`sciencemodels/model.py:530-535`, `scripts/bulk_runner.py:82-83`', 're-anchored',
     [('sciencemodels/model.py', 531, 'all_params["agent_seed"] = repeat'), ('scripts/bulk_runner.py', 82, 'repeats_per_setting=40'), ('scripts/bulk_runner.py', 83, 'change_landscape_seed=True')]),
    ('bulk_runner.py:39', '`scripts/bulk_runner.py:50`', 're-anchored',
     [('scripts/bulk_runner.py', 50, '[k / 10 for k in range(11)]')]),
    ('bulk_runner.py:87', '`scripts/bulk_runner.py:97`', 're-anchored',
     [('scripts/bulk_runner.py', 97, 'Parallel(n_jobs=10')]),
    ('mesa_model.py:627-630,670-675', '`sciencemodels/viz.py:48-53,60-67`', 'moved-module',
     [('sciencemodels/viz.py', 49, '(k.unique_id / model.number_agents - 0.5) ** 2'), ('sciencemodels/viz.py', 61, '(k.unique_id / model.number_agents - 0.5) ** 2')]),
    ('mesa_model.py:693-696', '`sciencemodels/viz.py:84-87,46`', 'moved-module',
     [('sciencemodels/viz.py', 85, 'vmin=0, vmax=1'), ('sciencemodels/viz.py', 46, '# , vmin=0, vmax=1')]),
    ('compare_landscapes_corr-prestige-merit.ipynb', '`notebooks/compare_landscapes_corr-prestige-merit.ipynb` (cell 0)', 're-anchored',
     [('notebooks/compare_landscapes_corr-prestige-merit.ipynb', 28, 'run_number0')]),
    ('compare_landscapes_flexible search.ipynb`, `compare_landscapes_normalizedknowledge.ipynb', '`notebooks/compare_landscapes_flexible_search.ipynb`, `notebooks/compare_landscapes_normalizedknowledge.ipynb` (cell 0)', 're-anchored',
     [('notebooks/compare_landscapes_flexible_search.ipynb', 11, 'sciencemodels'), ('notebooks/compare_landscapes_normalizedknowledge.ipynb', 11, 'sciencemodels')]),
    ('longitudinalcomparisions.ipynb (cell 1)', '`sciencemodels/plotting.py:172,204`', 'moved-module',
     [('sciencemodels/plotting.py', 172, 'length=None'), ('sciencemodels/plotting.py', 204, 'def plot_all_curiosities')]),
    ('mesa_model.py:605-618', '`sciencemodels/viz.py:26-39,42-55`', 'moved-module',
     [('sciencemodels/viz.py', 26, 'def Vizualize_3d'), ('sciencemodels/viz.py', 42, 'def plot(model)')]),
]

# Anchors cited in the prose bullets and in the role column, rather than in a
# Reference cell. Checked separately so a moving target fails the same way.
INLINE_ANCHORS = [
    ('sciencemodels/model.py', 346, '0.025'),
    ('sciencemodels/model.py', 315, 'prestige_agent_tuples'),
    ('sciencemodels/model.py', 336, 'coords = sup.pos'),
    ('sciencemodels/agents.py', 166, 'if best_reward - total_current_reward > 0:'),
    ('sciencemodels/agents.py', 146, 'current_rw_novelty = self.model.compute_knowledge(self.pos)'),
    ('sciencemodels/metrics.py', 130, '"harvested_10_step"'),
    ('sciencemodels/metrics.py', 131, '"harvested_25_step"'),
    ('sciencemodels/plotting.py', 37, 'LEGACY_COLUMN_RENAMES = {'),
]

_CACHE: dict[str, list[str]] = {}


def _lines(relpath: str) -> list[str]:
    if relpath not in _CACHE:
        with open(os.path.join(ROOT, relpath), encoding="utf-8", errors="replace") as f:
            _CACHE[relpath] = f.read().split("\n")
    return _CACHE[relpath]


def _doc_cells() -> list[str]:
    out = []
    started = False
    with open(DOC, encoding="utf-8") as f:
        for line in f:
            if line.startswith("# Purpose"):
                started = True
                continue
            if not started:
                continue
            if line.startswith("| `") or line.startswith("| **unresolvable"):
                out.append(line.strip().strip("|").split("|")[0].strip())
    return out


def main() -> int:
    cells = _doc_cells()
    if len(cells) != len(MAPPING):
        print(f"FAIL: document has {len(cells)} catalogue rows, mapping has {len(MAPPING)}")
        return 1

    failures = 0
    counts = {"re-anchored": 0, "moved-module": 0, "unresolvable": 0}
    print(f"{'#':>3}  {'old reference':<46} {'new reference':<96} status")
    print("-" * 175)
    for i, ((old, new, status, checks), cell) in enumerate(zip(MAPPING, cells), 1):
        counts[status] += 1
        if cell != new:
            failures += 1
            print(f"{i:>3}  {old:<46} {new:<96} ROW MISMATCH (doc has {cell})")
            continue
        for relpath, lineno, token in checks:
            lines = _lines(relpath)
            # lineno 0 means "anywhere in the file" (notebooks are single-line JSON)
            got = "" if lineno == 0 else (lines[lineno - 1] if 0 < lineno <= len(lines) else "")
            if token not in (got if lineno else "\n".join(lines)):
                failures += 1
                print(f"{i:>3}  {old:<46} {new:<96} ANCHOR FAIL {relpath}:{lineno}")
                print(f"     expected token: {token!r}")
                print(f"     line reads    : {got.strip()[:110]!r}")
                break
        else:
            print(f"{i:>3}  {old:<46} {new:<96} {status}")

    for relpath, lineno, token in INLINE_ANCHORS:
        lines = _lines(relpath)
        got = lines[lineno - 1] if 0 < lineno <= len(lines) else ""
        if token not in got:
            failures += 1
            print(f"INLINE ANCHOR FAIL {relpath}:{lineno} expected {token!r} got {got.strip()[:90]!r}")

    print("-" * 175)
    print(f"inline anchors: {len(INLINE_ANCHORS)}")
    print(f"rows: {len(MAPPING)}  re-anchored: {counts['re-anchored']}  "
          f"moved-module: {counts['moved-module']}  unresolvable: {counts['unresolvable']}")
    # one callout per referenced file: a two-file row counts twice, the two
    # unresolvable rows cite no file
    groups = sum(len(re.findall(r"[\w./ -]+\.(?:py|ipynb)", old)) for old, *_ in MAPPING)
    print(f"reference callouts: {groups} gross, {groups - counts['unresolvable']} resolvable")
    if failures:
        print(f"FAIL: {failures} anchor check(s) failed")
        return 1
    print("PASS: every catalogue anchor resolves to the token it quotes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
