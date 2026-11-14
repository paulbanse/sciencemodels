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
    ('mesa_model.py:305,328', '`sciencemodels/model.py:128,160`', 're-anchored',
     [('sciencemodels/model.py', 128, 'grid_size: int'),
             ('sciencemodels/model.py', 160, 'self.size = grid_size')]),
    ('mesa_model.py:231', '`sciencemodels/model.py:53`', 're-anchored',
     [('sciencemodels/model.py', 53, 'MultiGrid(self.size, self.size, torus=True)')]),
    ('runner.py:42`, `bulk_runner.py:51', '`scripts/runner.py:51`, `scripts/bulk_runner.py:62`', 're-anchored',
     [('scripts/runner.py', 51, 'grid_size'),
             ('scripts/bulk_runner.py', 62, 'grid_size')]),
    ('mesa_model.py:228-249', '`sciencemodels/model.py:54-71`', 're-anchored',
     [('sciencemodels/model.py', 54, 'add_property_layer'),
             ('sciencemodels/model.py', 70, '"explored"')]),
    ('landscapefunctions.py:13-15', '`sciencemodels/landscapes.py:13-15`', 're-anchored',
     [('sciencemodels/landscapes.py', 13, 'def randominit')]),
    ('landscapefunctions.py:18-29', '`sciencemodels/landscapes.py:18-29`', 're-anchored',
     [('sciencemodels/landscapes.py', 18, 'def Gaussian'),
             ('sciencemodels/landscapes.py', 27, 'vonmises.pdf')]),
    ('landscapefunctions.py:23', '`sciencemodels/landscapes.py:23`', 're-anchored',
     [('sciencemodels/landscapes.py', 23, 'rng_fix = np.random.default_rng(args["seed"])')]),
    ('landscapefunctions.py:19-22', '`sciencemodels/landscapes.py:19-22`', 're-anchored',
     [('sciencemodels/landscapes.py', 22, 'kappa = 1 / (args["std"] / size)')]),
    ('landscapefunctions.py:41-53', '`sciencemodels/landscapes.py:41-53`', 're-anchored',
     [('sciencemodels/landscapes.py', 41, 'def multipleGaussians'),
             ('sciencemodels/landscapes.py', 52, 'tot += Gaussian')]),
    ('runner.py:10,21,24', '`scripts/runner.py:19-21,31,34`', 're-anchored',
     [('scripts/runner.py', 19, 'chosen_setup'),
             ('scripts/runner.py', 31, '"std": 2, "number_gaussians": 2'),
             ('scripts/runner.py', 34, '"std": 5, "number_gaussians": 4')]),
    ('bulk_runner.py:22-26', '`scripts/bulk_runner.py:21-23,34,37`', 're-anchored',
     [('scripts/bulk_runner.py', 21, 'chosen_setup'),
             ('scripts/bulk_runner.py', 34, '"std": 5, "number_gaussians": 2'),
             ('scripts/bulk_runner.py', 37, '"std": 5, "number_gaussians": 6')]),
    ('landscapefunctions.py:32-38', '`sciencemodels/landscapes.py:32-38`', 're-anchored',
     [('sciencemodels/landscapes.py', 34, 'prop_random = args["prop_random"]'),
             ('sciencemodels/landscapes.py', 38, 'a + prop_random * b')]),
    ('runner.py:28`, `bulk_runner.py:30', '`scripts/runner.py:38`, `scripts/bulk_runner.py:41`', 're-anchored',
     [('scripts/runner.py', 38, '"prop_random": 5'),
             ('scripts/bulk_runner.py', 41, '"prop_random": 5')]),
    ('runner.py:30`, `bulk_runner.py:32', '`scripts/runner.py:40`, `scripts/bulk_runner.py:43`', 're-anchored',
     [('scripts/runner.py', 40, '"std_curiosity": 0.1'),
             ('scripts/bulk_runner.py', 43, '"std_curiosity": 0.1')]),
    ('landscapefunctions.py:56-64 (comment block)', '**unresolvable** (block deleted)', 'unresolvable',
     []),
    ('mesa_model.py:265-272', '`sciencemodels/model.py:87-94`', 're-anchored',
     [('sciencemodels/model.py', 87, 'if self.knowledge_normalization == "share"'),
             ('sciencemodels/model.py', 94, '* 0.95 + 0.05')]),
    ('mesa_model.py:270-272', '`sciencemodels/model.py:91-94`', 're-anchored',
     [('sciencemodels/model.py', 94, '* 0.95 + 0.05')]),
    ('mesa_model.py:266-268', '`sciencemodels/model.py:87-90`', 're-anchored',
     [('sciencemodels/model.py', 89, 'val / self.total_initial_knowledge')]),
    ('mesa_model.py:327,265,533', '`sciencemodels/model.py:159,174,463`', 're-anchored',
     [('sciencemodels/model.py', 159, 'self.new_questions = new_questions'),
             ('sciencemodels/model.py', 174, 'self.knowledge_normalization = knowledge_normalization'),
             ('sciencemodels/model.py', 463, 'if self.new_questions:')]),
    ('mesa_model.py:281', '`sciencemodels/model.py:103`', 're-anchored',
     [('sciencemodels/model.py', 103, 'self.total_initial_knowledge = np.float64(1.0)')]),
    ('mesa_model.py:274-279', '`sciencemodels/model.py:96-101`', 're-anchored',
     [('sciencemodels/model.py', 96, '"initial_knowledge"'),
             ('sciencemodels/model.py', 99, '"previous_knowledge"')]),
    ('mesa_model.py:503-507', '`sciencemodels/model.py:377-388`', 're-anchored',
     [('sciencemodels/model.py', 377, 'def farm'),
             ('sciencemodels/model.py', 380, '* (1 - self.harvest)')]),
    ('runner.py:41`, `bulk_runner.py:50', '`scripts/runner.py:50`, `scripts/bulk_runner.py:61`', 're-anchored',
     [('scripts/runner.py', 50, '"harvest":0.02'),
             ('scripts/bulk_runner.py', 61, '"harvest": 0.02')]),
    ('mesa_model.py:507', '`sciencemodels/model.py:377-388,462-469`', 're-anchored',
     [('sciencemodels/model.py', 388, 'self.update_knowledge()'),
             ('sciencemodels/model.py', 462, 'def _generate_new_questions')]),
    ('mesa_model.py:441-444', '`sciencemodels/model.py:357-360`', 're-anchored',
     [('sciencemodels/model.py', 357, 'def update_knowledge'),
             ('sciencemodels/model.py', 359, 'last_tile_knowledge')]),
    ('mesa_model.py:532-539', '`sciencemodels/model.py:462-469`', 're-anchored',
     [('sciencemodels/model.py', 466, '< 0.025')]),
    ('mesa_model.py:534', '`sciencemodels/model.py:464`', 're-anchored',
     [('sciencemodels/model.py', 464, 'occupied_pos = list(set(')]),
    ('mesa_model.py:59', '`sciencemodels/agents.py:43`', 'moved-module',
     [('sciencemodels/agents.py', 43, 'self.age = 23')]),
    ('mesa_model.py:290', '`sciencemodels/model.py:112`', 're-anchored',
     [('sciencemodels/model.py', 112, 'rng.choice(np.arange(23, 50))')]),
    ('mesa_model.py:206', '`sciencemodels/agents.py:185`', 'moved-module',
     [('sciencemodels/agents.py', 185, 'self.age += 1')]),
    ('mesa_model.py:45,206 (whole class)', '`sciencemodels/agents.py:43,185` (class 21-183)', 'moved-module',
     [('sciencemodels/agents.py', 43, 'self.age = 23'),
             ('sciencemodels/agents.py', 185, 'self.age += 1')]),
    ('mesa_model.py:289', '`sciencemodels/model.py:111`', 're-anchored',
     [('sciencemodels/model.py', 111, 'self.rng.integers(0, self.size)')]),
    ('mesa_model.py:288', '`sciencemodels/model.py:110`', 're-anchored',
     [('sciencemodels/model.py', 110, 'Scientist(self, w, self.epsilon)')]),
    ('mesa_model.py:56-58', '`sciencemodels/agents.py:40-42`', 'moved-module',
     [('sciencemodels/agents.py', 40, 'self.prestige = np.float64(0.0)'),
             ('sciencemodels/agents.py', 42, 'self.prestige_vanishing = np.float64(0.0)')]),
    ('mesa_model.py:313-315', '`sciencemodels/model.py:136-138`', 're-anchored',
     [('sciencemodels/model.py', 138, 'lambda cur, rng, params: cur')]),
    ('landscapefunctions.py:57-64', '`sciencemodels/landscapes.py:57-64`', 're-anchored',
     [('sciencemodels/landscapes.py', 57, 'def beta'),
             ('sciencemodels/landscapes.py', 59, 'params.get("std_curiosity", 0.1)')]),
    ('landscapefunctions.py:67-68', '`sciencemodels/landscapes.py:67-68`', 're-anchored',
     [('sciencemodels/landscapes.py', 67, 'def uniform'),
             ('sciencemodels/landscapes.py', 68, 'rng.uniform(0.0, 1.0)')]),
    ('bulk_runner.py:38-43', '`scripts/bulk_runner.py:49-54`', 're-anchored',
     [('scripts/bulk_runner.py', 50, '[k / 10 for k in range(11)]'),
             ('scripts/bulk_runner.py', 54, 'landscapes.uniform')]),
    ('runner.py:37', '`scripts/runner.py:46`', 're-anchored',
     [('scripts/runner.py', 46, '# minimum 3')]),
    ('bulk_runner.py:46', '`scripts/bulk_runner.py:57`', 're-anchored',
     [('scripts/bulk_runner.py', 57, '"n_agents": 20')]),
    ('mesa_model.py:174-176', '`sciencemodels/agents.py:152-154`', 'moved-module',
     [('sciencemodels/agents.py', 152, 'get_neighborhood'),
             ('sciencemodels/agents.py', 153, 'moore=False, include_center=False')]),
    ('mesa_model.py:182-198', '`sciencemodels/agents.py:160-179`', 'moved-module',
     [('sciencemodels/agents.py', 160, 'best_reward = total_current_reward'),
             ('sciencemodels/agents.py', 176, 'if best_reward - total_current_reward > 0:')]),
    ('mesa_model.py:188-190', '`sciencemodels/agents.py:166-169`', 'moved-module',
     [('sciencemodels/agents.py', 166, 'noise = self.rng.uniform(-1.0, 1.0)'),
             ('sciencemodels/agents.py', 168, '* (1 + self.epsilon * noise)')]),
    ('runner.py:40`, `bulk_runner.py:49', '`scripts/runner.py:49`, `scripts/bulk_runner.py:60`', 're-anchored',
     [('scripts/runner.py', 49, '"epsilon": 0.1'),
             ('scripts/bulk_runner.py', 60, '"epsilon": 0.1')]),
    ('mesa_model.py:178-181', '`sciencemodels/agents.py:156-159`', 'moved-module',
     [('sciencemodels/agents.py', 156, 'current_rw_novelty = self.model.compute_knowledge(self.pos)'),
             ('sciencemodels/agents.py', 157, 'total_current_reward, currentVis')]),
    ('mesa_model.py:177', '`sciencemodels/agents.py:155`', 'moved-module',
     [('sciencemodels/agents.py', 155, '# optionpos = self.model.rng.choice(neighbors_nodes)')]),
    ('mesa_model.py:152', '`sciencemodels/agents.py:144`', 'moved-module',
     [('sciencemodels/agents.py', 144, '#    reward = self.curiosity * Novelty/')]),
    ('mesa_model.py:147-150', '`sciencemodels/agents.py:139-142`', 'moved-module',
     [('sciencemodels/agents.py', 140, '(1) the above'),
             ('sciencemodels/agents.py', 142, 'weighted by prestige')]),
    ('mesa_model.py:599-601', '`sciencemodels/model.py:520-524`', 're-anchored',
     [('sciencemodels/model.py', 524, 'temp.shuffle_do("step")')]),
    ('mesa_model.py:107-108', '`sciencemodels/agents.py:99-100`', 'moved-module',
     [('sciencemodels/agents.py', 99, 'This part is just a test')]),
    ('mesa_model.py:143-145', '`sciencemodels/agents.py:135-137`', 'moved-module',
     [('sciencemodels/agents.py', 135, 'reward = (novelty / self.model.avg_current_agent_knowledge) ** ('),
             ('sciencemodels/agents.py', 137, '* visibility_used ** (1 - self.curiosity)')]),
    ('mesa_model.py:111,143', '`sciencemodels/agents.py:103,135`', 'moved-module',
     [('sciencemodels/agents.py', 103, 'self.curiosity * novelty / self.model.avg_current_agent_knowledge'),
             ('sciencemodels/agents.py', 135, 'avg_current_agent_knowledge')]),
    ('mesa_model.py:109-113', '`sciencemodels/agents.py:101-105`', 'moved-module',
     [('sciencemodels/agents.py', 101, 'if not self.model.use_distance:')]),
    ('mesa_model.py:101-102', '`sciencemodels/agents.py:93-94`', 'moved-module',
     [('sciencemodels/agents.py', 94, '+ 1.0) / (avg_agent_distance + 1.0)')]),
    ('mesa_model.py:576-585', '`sciencemodels/model.py:495-504`', 're-anchored',
     [('sciencemodels/model.py', 495, 'self.agent_avg_distance = np.mean(')]),
    ('mesa_model.py:66-80', '`sciencemodels/agents.py:58-72`', 'moved-module',
     [('sciencemodels/agents.py', 58, 'def compute_distance'),
             ('sciencemodels/agents.py', 67, 'dx = min(dx, n - dx)')]),
    ('mesa_model.py:136-139,317', '`sciencemodels/agents.py:128-132`, `sciencemodels/model.py:167`', 'moved-module',
     [('sciencemodels/agents.py', 128, 'if self.model.use_visibility_reward:'),
             ('sciencemodels/model.py', 167, 'self.use_visibility_reward = use_visibility_reward')]),
    ('mesa_model.py:121-133', '`sciencemodels/agents.py:113-125`', 'moved-module',
     [('sciencemodels/agents.py', 114, 'a.prestige_vanishing'),
             ('sciencemodels/agents.py', 121, 'if prest.sum() > 0.0:'),
             ('sciencemodels/agents.py', 122, 'np.average(')]),
    ('mesa_model.py:91|prestige_visibility', '`sciencemodels/agents.py:83`', 'moved-module',
     [('sciencemodels/agents.py', 83, 'we could also use individual distances')]),
    ('mesa_model.py:141-142', '`sciencemodels/agents.py:133-134`', 'moved-module',
     [('sciencemodels/agents.py', 133, 'visibility of a field vs. visibility of an agent?')]),
    ('mesa_model.py:27-36', '**unresolvable** (helper deleted)', 'unresolvable',
     []),
    ('mesa_model.py:355-363', '`sciencemodels/model.py:262-270`', 're-anchored',
     [('sciencemodels/model.py', 262, 'if not ('),
             ('sciencemodels/model.py', 269, 'Agents will NOT use distance information')]),
    ('mesa_model.py:204,207-210,82-83', '`sciencemodels/agents.py:183-205`', 'moved-module',
     [('sciencemodels/agents.py', 183, 'self.model.farm(self.pos, self)'),
             ('sciencemodels/agents.py', 203, 'increase_prestige(knowledge_before * self.model.harvest)')]),
    ('mesa_model.py:84-90', '`sciencemodels/agents.py:76-82`', 'moved-module',
     [('sciencemodels/agents.py', 76, 'self.prestige_visibility += np.mean('),
             ('sciencemodels/agents.py', 78, 'value ** (a.curiosity) * self.visibility')]),
    ('mesa_model.py:91|prestige_vanishing', '`sciencemodels/agents.py:83`', 'moved-module',
     [('sciencemodels/agents.py', 83, 'individual distances')]),
    ('mesa_model.py:92-99,316', '`sciencemodels/agents.py:84-91`, `sciencemodels/model.py:139`', 'moved-module',
     [('sciencemodels/agents.py', 84, 'x = self.model.vanishing_factor_for_prestige'),
             ('sciencemodels/model.py', 139, 'vanishing_factor_for_prestige: np.float64 = np.float64(0.9)')]),
    ('runner.py:50', '`scripts/runner.py:59`', 're-anchored',
     [('scripts/runner.py', 59, '"vanishing_factor_for_prestige": 0.9')]),
    ('bulk_runner.py:59', '`scripts/bulk_runner.py:70`', 're-anchored',
     [('scripts/bulk_runner.py', 70, '"vanishing_factor_for_prestige": 0.9')]),
    ('mesa_model.py:453-497', '`sciencemodels/metrics.py:53-97`', 'moved-module',
     [('sciencemodels/metrics.py', 53, 'def update_ideal_merit')]),
    ('mesa_model.py:469,476-477', '`sciencemodels/metrics.py:69,78-79`', 'moved-module',
     [('sciencemodels/metrics.py', 69, '((1- model.harvest) ** redone)'),
             ('sciencemodels/metrics.py', 79, 'temp_knowledge_map[pos] = temp_knowledge_map[pos] * (1- model.harvest)')]),
    ('mesa_model.py:455,497', '`sciencemodels/metrics.py:55,107`', 'moved-module',
     [('sciencemodels/metrics.py', 55, 'model.grid.properties["previous_knowledge"].data'),
             ('sciencemodels/metrics.py', 107, '.data.copy()')]),
    ('mesa_model.py:509-513', '`sciencemodels/model.py:390-394`', 're-anchored',
     [('sciencemodels/model.py', 390, 'def __generate_new_agents'),
             ('sciencemodels/model.py', 393, 'self.steps * self.agent_generation_time')]),
    ('runner.py:45`, `bulk_runner.py:54', '`scripts/runner.py:54`, `scripts/bulk_runner.py:65`', 're-anchored',
     [('scripts/runner.py', 54, '"agent_generation_time": 0'),
             ('scripts/bulk_runner.py', 65, '"agent_generation_time": 0')]),
    ('mesa_model.py:514-523', '`sciencemodels/model.py:396-407`', 're-anchored',
     [('sciencemodels/model.py', 396, 'prestige_agent_tuples'),
             ('sciencemodels/model.py', 404, 'if val <= p:')]),
    ('mesa_model.py:515-523', '`sciencemodels/model.py:402-407`', 're-anchored',
     [('sciencemodels/model.py', 402, 'self.rng.uniform(0, totalPrestige)')]),
    ('mesa_model.py:525', '`sciencemodels/model.py:409-412`', 're-anchored',
     [('sciencemodels/model.py', 409, 'w = self.agent_generation_function(')]),
    ('mesa_model.py:526-527', '`sciencemodels/model.py:419-420`', 're-anchored',
     [('sciencemodels/model.py', 419, 'coords = sup.pos')]),
    ('mesa_model.py:59,290', '`sciencemodels/agents.py:43`, `sciencemodels/model.py:112`', 'moved-module',
     [('sciencemodels/agents.py', 43, 'self.age = 23'),
             ('sciencemodels/model.py', 112, 'rng.choice(np.arange(23, 50))')]),
    ('mesa_model.py:288,525', '`sciencemodels/model.py:110,412`', 're-anchored',
     [('sciencemodels/model.py', 110, 'self.epsilon'),
             ('sciencemodels/model.py', 412, 'Scientist(self, w, self.epsilon)')]),
    ('runner.py:45|mislabel', '`scripts/runner.py:54`', 're-anchored',
     [('scripts/runner.py', 54, 'if 0.1, 10 new agent per generation')]),
    ('(new row, no v2 predecessor)', '`sciencemodels/model.py:143,181-187,395-399`', 're-anchored',
     [('sciencemodels/model.py', 143, 'entry_prestige_stock: str'),
             ('sciencemodels/model.py', 181, 'if entry_prestige_stock not in ('),
             ('sciencemodels/model.py', 395, 'entry_stock = self.entry_prestige_stock')]),
    ('(new row, no v2 predecessor)', '`sciencemodels/metrics.py:105`, `sciencemodels/model.py:416`', 're-anchored',
     [('sciencemodels/metrics.py', 105, 'a.ideal_merit_cum = getattr(a, "ideal_merit_cum", 0.0) + a.ideal_Merit'),
             ('sciencemodels/model.py', 416, 'a.ideal_merit_cum = np.float64(0.0)')]),
    ('mesa_model.py:155-170', '`sciencemodels/metrics.py:35-50`', 'moved-module',
     [('sciencemodels/metrics.py', 35, 'def local_merit'),
             ('sciencemodels/metrics.py', 41, 'positions = [pos, *neighbors]')]),
    ('mesa_model.py:453-497|ideal', '`sciencemodels/metrics.py:53-97`', 'moved-module',
     [('sciencemodels/metrics.py', 53, 'def update_ideal_merit')]),
    ('mesa_model.py:367-375', '`sciencemodels/metrics.py:308-316`', 'moved-module',
     [('sciencemodels/metrics.py', 308, 'def correlation_reporter'),
             ('sciencemodels/metrics.py', 313, 'np.corrcoef')]),
    ('mesa_model.py:423-428', '`sciencemodels/model.py:328-333`', 're-anchored',
     [('sciencemodels/model.py', 328, 'Corr_prestige_local_merit'),
             ('sciencemodels/model.py', 333, 'Corr_prestige_vanishing_ideal_merit')]),
    ('mesa_model.py:545-552', '`sciencemodels/metrics.py:323-330`', 'moved-module',
     [('sciencemodels/metrics.py', 328, '"harvested_10_step"'),
             ('sciencemodels/metrics.py', 329, '"harvested_25_step"')]),
    ('mesa_model.py:344', '`sciencemodels/model.py:251`', 're-anchored',
     [('sciencemodels/model.py', 251, '_default_steps_thresholds = step_limit * (1.25)')]),
    ('mesa_model.py:312', '`sciencemodels/model.py:135`', 're-anchored',
     [('sciencemodels/model.py', 135, 'step_limit: int = 400')]),
    ('runner.py:48', '`scripts/runner.py:57`', 're-anchored',
     [('scripts/runner.py', 57, '"step_limit": 200')]),
    ('bulk_runner.py:57', '`scripts/bulk_runner.py:68`', 're-anchored',
     [('scripts/bulk_runner.py', 68, '"step_limit": 400')]),
    ('mesa_model.py:387-412', '`sciencemodels/model.py:292-317`', 're-anchored',
     [('sciencemodels/model.py', 292, 'top1pct_prestige_share'),
             ('sciencemodels/model.py', 305, 'top10pct_prestige_share')]),
    ('mesa_model.py:566-575', '`sciencemodels/model.py:485-494`', 're-anchored',
     [('sciencemodels/model.py', 485, 'self.explored_percentage = np.sum('),
             ('sciencemodels/model.py', 488, 'explored_weighted_by_initial_knowledge')]),
    ('mesa_model.py:280,451,291', '`sciencemodels/model.py:114,367`', 're-anchored',
     [('sciencemodels/model.py', 114, 'self.grid.properties["explored"].data[coords] = True'),
             ('sciencemodels/model.py', 367, 'self.grid.properties["explored"].data[coords] = True')]),
    ('mesa_model.py:586-589', '`sciencemodels/model.py:505-508`', 're-anchored',
     [('sciencemodels/model.py', 505, 'self.percentage_knowledge_harvested = (')]),
    ('mesa_model.py:645-664', '`sciencemodels/io.py:25-44`', 'moved-module',
     [('sciencemodels/io.py', 25, 'def build_report'),
             ('sciencemodels/io.py', 29, 'if k[0].islower()'),
             ('sciencemodels/io.py', 40, '"Mean_of_"+k')]),
    ('mesa_model.py:856-864', '`sciencemodels/model.py:636-644`', 're-anchored',
     [('sciencemodels/model.py', 636, 'if needs_header or intention'),
             ('sciencemodels/model.py', 640, 'all_params["step_limit"] = 1')]),
    ('mesa_model.py:881-886', '`sciencemodels/model.py:656-661`, `scripts/bulk_runner.py:82-83`', 're-anchored',
     [('sciencemodels/model.py', 661, 'all_params["agent_seed"] = repeat'),
             ('scripts/bulk_runner.py', 82, 'repeats_per_setting=40'),
             ('scripts/bulk_runner.py', 83, 'change_landscape_seed=True')]),
    ('bulk_runner.py:39', '`scripts/bulk_runner.py:50`', 're-anchored',
     [('scripts/bulk_runner.py', 50, '[k / 10 for k in range(11)]')]),
    ('bulk_runner.py:87', '`scripts/bulk_runner.py:97`', 're-anchored',
     [('scripts/bulk_runner.py', 97, 'Parallel(n_jobs=10')]),
    ('mesa_model.py:627-630,670-675', '`sciencemodels/viz.py:48-53,60-67`', 'moved-module',
     [('sciencemodels/viz.py', 49, '(k.unique_id / model.number_agents - 0.5) ** 2'),
             ('sciencemodels/viz.py', 61, '(k.unique_id / model.number_agents - 0.5) ** 2')]),
    ('mesa_model.py:693-696', '`sciencemodels/viz.py:84-87,46`', 'moved-module',
     [('sciencemodels/viz.py', 85, 'vmin=0, vmax=1'),
             ('sciencemodels/viz.py', 46, '# , vmin=0, vmax=1')]),
    ('compare_landscapes_corr-prestige-merit.ipynb', '`notebooks/compare_landscapes_corr-prestige-merit.ipynb` (cell 0)', 're-anchored',
     [('notebooks/compare_landscapes_corr-prestige-merit.ipynb', 28, 'run_number0')]),
    ('compare_landscapes_flexible search.ipynb`, `compare_landscapes_normalizedknowledge.ipynb', '`notebooks/compare_landscapes_flexible_search.ipynb`, `notebooks/compare_landscapes_normalizedknowledge.ipynb` (cell 0)', 're-anchored',
     [('notebooks/compare_landscapes_flexible_search.ipynb', 11, 'sciencemodels'),
             ('notebooks/compare_landscapes_normalizedknowledge.ipynb', 11, 'sciencemodels')]),
    ('longitudinalcomparisions.ipynb (cell 1)', '`sciencemodels/plotting.py:172,204`', 'moved-module',
     [('sciencemodels/plotting.py', 172, 'length=None'),
             ('sciencemodels/plotting.py', 204, 'def plot_all_curiosities')]),
    ('mesa_model.py:605-618', '`sciencemodels/viz.py:26-39,42-55`', 'moved-module',
     [('sciencemodels/viz.py', 26, 'def Vizualize_3d'),
             ('sciencemodels/viz.py', 42, 'def plot(model)')]),
]

# Anchors cited in the prose bullets and in the role column, rather than in a
# Reference cell. Checked separately so a moving target fails the same way.
INLINE_ANCHORS = [
    ('sciencemodels/model.py', 466, '0.025'),
    ('sciencemodels/model.py', 396, 'prestige_agent_tuples'),
    ('sciencemodels/model.py', 419, 'coords = sup.pos'),
    ('sciencemodels/agents.py', 176, 'if best_reward - total_current_reward > 0:'),
    ('sciencemodels/agents.py', 156, 'current_rw_novelty = self.model.compute_knowledge(self.pos)'),
    ('sciencemodels/metrics.py', 328, '"harvested_10_step"'),
    ('sciencemodels/metrics.py', 329, '"harvested_25_step"'),
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
