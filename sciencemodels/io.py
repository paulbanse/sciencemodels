"""CSV and report writing.

Extracted verbatim from the pre-restructure ``mesa_model.py``:

- ``is_non_zero_file`` (was a module-level helper)
- ``build_report``     (was ``MyModel.get_report``; ``self.__getattribute__(k)``
  is spelled ``getattr(model, k)`` here, which is the identical lookup)
- ``save_run_csvs``    (was the CSV-writing tail of ``MyModel.animate_steps``)
- ``write_end_report`` (was the end-report tail of ``MyModel.animate_steps``)

``MyModel.get_report`` delegates to ``build_report``; the visualization module
calls ``save_run_csvs`` / ``write_end_report`` where ``animate_steps`` used to
write inline. All file paths and print strings are unchanged.
"""
from __future__ import annotations

import csv
import os


def is_non_zero_file(fpath):
    return os.path.isfile(fpath) and os.path.getsize(fpath) > 0


def build_report(model):
    a = {
        k: model.datacollector.get_model_vars_dataframe()[k].iloc[-1]
        for k in model.datacollector.get_model_vars_dataframe().columns 
        if k[0].islower()
    }
    b = {
        k: getattr(model, k)
        for k in model.__dict__
        if (
            type(getattr(model, k)) in [int, float, str, list, dict]
            and k[0] != "_"
        )
    }
    c = {
        "Mean_of_"+k: model.datacollector.get_model_vars_dataframe()[k].mean()
        for k in model.datacollector.get_model_vars_dataframe().columns 
        if k[0].isupper()
    }
    return {**a, **b, **c}


def save_run_csvs(model, csv_name):
    model.datacollector.get_agent_vars_dataframe().to_csv(
        "data/agent_" + csv_name + ".csv"
    )
    model.datacollector.get_model_vars_dataframe().to_csv(
        "data/model_" + csv_name + ".csv"
    )
    print(
        "data saved to ",
        "agent_" + csv_name + ".csv",
        "model_" + csv_name + ".csv",
    )


def write_end_report(model, end_report_file):
    row = build_report(model)
    needs_header = not (is_non_zero_file("data/" + end_report_file))
    with open("data/" + end_report_file, "a") as f:
        writer = csv.writer(f)
        if needs_header:
            writer.writerow(sorted(row.keys()))
        row = [row[k] for k in sorted(row.keys())]
        writer.writerow(row)
    print("end report saved to ", end_report_file)
