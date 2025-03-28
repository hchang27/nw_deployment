from pathlib import Path

from ml_logger.job import RUN, instr
from termcolor import colored

assert instr, "force import"  # single-entry for the instrumentation thunk factory
RUN.project = "lucidsim"  # Specify the project name
RUN.job_name += "/{job_counter}"

# RUN.prefix = "{project}/{project}/{username}/{now:%Y/%m-%d}/{file_stem}/{job_name}"
# RUN.prefix = "{project}/{project}/{username}/{file_stem}/{job_name}"
# # RUN.prefix = "lucid-sim/lucid-sim/{username}/{file_stem}/{job_name}"
# RUN.prefix = f"lucid-sim/lucid-sim/{{file_stem}}/{{job_name}}"

# WARNING: do NOT change these prefixes.
RUN.prefix = "{project}/{project}/parkour/{file_stem}/{job_name}"
RUN.script_root = Path(__file__).parent  # specify that this is the script root.

print(colored("set", "blue"), colored("RUN.script_root", "yellow"), colored("to", "blue"), RUN.script_root)
