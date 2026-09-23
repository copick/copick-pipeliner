"""The ``copick-pipeliner-tools`` side: runs where copick is, does the actual work.

``coords``              coordinate and orientation conventions (pure numpy/scipy)
``portal_annotations``  cryoET Data Portal mirror readers (JSON/NDJSON; no copick)
``export_star``         RELION particle STAR + manifest writers
``external``            argv composition for the copick / octopi CLIs and a runner
``orchestrate``         the per-job orchestrations the CLI verbs call
``cli``                 the click entry point
"""
