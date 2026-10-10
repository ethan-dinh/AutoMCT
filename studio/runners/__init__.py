"""
Non-interactive entry points the job queue runs as subprocesses.

Each wraps a function the repository already has -- the converters' shared
pipeline, ``segment_mandible``, the preview builder -- without the prompts and
viewers its own CLI opens, so a job can run unattended. Progress is reported
on stdout as ``@@progress <fraction> <label>`` lines, which the queue parses.
"""
