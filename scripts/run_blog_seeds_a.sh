#!/usr/bin/env bash
# Lane A of the blog seed sweep: 1 LLM job at a time.
# Pair with ./scripts/run_blog_seeds_b.sh in a second terminal (2 LLM total).
#
#   hartmann6, gp_sample6, bolt_lora misleading (RQ4 subset)
exec "$(dirname "$0")/run_blog_seeds.sh" --lane a --logdir /tmp/blog_seeds_logs/a "$@"
