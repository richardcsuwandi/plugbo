#!/usr/bin/env bash
# Lane B of the blog seed sweep: 1 LLM job at a time.
# Pair with ./scripts/run_blog_seeds_a.sh in a second terminal (2 LLM total).
#
#   ackley10, bolt_lora domain
exec "$(dirname "$0")/run_blog_seeds.sh" --lane b --logdir /tmp/blog_seeds_logs/b "$@"
