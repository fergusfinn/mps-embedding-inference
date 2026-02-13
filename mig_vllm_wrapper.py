"""Wrapper to launch vLLM on a MIG partition.

vLLM assumes CUDA_VISIBLE_DEVICES contains integer device IDs, but MIG
uses UUIDs (e.g. MIG-4f79e892-...). This wrapper initializes CUDA with
the MIG UUID, then overrides the env var to '0' before vLLM reads it.

Usage:
    CUDA_VISIBLE_DEVICES=MIG-xxx python mig_vllm_wrapper.py [vllm args...]
"""
import os
import sys

# Force CUDA runtime to initialize and scope to the MIG partition
import torch
torch.cuda.init()

# Override so vLLM sees a plain integer device ID
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Now run vLLM's api_server as if invoked via python -m
from vllm.entrypoints.openai.api_server import (
    FlexibleArgumentParser, cli_env_setup, make_arg_parser,
    run_server, validate_parsed_serve_args,
)
import uvloop

cli_env_setup()
parser = FlexibleArgumentParser(description="vLLM on MIG")
parser = make_arg_parser(parser)
args = parser.parse_args()
validate_parsed_serve_args(args)
uvloop.run(run_server(args))
