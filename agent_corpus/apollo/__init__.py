"""Drivora ↔ Baidu Apollo integration (CARLA + real-sensor perception).

This package is fully self-contained: it does NOT import anything from any
external project except, indirectly, the Apollo source tree (the official Baidu
Apollo repo) which the Apollo Docker container bind-mounts at runtime. The
CyberBridge communication layer here is an independent, self-contained rewrite
owned by Drivora.

Entry point for Drivora's agent loader:
    agent_corpus.apollo.apollo_real_agent:ApolloRealAgent

See ``agent_corpus/apollo/README.md`` for usage. Running it requires a built
Apollo + HD map + sensor calibration.
"""
