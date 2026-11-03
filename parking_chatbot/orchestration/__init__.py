"""Stage 4: the LangGraph workflow that runs the whole pipeline as one graph.

Every node in here is a thin wrapper around a component built in Stages 1 to 3 -
the RAG chat agent, the administrator approval data, the MCP recording client.
None of their logic is reimplemented or copied; this package only decides what
runs next and what the user is told.
"""
