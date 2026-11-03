"""The LangGraph workflow itself: nodes wired together by the routing rules.

    START -> ingest --+-- (message) --> user_interaction --+-- cancelled --> END
                      |                                    +-- admin_approval
                      +-- (tick, booking pending) ---------+-- (nothing) --> END
                      +-- (tick, nothing pending) --> END

    admin_approval --+-- approved --> record --> notify --> END
                     +-- refused --> notify --> END
                     +-- expired --> notify --> END
                     +-- pending --> END   (resumed by the next turn or tick)

Everything the graph needs from the outside - the chat agent, the booking reader,
the recording client, the checkpointer - is injected, so the same graph runs
against the real components in the CLI and against fakes in the tests.
"""
import logging
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph

from parking_chatbot.orchestration import routing
from parking_chatbot.orchestration.checkpointing import build_checkpointer
from parking_chatbot.orchestration.nodes import WorkflowDependencies, WorkflowNodes
from parking_chatbot.orchestration.state import TurnKind, WorkflowState, initial_state, turn_input

logger = logging.getLogger(__name__)


def build_workflow(
    deps: WorkflowDependencies | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> Any:
    """Compile the workflow graph.

    Args:
        deps: the components the nodes call out to. Defaults to the real Stage 1
            to 3 components with a freshly built chat agent - built lazily, here,
            so importing this module never constructs an LLM client.
        checkpointer: where shared state is persisted. Defaults to the configured
            backend.

    Returns:
        The compiled graph. Invoke it with a thread id in the config, the same
        way the Stage 1 and Stage 2 agents are invoked.
    """
    if deps is None:
        from parking_chatbot.chatbot.agent import build_agent

        deps = WorkflowDependencies(agent=build_agent())
    nodes = WorkflowNodes(deps)

    builder = StateGraph(WorkflowState)
    builder.add_node(routing.INGEST, nodes.ingest)
    builder.add_node(routing.USER_INTERACTION, nodes.user_interaction)
    builder.add_node(routing.ADMIN_APPROVAL, nodes.admin_approval)
    builder.add_node(routing.RECORD, nodes.record)
    builder.add_node(routing.NOTIFY, nodes.notify)
    builder.add_node(routing.CANCELLED, nodes.cancelled)

    builder.add_edge(START, routing.INGEST)
    # Explicit path maps rather than bare routers: they document the reachable
    # destinations, and they are what lets the graph be drawn accurately.
    builder.add_conditional_edges(
        routing.INGEST,
        routing.route_turn,
        {routing.USER_INTERACTION: routing.USER_INTERACTION, routing.ADMIN_APPROVAL: routing.ADMIN_APPROVAL, END: END},
    )
    builder.add_conditional_edges(
        routing.USER_INTERACTION,
        routing.route_after_user_interaction,
        {routing.CANCELLED: routing.CANCELLED, routing.ADMIN_APPROVAL: routing.ADMIN_APPROVAL, END: END},
    )
    builder.add_conditional_edges(
        routing.ADMIN_APPROVAL,
        routing.route_after_approval,
        {routing.RECORD: routing.RECORD, routing.NOTIFY: routing.NOTIFY, END: END},
    )
    # Unconditional: both a successful and a failed recording have to be reported,
    # and the notify node picks the wording from the state the record node wrote.
    builder.add_edge(routing.RECORD, routing.NOTIFY)
    builder.add_edge(routing.NOTIFY, END)
    builder.add_edge(routing.CANCELLED, END)

    return builder.compile(checkpointer=checkpointer or build_checkpointer())


def run_turn(
    workflow: Any,
    thread_id: str,
    user_input: str = "",
    turn_kind: TurnKind = "message",
) -> WorkflowState:
    """Run one turn of the conversation and return the state it ended in.

    The first turn of a thread seeds the full initial state; later turns send
    only the per-turn fields, and the checkpointer supplies the rest - which is
    how a pending booking survives between turns.
    """
    run_config = {"configurable": {"thread_id": thread_id}}
    if workflow.get_state(run_config).created_at is None:
        update: WorkflowState = {**initial_state(thread_id), **turn_input(user_input, turn_kind=turn_kind)}
    else:
        update = turn_input(user_input, turn_kind=turn_kind)
    return workflow.invoke(update, config=run_config)


def workflow_state(workflow: Any, thread_id: str) -> WorkflowState:
    """The checkpointed state of one conversation, without running a turn.

    Used by the CLI to decide whether a background tick is worth running at all,
    and by the tests to prove a pending approval was persisted.
    """
    return workflow.get_state({"configurable": {"thread_id": thread_id}}).values or {}
