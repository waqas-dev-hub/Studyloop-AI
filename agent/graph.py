from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import CompiledStateGraph
from .state import AgentState
from .nodes import observe_state, diagnose_weaknesses, choose_teaching_action, execute_action, critic_verify, human_approval_check, update_state_node, should_continue, should_retry, needs_approval

def create_agent_graph() -> CompiledStateGraph:
    graph = StateGraph(AgentState)
    graph.add_node('observe', observe_state)
    graph.add_node('diagnose', diagnose_weaknesses)
    graph.add_node('decide', choose_teaching_action)
    graph.add_node('execute', execute_action)
    graph.add_node('verify', critic_verify)
    graph.add_node('approval', human_approval_check)
    graph.add_node('update', update_state_node)
    graph.add_edge(START, 'observe')
    graph.add_edge('observe', 'diagnose')
    graph.add_edge('diagnose', 'decide')
    graph.add_edge('decide', 'execute')
    graph.add_edge('execute', 'verify')
    graph.add_conditional_edges('verify', should_retry, {'retry': 'execute', 'proceed': 'approval'})
    graph.add_conditional_edges('approval', needs_approval, {'needs_approval': END, 'skip_approval': 'update'})
    graph.add_conditional_edges('update', should_continue, {'continue': 'observe', 'end': END})
    return graph.compile(checkpointer=MemorySaver())