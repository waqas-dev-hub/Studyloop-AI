from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    subject_id: int
    student_id: str
    mastery_data: dict
    quiz_history: list[dict]
    misconceptions: list[dict]
    days_until_exam: int
    missed_sessions: int
    diagnosis: dict
    chosen_action: dict
    action_reasoning: str
    action_result: dict
    critic_passed: bool
    critic_feedback: str
    needs_human_approval: bool
    plan_proposal: dict
    human_approved: bool
    questions_asked: int
    correct_answers: int
    session_complete: bool
    messages: Annotated[list, add_messages]