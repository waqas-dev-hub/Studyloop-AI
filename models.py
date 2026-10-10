from datetime import date, datetime
from typing import Optional, Literal
from pydantic import BaseModel, Field

class SubjectCreate(BaseModel):
    name: str = Field(..., description="Subject name, e.g. 'Physics'")
    exam_date: date = Field(..., description='Exam date (YYYY-MM-DD)')

class SubjectResponse(BaseModel):
    id: int
    name: str
    exam_date: date
    topic_count: int = 0
    avg_mastery: float = 0.0
    days_until_exam: int = 0

    class Config:
        from_attributes = True

class TopicResponse(BaseModel):
    id: int
    name: str
    mastery_score: float
    difficulty: str
    last_reviewed: Optional[datetime] = None
    review_count: int
    misconception_count: int = 0

    class Config:
        from_attributes = True

class TopicMastery(BaseModel):
    topic_id: int
    topic_name: str
    mastery_score: float
    trend: str = 'stable'
    misconceptions: list[str] = []
    last_reviewed: Optional[datetime] = None

class QuizQuestion(BaseModel):
    question: str
    question_type: Literal['mcq', 'short_answer', 'explain_back'] = 'mcq'
    options: Optional[list[str]] = None
    difficulty: str = 'medium'
    topic_id: int
    topic_name: str

class QuizAnswerRequest(BaseModel):
    session_id: int
    question: str
    student_answer: str
    topic_id: int

class GradeResult(BaseModel):
    is_correct: bool
    correct_answer: str
    explanation: str
    misconception_detected: Optional[str] = None
    mastery_change: float = 0.0
    new_mastery: float = 0.0

class AgentAction(BaseModel):
    action_type: Literal['quiz', 'explain', 'worked_example', 'easier_question', 'explain_back', 'move_on', 'replan', 'mock_paper']
    topic_id: Optional[int] = None
    topic_name: Optional[str] = None
    reasoning: str = ''
    content: Optional[dict] = None

class DiagnosisResult(BaseModel):
    weak_topics: list[str] = []
    misconceptions: list[dict] = []
    knowledge_gaps: list[str] = []
    overall_assessment: str = ''
    recommended_focus: str = ''

class StudyPlanEntryResponse(BaseModel):
    id: int
    date: date
    topic_name: Optional[str] = None
    action_type: str
    status: str
    priority: int
    notes: Optional[str] = None

    class Config:
        from_attributes = True

class PlanProposal(BaseModel):
    subject_id: int
    reasoning: str
    changes: list[dict]
    entries: list[StudyPlanEntryResponse] = []

class PlanApproval(BaseModel):
    approved: bool
    adjustments: Optional[str] = None

class SessionStart(BaseModel):
    subject_id: int

class SessionResponse(BaseModel):
    session_id: int
    subject_name: str
    days_until_exam: int
    todays_topics: list[str]
    current_action: AgentAction

class SessionNextAction(BaseModel):
    action: AgentAction
    session_stats: dict = {}
    mastery_update: Optional[TopicMastery] = None

class MockPaperQuestion(BaseModel):
    number: int
    question: str
    question_type: str
    options: Optional[list[str]] = None
    marks: int
    topic_name: str
    difficulty: str

class MockPaper(BaseModel):
    title: str
    subject: str
    total_marks: int
    duration_minutes: int
    questions: list[MockPaperQuestion]
    focus_areas: list[str]

class MockPaperResult(BaseModel):
    total_score: int
    max_score: int
    percentage: float
    topic_scores: dict[str, dict]
    weak_areas: list[str]
    recommendations: list[str]

class UploadResponse(BaseModel):
    document_id: int
    filename: str
    topics_extracted: list[str]
    topic_count: int

class DashboardData(BaseModel):
    subject: SubjectResponse
    mastery_data: list[TopicMastery]
    recent_quiz_stats: dict
    plan_completion: dict
    streak: int
    mock_available: bool