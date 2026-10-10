import os
import json
import uuid
from datetime import datetime, date, timedelta
from typing import List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status
from sqlalchemy.orm import Session
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from database import get_db, Subject, Topic, StudyPlanEntry, UploadedDocument, QuizHistory, Misconception, PlanStatus, ActionType
from models import SubjectCreate, SubjectResponse, TopicMastery, UploadResponse, SessionStart, SessionResponse, SessionNextAction, QuizAnswerRequest, GradeResult, StudyPlanEntryResponse, PlanApproval, DashboardData, MockPaper, MockPaperResult, AgentAction
load_dotenv()
try:
    from agent import create_agent_graph
    agent_graph = create_agent_graph()
except Exception:
    agent_graph = None
router = APIRouter()
active_sessions: dict[int, dict] = {}
_session_counter = 0
MODEL_NAME = os.getenv('MODEL_NAME', 'meta-llama/llama-4-scout-17b-16e-instruct')

def get_llm(temperature: float=0.3) -> ChatGroq:
    return ChatGroq(model=MODEL_NAME, temperature=temperature, max_tokens=4096)

def _next_session_id() -> int:
    global _session_counter
    _session_counter += 1
    return _session_counter

@router.post('/subjects', response_model=SubjectResponse, status_code=status.HTTP_201_CREATED)
def create_subject(subject: SubjectCreate, db: Session=Depends(get_db)):
    db_subject = Subject(name=subject.name, exam_date=subject.exam_date)
    db.add(db_subject)
    db.commit()
    db.refresh(db_subject)
    days_until = max(0, (db_subject.exam_date - date.today()).days)
    return SubjectResponse(id=db_subject.id, name=db_subject.name, exam_date=db_subject.exam_date, topic_count=0, avg_mastery=0.0, days_until_exam=days_until)

@router.get('/subjects', response_model=List[SubjectResponse])
def list_subjects(db: Session=Depends(get_db)):
    subjects = db.query(Subject).all()
    result = []
    for sub in subjects:
        topics = db.query(Topic).filter(Topic.subject_id == sub.id).all()
        tc = len(topics)
        avg = sum((t.mastery_score for t in topics)) / tc if tc > 0 else 0.0
        days = max(0, (sub.exam_date - date.today()).days)
        result.append(SubjectResponse(id=sub.id, name=sub.name, exam_date=sub.exam_date, topic_count=tc, avg_mastery=avg, days_until_exam=days))
    return result

@router.get('/subjects/{subject_id}', response_model=SubjectResponse)
def get_subject(subject_id: int, db: Session=Depends(get_db)):
    sub = db.query(Subject).filter(Subject.id == subject_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail='Subject not found')
    topics = db.query(Topic).filter(Topic.subject_id == sub.id).all()
    tc = len(topics)
    avg = sum((t.mastery_score for t in topics)) / tc if tc > 0 else 0.0
    days = max(0, (sub.exam_date - date.today()).days)
    return SubjectResponse(id=sub.id, name=sub.name, exam_date=sub.exam_date, topic_count=tc, avg_mastery=avg, days_until_exam=days)

@router.delete('/subjects/{subject_id}', status_code=status.HTTP_204_NO_CONTENT)
def delete_subject(subject_id: int, db: Session=Depends(get_db)):
    sub = db.query(Subject).filter(Subject.id == subject_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail='Subject not found')
    db.delete(sub)
    db.commit()

@router.post('/subjects/{subject_id}/upload', response_model=UploadResponse)
async def upload_document(subject_id: int, file: UploadFile=File(...), db: Session=Depends(get_db)):
    sub = db.query(Subject).filter(Subject.id == subject_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail='Subject not found')
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in ('pdf', 'txt'):
        raise HTTPException(status_code=400, detail='Upload PDF or TXT files.')
    os.makedirs('uploads', exist_ok=True)
    file_path = os.path.join('uploads', f'{uuid.uuid4()}_{file.filename}')
    with open(file_path, 'wb') as f:
        f.write(await file.read())
    try:
        if ext == 'pdf':
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)
        docs = loader.load()
        full_text = '\n\n'.join((doc.page_content for doc in docs))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to process file: {e}')
    doc_record = UploadedDocument(subject_id=subject_id, filename=file.filename, file_type=ext, content=full_text, file_path=file_path)
    db.add(doc_record)
    try:
        llm = get_llm(temperature=0.1)
        response = llm.invoke([HumanMessage(content=f'Extract the main academic topics from this text. Return ONLY a JSON array of strings.\n\nText:\n{full_text[:12000]}')])
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        topic_names = json.loads(content.strip())
        if not isinstance(topic_names, list):
            topic_names = ['General']
    except Exception:
        topic_names = [f'Topics from {file.filename}']
    for name in set(topic_names):
        db.add(Topic(subject_id=subject_id, name=name, content=full_text))
    db.commit()
    return UploadResponse(document_id=doc_record.id, filename=file.filename, topics_extracted=list(set(topic_names)), topic_count=len(set(topic_names)))

@router.post('/session/start', response_model=SessionResponse)
def start_session(req: SessionStart, db: Session=Depends(get_db)):
    sub = db.query(Subject).filter(Subject.id == req.subject_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail='Subject not found')
    session_id = _next_session_id()
    days_until = max(0, (sub.exam_date - date.today()).days)
    topic = db.query(Topic).filter(Topic.subject_id == sub.id).order_by(Topic.mastery_score.asc()).first()
    topics_list = db.query(Topic).filter(Topic.subject_id == sub.id).limit(5).all()
    todays_topics = [t.name for t in topics_list] if topics_list else ['No topics yet']
    if not topic:
        first_action = AgentAction(action_type='quiz', topic_name='General', reasoning='No topics found. Upload study materials first.', content={'message': 'Please upload study materials to get started.'})
    else:
        try:
            llm = get_llm(temperature=0.5)
            response = llm.invoke([HumanMessage(content=f"""Generate 1 MCQ question about '{topic.name}'. Context: {(topic.content or '')[:2000]}\n\nReturn ONLY valid JSON:\n{{"question": "...", "options": ["A) ...", "B) ...", "C) ...", "D) ..."], "correct_answer": "exact text of correct option", "explanation": "..."}}""")])
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            if content.endswith('```'):
                content = content[:-3]
            q_data = json.loads(content.strip())
            first_action = AgentAction(action_type='quiz', topic_id=topic.id, topic_name=topic.name, reasoning=f'Starting with your weakest topic: {topic.name} ({topic.mastery_score:.0f}% mastery)', content={'questions': [q_data]})
        except Exception as e:
            first_action = AgentAction(action_type='quiz', topic_id=topic.id, topic_name=topic.name, reasoning=f'Starting with {topic.name}. (Question generation had an issue: {e})', content={'questions': [{'question': f'What is {topic.name}?', 'options': [], 'correct_answer': '', 'explanation': ''}]})
    active_sessions[session_id] = {'subject_id': req.subject_id, 'state': {}}
    return SessionResponse(session_id=session_id, subject_name=sub.name, days_until_exam=days_until, todays_topics=todays_topics, current_action=first_action)

@router.post('/session/{session_id}/answer')
def submit_answer(session_id: int, req: QuizAnswerRequest, db: Session=Depends(get_db)):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail='Session not found')
    session = active_sessions[session_id]
    topic_id = req.topic_id
    topic = db.query(Topic).filter(Topic.id == topic_id).first() if topic_id else None
    topic_content = (topic.content or '')[:1000] if topic else ''
    llm = get_llm(temperature=0.2)
    grade_prompt = f'Grade this answer.\nQuestion: {req.question}\nStudent Answer: {req.student_answer}\nContext: {topic_content}\n\nReturn ONLY JSON: {{"is_correct": bool, "correct_answer": "...", "explanation": "...", "misconception_detected": "..." or null}}'
    try:
        response = llm.invoke([HumanMessage(content=grade_prompt)])
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        grade = json.loads(content.strip())
    except Exception:
        grade = {'is_correct': False, 'correct_answer': 'Unable to grade', 'explanation': 'Grading error occurred', 'misconception_detected': None}
    mastery_change = 5.0 if grade.get('is_correct') else -3.0
    new_mastery = 50.0
    if topic:
        topic.mastery_score = max(0.0, min(100.0, topic.mastery_score + mastery_change))
        topic.last_reviewed = datetime.utcnow()
        topic.review_count += 1
        new_mastery = topic.mastery_score
        db.add(QuizHistory(topic_id=topic_id, question=req.question, student_answer=req.student_answer, correct_answer=grade.get('correct_answer', ''), is_correct=grade.get('is_correct', False), explanation=grade.get('explanation', '')))
        misconception = grade.get('misconception_detected')
        if misconception and (not grade.get('is_correct')):
            existing = db.query(Misconception).filter(Misconception.topic_id == topic_id, Misconception.description == misconception).first()
            if existing:
                existing.frequency += 1
            else:
                db.add(Misconception(topic_id=topic_id, description=misconception))
        db.commit()
    return {'is_correct': grade.get('is_correct', False), 'correct_answer': grade.get('correct_answer', ''), 'explanation': grade.get('explanation', ''), 'misconception_detected': grade.get('misconception_detected'), 'mastery_change': mastery_change, 'new_mastery': new_mastery}

@router.get('/session/{session_id}/next')
def get_next_action(session_id: int, db: Session=Depends(get_db)):
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail='Session not found')
    session = active_sessions[session_id]
    subject_id = session.get('subject_id', 1)
    topic = db.query(Topic).filter(Topic.subject_id == subject_id).order_by(Topic.mastery_score.asc()).first()
    if not topic:
        return {'action': {'action_type': 'move_on', 'reasoning': 'No topics available.'}, 'session_stats': {}}
    llm = get_llm(temperature=0.5)
    try:
        response = llm.invoke([HumanMessage(content=f"""Generate 1 MCQ question for '{topic.name}'. Context: {(topic.content or '')[:2000]}. Return JSON: {{"question": "...", "options": ["A","B","C","D"], "correct_answer": "...", "explanation": "..."}}""")])
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        question_data = json.loads(content.strip())
    except Exception:
        question_data = {'question': f'What is {topic.name}?', 'options': ['A', 'B', 'C', 'D'], 'correct_answer': 'A'}
    return {'action': {'action_type': 'quiz', 'topic_id': topic.id, 'topic_name': topic.name, 'reasoning': f'Quizzing on weakest topic: {topic.name} ({topic.mastery_score:.0f}% mastery)', 'content': question_data}, 'session_stats': {}}

@router.get('/subjects/{subject_id}/plan', response_model=List[StudyPlanEntryResponse])
def get_study_plan(subject_id: int, db: Session=Depends(get_db)):
    entries = db.query(StudyPlanEntry).filter(StudyPlanEntry.subject_id == subject_id).order_by(StudyPlanEntry.date).all()
    return [StudyPlanEntryResponse(id=e.id, date=e.date, topic_name=e.topic.name if e.topic else None, action_type=e.action_type, status=e.status, priority=e.priority, notes=e.notes) for e in entries]

@router.post('/subjects/{subject_id}/plan/approve')
def approve_plan(subject_id: int, approval: PlanApproval, db: Session=Depends(get_db)):
    if approval.approved:
        return {'status': 'approved', 'message': 'Plan changes applied.'}
    return {'status': 'rejected', 'message': 'Plan unchanged.'}

@router.post('/subjects/{subject_id}/plan/replan')
def force_replan(subject_id: int, db: Session=Depends(get_db)):
    sub = db.query(Subject).filter(Subject.id == subject_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail='Subject not found')
    topics = db.query(Topic).filter(Topic.subject_id == subject_id).all()
    topic_data = [{'id': t.id, 'name': t.name, 'mastery': t.mastery_score} for t in topics]
    days_left = max(1, (sub.exam_date - date.today()).days)
    llm = get_llm()
    try:
        response = llm.invoke([HumanMessage(content=f"""Create a study plan for '{sub.name}'. Days left: {days_left}. Topics: {json.dumps(topic_data)}. Prioritize low-mastery topics. Return ONLY JSON array: [{{"day_offset": 0, "topic_id": 1, "action_type": "quiz", "priority": 8, "notes": "..."}}]""")])
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        plan_data = json.loads(content.strip())
    except Exception:
        plan_data = []
    db.query(StudyPlanEntry).filter(StudyPlanEntry.subject_id == subject_id, StudyPlanEntry.date >= date.today(), StudyPlanEntry.status == PlanStatus.PENDING.value).delete()
    for item in plan_data:
        entry_date = date.today() + timedelta(days=item.get('day_offset', 0))
        db.add(StudyPlanEntry(subject_id=subject_id, topic_id=item.get('topic_id'), date=entry_date, action_type=item.get('action_type', ActionType.REVIEW.value), status=PlanStatus.PENDING.value, priority=item.get('priority', 5), notes=item.get('notes', '')))
    db.commit()
    return {'status': 'success', 'entries_created': len(plan_data)}

@router.get('/subjects/{subject_id}/mastery', response_model=List[TopicMastery])
def get_mastery(subject_id: int, db: Session=Depends(get_db)):
    topics = db.query(Topic).filter(Topic.subject_id == subject_id).all()
    result = []
    for t in topics:
        misconceptions = [m.description for m in t.misconceptions if not m.resolved]
        recent = db.query(QuizHistory).filter(QuizHistory.topic_id == t.id).order_by(QuizHistory.created_at.desc()).limit(5).all()
        correct_count = sum((1 for r in recent if r.is_correct))
        trend = 'stable'
        if len(recent) >= 3:
            trend = 'up' if correct_count >= 3 else 'down' if correct_count <= 1 else 'stable'
        result.append(TopicMastery(topic_id=t.id, topic_name=t.name, mastery_score=t.mastery_score, trend=trend, misconceptions=misconceptions, last_reviewed=t.last_reviewed))
    return result

@router.get('/subjects/{subject_id}/dashboard', response_model=DashboardData)
def get_dashboard(subject_id: int, db: Session=Depends(get_db)):
    sub = db.query(Subject).filter(Subject.id == subject_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail='Subject not found')
    topics = db.query(Topic).filter(Topic.subject_id == sub.id).all()
    tc = len(topics)
    avg = sum((t.mastery_score for t in topics)) / tc if tc > 0 else 0.0
    days = max(0, (sub.exam_date - date.today()).days)
    mastery_data = [TopicMastery(topic_id=t.id, topic_name=t.name, mastery_score=t.mastery_score, trend='stable', last_reviewed=t.last_reviewed, misconceptions=[m.description for m in t.misconceptions if not m.resolved]) for t in topics]
    week_ago = datetime.utcnow() - timedelta(days=7)
    recent_quizzes = db.query(QuizHistory).join(Topic).filter(Topic.subject_id == sub.id, QuizHistory.created_at >= week_ago).all()
    total_q = len(recent_quizzes)
    correct_q = sum((1 for q in recent_quizzes if q.is_correct))
    plan_entries = db.query(StudyPlanEntry).filter(StudyPlanEntry.subject_id == sub.id).all()
    plan_stats = {'completed': 0, 'missed': 0, 'pending': 0}
    for e in plan_entries:
        if e.status == PlanStatus.COMPLETED.value:
            plan_stats['completed'] += 1
        elif e.status == PlanStatus.MISSED.value:
            plan_stats['missed'] += 1
        else:
            plan_stats['pending'] += 1
    return DashboardData(subject=SubjectResponse(id=sub.id, name=sub.name, exam_date=sub.exam_date, topic_count=tc, avg_mastery=avg, days_until_exam=days), mastery_data=mastery_data, recent_quiz_stats={'correct': correct_q, 'total': total_q, 'accuracy': round(correct_q / total_q * 100, 1) if total_q > 0 else 0}, plan_completion=plan_stats, streak=0, mock_available=days <= 7)

@router.post('/subjects/{subject_id}/mock')
def generate_mock(subject_id: int, db: Session=Depends(get_db)):
    sub = db.query(Subject).filter(Subject.id == subject_id).first()
    if not sub:
        raise HTTPException(status_code=404, detail='Subject not found')
    topics = db.query(Topic).filter(Topic.subject_id == subject_id).order_by(Topic.mastery_score.asc()).all()
    topic_info = [{'name': t.name, 'mastery': t.mastery_score} for t in topics]
    llm = get_llm(temperature=0.5)
    try:
        response = llm.invoke([HumanMessage(content=f'''Generate a 15-question mock exam for '{sub.name}'. Weight questions toward weaker topics. Mix easy/medium/hard. Topics (sorted weakest first): {json.dumps(topic_info)}. Return ONLY JSON: {{"title": "...", "subject": "{sub.name}", "total_marks": N, "duration_minutes": N, "questions": [{{"number": 1, "question": "...", "question_type": "mcq", "options": ["A","B","C","D"], "marks": 2, "topic_name": "...", "difficulty": "medium"}}], "focus_areas": ["..."]}}''')])
        content = response.content.strip()
        if content.startswith('```json'):
            content = content[7:]
        if content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        return json.loads(content.strip())
    except Exception as e:
        return {'title': f'{sub.name} Mock Paper', 'subject': sub.name, 'total_marks': 0, 'duration_minutes': 60, 'questions': [], 'focus_areas': [], 'error': str(e)}