import json
import os
from datetime import datetime, date, timedelta
from typing import Any
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from langsmith import traceable
from .state import AgentState
from .prompts import DIAGNOSE_PROMPT, CHOOSE_ACTION_PROMPT, GENERATE_QUIZ_PROMPT, GRADE_ANSWER_PROMPT, CREATE_EXPLANATION_PROMPT, WORKED_EXAMPLE_PROMPT, CRITIC_PROMPT, REPLAN_PROMPT, GENERATE_MOCK_PROMPT
from database import get_db_session, Subject, Topic, Misconception, QuizHistory, StudyPlanEntry, UploadedDocument, PlanStatus, ActionType
load_dotenv()
MODEL_NAME = os.getenv('MODEL_NAME', 'meta-llama/llama-4-scout-17b-16e-instruct')

def get_llm(temperature: float=0.3) -> ChatGroq:
    return ChatGroq(model=MODEL_NAME, temperature=temperature, max_tokens=4096)

def parse_json_response(text: str) -> Any:
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:]
    elif text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    return json.loads(text.strip())

@traceable(name='observe_state', run_type='chain')
def observe_state(state: AgentState) -> dict:
    db = get_db_session()
    try:
        subject_id = state.get('subject_id', 1)
        subject = db.query(Subject).filter(Subject.id == subject_id).first()
        if not subject:
            return {'mastery_data': {}, 'quiz_history': [], 'misconceptions': [], 'days_until_exam': 0, 'missed_sessions': 0}
        days_until_exam = max(0, (subject.exam_date - date.today()).days)
        topics = db.query(Topic).filter(Topic.subject_id == subject_id).all()
        mastery_data = {}
        for t in topics:
            mastery_data[t.name] = {'id': t.id, 'score': t.mastery_score, 'last_reviewed': t.last_reviewed.isoformat() if t.last_reviewed else None, 'review_count': t.review_count}
        misconceptions = db.query(Misconception).join(Topic).filter(Topic.subject_id == subject_id, Misconception.resolved == False).all()
        misconception_list = [{'topic_id': m.topic_id, 'topic': m.topic.name, 'description': m.description, 'frequency': m.frequency} for m in misconceptions]
        history = db.query(QuizHistory).join(Topic).filter(Topic.subject_id == subject_id).order_by(QuizHistory.created_at.desc()).limit(15).all()
        quiz_history = [{'topic': h.topic.name, 'topic_id': h.topic_id, 'question': h.question[:100], 'is_correct': h.is_correct, 'difficulty': h.difficulty, 'created_at': h.created_at.isoformat() if h.created_at else None} for h in history]
        missed = db.query(StudyPlanEntry).filter(StudyPlanEntry.subject_id == subject_id, StudyPlanEntry.date < date.today(), StudyPlanEntry.status == PlanStatus.PENDING.value).count()
        return {'mastery_data': mastery_data, 'quiz_history': quiz_history, 'misconceptions': misconception_list, 'days_until_exam': days_until_exam, 'missed_sessions': missed}
    finally:
        db.close()

@traceable(name='diagnose_weaknesses', run_type='chain')
def diagnose_weaknesses(state: AgentState) -> dict:
    llm = get_llm()
    prompt = DIAGNOSE_PROMPT.format(mastery_data=json.dumps(state.get('mastery_data', {}), indent=2), quiz_history=json.dumps(state.get('quiz_history', []), indent=2), misconceptions=json.dumps(state.get('misconceptions', []), indent=2), days_until_exam=state.get('days_until_exam', 0))
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        diagnosis = parse_json_response(response.content)
        return {'diagnosis': diagnosis}
    except Exception as e:
        return {'diagnosis': {'weak_topics': [], 'misconceptions': [], 'knowledge_gaps': [], 'overall_assessment': f'Diagnosis skipped: {str(e)}', 'recommended_focus': 'Continue with general review'}}

@traceable(name='choose_teaching_action', run_type='chain')
def choose_teaching_action(state: AgentState) -> dict:
    mastery = state.get('mastery_data', {})
    topic_list = {}
    for name, data in mastery.items():
        if isinstance(data, dict):
            topic_list[data.get('id', 0)] = name
        else:
            topic_list[0] = name
    llm = get_llm()
    prompt = CHOOSE_ACTION_PROMPT.format(diagnosis=json.dumps(state.get('diagnosis', {}), indent=2), mastery_data=json.dumps(mastery, indent=2), days_until_exam=state.get('days_until_exam', 30), questions_asked=state.get('questions_asked', 0), correct_answers=state.get('correct_answers', 0), topic_list=json.dumps(topic_list))
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        action = parse_json_response(response.content)
        return {'chosen_action': action, 'action_reasoning': action.get('reasoning', '')}
    except Exception as e:
        weakest = min(mastery.items(), key=lambda x: x[1].get('score', 0) if isinstance(x[1], dict) else 0, default=('General', {'id': None, 'score': 0}))
        return {'chosen_action': {'action_type': 'quiz', 'topic_id': weakest[1].get('id') if isinstance(weakest[1], dict) else None, 'topic_name': weakest[0], 'difficulty': 'medium', 'reasoning': f'Fallback: quizzing weakest topic due to error: {e}'}, 'action_reasoning': f'Fallback due to error: {e}'}

@traceable(name='execute_action', run_type='chain')
def execute_action(state: AgentState) -> dict:
    action = state.get('chosen_action', {})
    action_type = action.get('action_type', 'quiz')
    topic_id = action.get('topic_id')
    topic_name = action.get('topic_name', 'General')
    difficulty = action.get('difficulty', 'medium')
    db = get_db_session()
    llm = get_llm(temperature=0.5)
    try:
        topic_content = ''
        if topic_id:
            topic = db.query(Topic).filter(Topic.id == topic_id).first()
            if topic:
                topic_content = (topic.content or '')[:3000]
                topic_name = topic.name
        if action_type in ('quiz', 'easier_question'):
            if action_type == 'easier_question':
                difficulty = 'easy'
            prompt = GENERATE_QUIZ_PROMPT.format(num_questions=3, topic_name=topic_name, difficulty=difficulty, topic_content=topic_content or 'Use your knowledge of this topic.')
            response = llm.invoke([HumanMessage(content=prompt)])
            questions = parse_json_response(response.content)
            return {'action_result': {'type': action_type, 'topic_id': topic_id, 'topic_name': topic_name, 'questions': questions if isinstance(questions, list) else [questions], 'difficulty': difficulty}}
        elif action_type == 'explain':
            misconceptions = state.get('misconceptions', [])
            misconception_text = 'General understanding gap'
            for m in misconceptions:
                if m.get('topic_id') == topic_id or m.get('topic') == topic_name:
                    misconception_text = m.get('description', misconception_text)
                    break
            prompt = CREATE_EXPLANATION_PROMPT.format(topic_name=topic_name, misconception=misconception_text, topic_content=topic_content or 'Use your knowledge.')
            response = llm.invoke([HumanMessage(content=prompt)])
            explanation = parse_json_response(response.content)
            return {'action_result': {'type': 'explain', 'topic_id': topic_id, 'topic_name': topic_name, 'content': explanation}}
        elif action_type == 'worked_example':
            prompt = WORKED_EXAMPLE_PROMPT.format(topic_name=topic_name, difficulty=difficulty, topic_content=topic_content or 'Use your knowledge.')
            response = llm.invoke([HumanMessage(content=prompt)])
            example = parse_json_response(response.content)
            return {'action_result': {'type': 'worked_example', 'topic_id': topic_id, 'topic_name': topic_name, 'content': example}}
        elif action_type == 'explain_back':
            return {'action_result': {'type': 'explain_back', 'topic_id': topic_id, 'topic_name': topic_name, 'content': {'prompt': f"In your own words, explain the key concept of '{topic_name}'. What is it, why does it matter, and give an example.", 'evaluation_criteria': 'Checks for correct terminology, logical structure, and a valid example.'}}}
        elif action_type == 'replan':
            mastery = state.get('mastery_data', {})
            topic_mastery = []
            for name, data in mastery.items():
                if isinstance(data, dict):
                    topic_mastery.append({'id': data.get('id'), 'name': name, 'mastery': data.get('score', 0)})
            subject = db.query(Subject).filter(Subject.id == state.get('subject_id', 1)).first()
            prompt = REPLAN_PROMPT.format(subject_name=subject.name if subject else 'Unknown', days_until_exam=state.get('days_until_exam', 30), missed_days=state.get('missed_sessions', 0), topic_mastery=json.dumps(topic_mastery, indent=2))
            response = llm.invoke([HumanMessage(content=prompt)])
            plan = parse_json_response(response.content)
            return {'action_result': {'type': 'replan', 'content': plan}, 'plan_proposal': plan, 'needs_human_approval': True}
        elif action_type == 'mock_paper':
            mastery = state.get('mastery_data', {})
            topic_mastery = []
            for name, data in mastery.items():
                if isinstance(data, dict):
                    topic_mastery.append({'name': name, 'mastery': data.get('score', 0)})
            subject = db.query(Subject).filter(Subject.id == state.get('subject_id', 1)).first()
            prompt = GENERATE_MOCK_PROMPT.format(subject_name=subject.name if subject else 'Unknown', total_questions=20, topic_mastery=json.dumps(topic_mastery, indent=2))
            response = llm.invoke([HumanMessage(content=prompt)])
            paper = parse_json_response(response.content)
            return {'action_result': {'type': 'mock_paper', 'content': paper}, 'needs_human_approval': True}
        else:
            return {'action_result': {'type': 'move_on', 'message': 'Moving to the next topic.'}}
    except Exception as e:
        return {'action_result': {'type': 'error', 'error': str(e), 'fallback': 'The agent encountered an error executing the action.'}}
    finally:
        db.close()

@traceable(name='critic_verify', run_type='chain')
def critic_verify(state: AgentState) -> dict:
    action_result = state.get('action_result', {})
    result_type = action_result.get('type', '')
    if result_type in ('move_on', 'explain_back', 'replan', 'error'):
        return {'critic_passed': True, 'critic_feedback': 'Skipped — no content to verify.'}
    db = get_db_session()
    try:
        topic_id = action_result.get('topic_id')
        source_notes = ''
        if topic_id:
            topic = db.query(Topic).filter(Topic.id == topic_id).first()
            if topic and topic.content:
                source_notes = topic.content[:2000]
        if not source_notes:
            return {'critic_passed': True, 'critic_feedback': 'No source notes to verify against — passed by default.'}
        llm = get_llm(temperature=0.1)
        prompt = CRITIC_PROMPT.format(content=json.dumps(action_result, indent=2)[:2000], source_notes=source_notes)
        response = llm.invoke([HumanMessage(content=prompt)])
        critic_result = parse_json_response(response.content)
        return {'critic_passed': critic_result.get('passed', True), 'critic_feedback': critic_result.get('feedback', '')}
    except Exception as e:
        return {'critic_passed': True, 'critic_feedback': f'Verification skipped due to error: {e}'}
    finally:
        db.close()

def human_approval_check(state: AgentState) -> dict:
    action_type = state.get('chosen_action', {}).get('action_type', '')
    needs_approval = action_type in ('replan', 'mock_paper')
    return {'needs_human_approval': needs_approval}

@traceable(name='update_state', run_type='chain')
def update_state_node(state: AgentState) -> dict:
    db = get_db_session()
    try:
        action_result = state.get('action_result', {})
        result_type = action_result.get('type', '')
        subject_id = state.get('subject_id', 1)
        questions_asked = state.get('questions_asked', 0)
        correct_answers = state.get('correct_answers', 0)
        if result_type in ('quiz', 'easier_question') and 'graded' in action_result:
            graded = action_result['graded']
            topic_id = action_result.get('topic_id')
            if topic_id and graded:
                topic = db.query(Topic).filter(Topic.id == topic_id).first()
                if topic:
                    for g in graded:
                        is_correct = g.get('is_correct', False)
                        delta = 5.0 if is_correct else -3.0
                        topic.mastery_score = max(0.0, min(100.0, topic.mastery_score + delta))
                        topic.last_reviewed = datetime.utcnow()
                        topic.review_count += 1
                        db.add(QuizHistory(topic_id=topic_id, question=g.get('question', ''), question_type=g.get('question_type', 'mcq'), student_answer=g.get('student_answer', ''), correct_answer=g.get('correct_answer', ''), is_correct=is_correct, explanation=g.get('explanation', ''), difficulty=action_result.get('difficulty', 'medium')))
                        misconception = g.get('misconception_detected')
                        if misconception and (not is_correct):
                            existing = db.query(Misconception).filter(Misconception.topic_id == topic_id, Misconception.description == misconception).first()
                            if existing:
                                existing.frequency += 1
                                existing.resolved = False
                            else:
                                db.add(Misconception(topic_id=topic_id, description=misconception))
                        questions_asked += 1
                        if is_correct:
                            correct_answers += 1
        elif result_type == 'replan' and state.get('human_approved', False):
            plan = action_result.get('content', {})
            entries = plan.get('entries', [])
            db.query(StudyPlanEntry).filter(StudyPlanEntry.subject_id == subject_id, StudyPlanEntry.date >= date.today(), StudyPlanEntry.status == PlanStatus.PENDING.value).delete()
            for item in entries:
                entry_date = date.today() + timedelta(days=item.get('day_offset', 0))
                db.add(StudyPlanEntry(subject_id=subject_id, topic_id=item.get('topic_id'), date=entry_date, action_type=item.get('action_type', ActionType.REVIEW.value), status=PlanStatus.PENDING.value, priority=item.get('priority', 5), notes=item.get('notes', '')))
        db.query(StudyPlanEntry).filter(StudyPlanEntry.subject_id == subject_id, StudyPlanEntry.date < date.today(), StudyPlanEntry.status == PlanStatus.PENDING.value).update({'status': PlanStatus.MISSED.value})
        db.commit()
        session_complete = questions_asked >= 10
        return {'questions_asked': questions_asked, 'correct_answers': correct_answers, 'session_complete': session_complete}
    except Exception as e:
        db.rollback()
        return {'questions_asked': state.get('questions_asked', 0), 'correct_answers': state.get('correct_answers', 0), 'session_complete': False}
    finally:
        db.close()

def should_continue(state: AgentState) -> str:
    if state.get('session_complete', False):
        return 'end'
    if state.get('questions_asked', 0) >= 10:
        return 'end'
    return 'continue'

def should_retry(state: AgentState) -> str:
    if not state.get('critic_passed', True):
        return 'retry'
    return 'proceed'

def needs_approval(state: AgentState) -> str:
    if state.get('needs_human_approval', False):
        return 'needs_approval'
    return 'skip_approval'