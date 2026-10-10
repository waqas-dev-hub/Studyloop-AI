import os
import json
import asyncio
from datetime import datetime, date, timedelta
from typing import Any
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from langchain_groq import ChatGroq
from database import get_db_session, Subject, Topic, Misconception, QuizHistory, StudyPlanEntry, UploadedDocument, PlanStatus, ActionType
load_dotenv()
server = Server('adaptive-study-planner')
MODEL_NAME = os.getenv('MODEL_NAME', 'meta-llama/llama-4-scout-17b-16e-instruct')

def get_llm(temperature: float=0.5) -> ChatGroq:
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        raise RuntimeError('GROQ_API_KEY is not set. Add it to your .env file.')
    return ChatGroq(model=MODEL_NAME, temperature=temperature, max_tokens=4096)

def parse_json_response(text: str) -> Any:
    text = text.strip()
    if text.startswith('```json'):
        text = text[7:]
    elif text.startswith('```'):
        text = text[3:]
    if text.endswith('```'):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {'error': 'Failed to parse LLM response as JSON', 'raw': text[:500]}

@server.tool()
async def search_notes(query: str, subject_id: int) -> list[dict]:
    db = get_db_session()
    try:
        results = []
        query_lower = query.lower()
        documents = db.query(UploadedDocument).filter(UploadedDocument.subject_id == subject_id).all()
        for doc in documents:
            if doc.content and query_lower in doc.content.lower():
                idx = doc.content.lower().find(query_lower)
                start = max(0, idx - 150)
                end = min(len(doc.content), idx + len(query) + 150)
                results.append({'source': 'document', 'filename': doc.filename, 'snippet': '...' + doc.content[start:end] + '...'})
        topics = db.query(Topic).filter(Topic.subject_id == subject_id).all()
        for topic in topics:
            if topic.content and query_lower in topic.content.lower():
                idx = topic.content.lower().find(query_lower)
                start = max(0, idx - 150)
                end = min(len(topic.content), idx + len(query) + 150)
                results.append({'source': 'topic', 'topic_name': topic.name, 'topic_id': topic.id, 'snippet': '...' + topic.content[start:end] + '...'})
        return results if results else [{'message': f"No results found for '{query}'"}]
    finally:
        db.close()

@server.tool()
async def generate_quiz(topic_id: int, difficulty: str='medium', num_questions: int=5, question_type: str='mcq') -> list[dict]:
    db = get_db_session()
    try:
        topic = db.query(Topic).filter(Topic.id == topic_id).first()
        if not topic:
            return [{'error': f'Topic ID {topic_id} not found'}]
        llm = get_llm()
        prompt = f'''Generate {num_questions} {difficulty}-difficulty {question_type} questions.\n\nTopic: {topic.name}\nContext: {(topic.content or 'Use your general knowledge')[:3000]}\n\nReturn ONLY a JSON array:\n[\n  {{\n    "question": "...",\n    "question_type": "{question_type}",\n    "options": ["A", "B", "C", "D"],\n    "correct_answer": "exact correct option text",\n    "explanation": "why this is correct"\n  }}\n]'''
        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception as e:
        return [{'error': str(e)}]
    finally:
        db.close()

@server.tool()
async def grade_answer(question: str, correct_answer: str, student_answer: str, topic_content: str='') -> dict:
    llm = get_llm(temperature=0.2)
    prompt = f'Grade this answer carefully.\n\nQuestion: {question}\nCorrect Answer: {correct_answer}\nStudent Answer: {student_answer}\nContext: {topic_content[:1000]}\n\nReturn ONLY a JSON object:\n{{\n  "is_correct": true/false,\n  "explanation": "educational explanation of why the answer is right/wrong",\n  "misconception_detected": "description of conceptual misunderstanding, or null"\n}}'
    try:
        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception as e:
        return {'error': str(e)}

@server.tool()
async def get_mastery_data(subject_id: int) -> dict:
    db = get_db_session()
    try:
        subject = db.query(Subject).filter(Subject.id == subject_id).first()
        if not subject:
            return {'error': 'Subject not found'}
        topics_data = []
        for topic in subject.topics:
            active_misconceptions = [m.description for m in topic.misconceptions if not m.resolved]
            topics_data.append({'topic_id': topic.id, 'name': topic.name, 'mastery_score': topic.mastery_score, 'difficulty': topic.difficulty, 'last_reviewed': topic.last_reviewed.isoformat() if topic.last_reviewed else None, 'review_count': topic.review_count, 'active_misconceptions': active_misconceptions})
        return {'subject_id': subject.id, 'subject_name': subject.name, 'exam_date': subject.exam_date.isoformat(), 'days_until_exam': max(0, (subject.exam_date - date.today()).days), 'topics': topics_data}
    finally:
        db.close()

@server.tool()
async def update_mastery(topic_id: int, score_delta: float, misconception: str=None) -> dict:
    db = get_db_session()
    try:
        topic = db.query(Topic).filter(Topic.id == topic_id).first()
        if not topic:
            return {'error': 'Topic not found'}
        old_score = topic.mastery_score
        topic.mastery_score = max(0.0, min(100.0, topic.mastery_score + score_delta))
        topic.last_reviewed = datetime.utcnow()
        topic.review_count += 1
        if misconception:
            existing = db.query(Misconception).filter(Misconception.topic_id == topic_id, Misconception.description == misconception).first()
            if existing:
                existing.frequency += 1
                existing.resolved = False
            else:
                db.add(Misconception(topic_id=topic_id, description=misconception))
        db.commit()
        return {'success': True, 'topic_id': topic_id, 'old_score': old_score, 'new_score': topic.mastery_score, 'delta': score_delta, 'misconception_logged': bool(misconception)}
    except Exception as e:
        db.rollback()
        return {'error': str(e)}
    finally:
        db.close()

@server.tool()
async def manage_study_plan(subject_id: int, action: str='get') -> dict:
    db = get_db_session()
    try:
        if action == 'get':
            entries = db.query(StudyPlanEntry).filter(StudyPlanEntry.subject_id == subject_id).order_by(StudyPlanEntry.date).all()
            return {'entries': [{'id': e.id, 'date': e.date.isoformat(), 'topic_id': e.topic_id, 'topic_name': e.topic.name if e.topic else None, 'action_type': e.action_type, 'status': e.status, 'priority': e.priority, 'notes': e.notes} for e in entries]}
        elif action == 'mark_complete':
            today = date.today()
            entries = db.query(StudyPlanEntry).filter(StudyPlanEntry.subject_id == subject_id, StudyPlanEntry.date == today, StudyPlanEntry.status == PlanStatus.PENDING.value).all()
            for e in entries:
                e.status = PlanStatus.COMPLETED.value
            db.commit()
            return {'success': True, 'marked_count': len(entries)}
        elif action == 'mark_missed':
            today = date.today()
            count = db.query(StudyPlanEntry).filter(StudyPlanEntry.subject_id == subject_id, StudyPlanEntry.date < today, StudyPlanEntry.status == PlanStatus.PENDING.value).update({'status': PlanStatus.MISSED.value})
            db.commit()
            return {'success': True, 'marked_count': count}
        elif action == 'replan':
            subject = db.query(Subject).filter(Subject.id == subject_id).first()
            if not subject:
                return {'error': 'Subject not found'}
            topics = db.query(Topic).filter(Topic.subject_id == subject_id).all()
            topic_data = [{'id': t.id, 'name': t.name, 'mastery': t.mastery_score} for t in topics]
            days_left = max(1, (subject.exam_date - date.today()).days)
            llm = get_llm()
            prompt = f"""Create a study plan for '{subject.name}'.\nDays left: {days_left}\nTopics: {json.dumps(topic_data)}\n\nPrioritize low-mastery topics. Return ONLY a JSON array:\n[{{"day_offset": 0, "topic_id": 1, "action_type": "quiz", "priority": 8, "notes": "..."}}]"""
            response = llm.invoke(prompt)
            plan_data = parse_json_response(response.content)
            if isinstance(plan_data, dict) and 'error' in plan_data:
                return plan_data
            db.query(StudyPlanEntry).filter(StudyPlanEntry.subject_id == subject_id, StudyPlanEntry.date >= date.today(), StudyPlanEntry.status == PlanStatus.PENDING.value).delete()
            for item in plan_data:
                entry_date = date.today() + timedelta(days=item.get('day_offset', 0))
                db.add(StudyPlanEntry(subject_id=subject_id, topic_id=item.get('topic_id'), date=entry_date, action_type=item.get('action_type', ActionType.REVIEW.value), status=PlanStatus.PENDING.value, priority=item.get('priority', 5), notes=item.get('notes', '')))
            db.commit()
            return {'success': True, 'replanned': True, 'entries_created': len(plan_data)}
        else:
            return {'error': f'Unknown action: {action}'}
    except Exception as e:
        db.rollback()
        return {'error': str(e)}
    finally:
        db.close()

@server.tool()
async def generate_mock_paper(subject_id: int, total_questions: int=20) -> dict:
    db = get_db_session()
    try:
        subject = db.query(Subject).filter(Subject.id == subject_id).first()
        if not subject:
            return {'error': 'Subject not found'}
        topics = db.query(Topic).filter(Topic.subject_id == subject_id).order_by(Topic.mastery_score.asc()).all()
        topic_info = [{'id': t.id, 'name': t.name, 'mastery': t.mastery_score} for t in topics]
        llm = get_llm()
        prompt = f'''Generate a {total_questions}-question mock exam for '{subject.name}'.\nWeight questions toward weaker topics. Mix easy/medium/hard difficulty.\n\nTopics (sorted weakest first):\n{json.dumps(topic_info)}\n\nReturn ONLY a JSON object:\n{{\n  "title": "Mock Exam — {subject.name}",\n  "subject": "{subject.name}",\n  "total_marks": <sum>,\n  "duration_minutes": <int>,\n  "questions": [\n    {{"number": 1, "question": "...", "question_type": "mcq",\n      "options": ["A","B","C","D"], "correct_answer": "...",\n      "marks": 2, "topic_name": "...", "difficulty": "medium"}}\n  ],\n  "focus_areas": ["weak topic 1", "weak topic 2"]\n}}'''
        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception as e:
        return {'error': str(e)}
    finally:
        db.close()

@server.tool()
async def create_explanation(topic_id: int, misconception: str, context: str='') -> dict:
    db = get_db_session()
    try:
        topic = db.query(Topic).filter(Topic.id == topic_id).first()
        if not topic:
            return {'error': 'Topic not found'}
        llm = get_llm()
        prompt = f'Create a mini-lesson to fix this misconception.\n\nTopic: {topic.name}\nContext: {(topic.content or context)[:2000]}\nMisconception: {misconception}\n\nUse a clear analogy and end with a verification question.\n\nReturn ONLY a JSON object:\n{{\n  "title": "...",\n  "misconception_stated": "what the student wrongly believes",\n  "correct_understanding": "the correct concept",\n  "explanation": "clear explanation with analogy",\n  "verification_question": "simple check question",\n  "verification_answer": "expected answer"\n}}'
        response = llm.invoke(prompt)
        return parse_json_response(response.content)
    except Exception as e:
        return {'error': str(e)}
    finally:
        db.close()

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
if __name__ == '__main__':
    asyncio.run(main())