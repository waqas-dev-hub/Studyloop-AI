from datetime import datetime, date
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, Text, Boolean, Date, DateTime, ForeignKey, Enum as SAEnum
from sqlalchemy.orm import DeclarativeBase, relationship, sessionmaker, Session
import enum
import os
from dotenv import load_dotenv
load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///./study_planner.db')
engine = create_engine(DATABASE_URL, connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class PlanStatus(str, enum.Enum):
    PENDING = 'pending'
    COMPLETED = 'completed'
    MISSED = 'missed'
    SKIPPED = 'skipped'

class ActionType(str, enum.Enum):
    QUIZ = 'quiz'
    EXPLAIN = 'explain'
    WORKED_EXAMPLE = 'worked_example'
    EASIER_QUESTION = 'easier_question'
    EXPLAIN_BACK = 'explain_back'
    REVIEW = 'review'
    MOCK_PAPER = 'mock_paper'

class Difficulty(str, enum.Enum):
    EASY = 'easy'
    MEDIUM = 'medium'
    HARD = 'hard'
    APPLICATION = 'application'

class Base(DeclarativeBase):
    pass

class Subject(Base):
    __tablename__ = 'subjects'
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    exam_date = Column(Date, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    topics = relationship('Topic', back_populates='subject', cascade='all, delete-orphan')
    plan_entries = relationship('StudyPlanEntry', back_populates='subject', cascade='all, delete-orphan')
    documents = relationship('UploadedDocument', back_populates='subject', cascade='all, delete-orphan')

class Topic(Base):
    __tablename__ = 'topics'
    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, ForeignKey('subjects.id'), nullable=False)
    name = Column(String(255), nullable=False)
    content = Column(Text, nullable=True)
    mastery_score = Column(Float, default=0.0)
    difficulty = Column(String(20), default=Difficulty.MEDIUM.value)
    last_reviewed = Column(DateTime, nullable=True)
    review_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    subject = relationship('Subject', back_populates='topics')
    misconceptions = relationship('Misconception', back_populates='topic', cascade='all, delete-orphan')
    quiz_history = relationship('QuizHistory', back_populates='topic', cascade='all, delete-orphan')
    plan_entries = relationship('StudyPlanEntry', back_populates='topic')

class Misconception(Base):
    __tablename__ = 'misconceptions'
    id = Column(Integer, primary_key=True, index=True)
    topic_id = Column(Integer, ForeignKey('topics.id'), nullable=False)
    description = Column(Text, nullable=False)
    frequency = Column(Integer, default=1)
    detected_at = Column(DateTime, default=datetime.utcnow)
    resolved = Column(Boolean, default=False)
    topic = relationship('Topic', back_populates='misconceptions')

class QuizHistory(Base):
    __tablename__ = 'quiz_history'
    id = Column(Integer, primary_key=True, index=True)
    topic_id = Column(Integer, ForeignKey('topics.id'), nullable=False)
    question = Column(Text, nullable=False)
    question_type = Column(String(20), default='mcq')
    student_answer = Column(Text, nullable=True)
    correct_answer = Column(Text, nullable=False)
    is_correct = Column(Boolean, nullable=True)
    explanation = Column(Text, nullable=True)
    difficulty = Column(String(20), default=Difficulty.MEDIUM.value)
    created_at = Column(DateTime, default=datetime.utcnow)
    topic = relationship('Topic', back_populates='quiz_history')

class StudyPlanEntry(Base):
    __tablename__ = 'study_plan_entries'
    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, ForeignKey('subjects.id'), nullable=False)
    topic_id = Column(Integer, ForeignKey('topics.id'), nullable=True)
    date = Column(Date, nullable=False)
    action_type = Column(String(30), default=ActionType.QUIZ.value)
    status = Column(String(20), default=PlanStatus.PENDING.value)
    priority = Column(Integer, default=5)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    subject = relationship('Subject', back_populates='plan_entries')
    topic = relationship('Topic', back_populates='plan_entries')

class UploadedDocument(Base):
    __tablename__ = 'uploaded_documents'
    id = Column(Integer, primary_key=True, index=True)
    subject_id = Column(Integer, ForeignKey('subjects.id'), nullable=False)
    filename = Column(String(500), nullable=False)
    file_type = Column(String(20), nullable=False)
    content = Column(Text, nullable=True)
    file_path = Column(String(1000), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    subject = relationship('Subject', back_populates='documents')

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_session() -> Session:
    return SessionLocal()