import streamlit as st
import httpx
import pandas as pd
import plotly.express as px
from datetime import date
st.set_page_config(page_title='StudyLoop', layout='wide')
API_BASE = 'http://localhost:8000/api'
if 'subject_id' not in st.session_state:
    st.session_state.subject_id = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'current_action' not in st.session_state:
    st.session_state.current_action = None
if 'mock_paper' not in st.session_state:
    st.session_state.mock_paper = None

def api_get(endpoint):
    try:
        r = httpx.get(f'{API_BASE}{endpoint}', timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f'Failed to fetch data: {e}')
        return None

def api_post(endpoint, data=None, files=None):
    try:
        if files:
            r = httpx.post(f'{API_BASE}{endpoint}', files=files, timeout=120)
        else:
            r = httpx.post(f'{API_BASE}{endpoint}', json=data, timeout=120)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f'Action failed: {e}')
        return None

def api_delete(endpoint):
    try:
        r = httpx.delete(f'{API_BASE}{endpoint}', timeout=30)
        r.raise_for_status()
        return True
    except Exception as e:
        st.error(f'Delete failed: {e}')
        return False
st.sidebar.title('StudyLoop')
pages = ['Subjects', 'Study Session', 'Dashboard', 'Study Plan', 'Mock Exam']
current_page = st.sidebar.radio('Menu', pages)
st.sidebar.divider()
subjects = api_get('/subjects') or []
if subjects:
    subject_map = {s['id']: s['name'] for s in subjects}
    _keys = list(subject_map.keys())
    _default_idx = _keys.index(st.session_state.subject_id) if st.session_state.subject_id in _keys else 0
    selected_id = st.sidebar.selectbox('Current Subject', options=_keys, format_func=lambda x: subject_map[x], index=_default_idx)
    st.session_state.subject_id = selected_id
else:
    st.sidebar.write('No subjects yet. Create one in Subjects.')
    st.session_state.subject_id = None

def render_subjects():
    st.header('Subjects & Materials')
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader('Your Subjects')
        if not subjects:
            st.info("You don't have any subjects. Create one to get started.")
        for sub in subjects:
            with st.container(border=True):
                st.write(f"**{sub['name']}** (Exam: {sub['exam_date']})")
                st.caption(f"{sub['topic_count']} topics • {sub['days_until_exam']} days left")
                if st.button('Delete', key=f"del_{sub['id']}"):
                    api_delete(f"/subjects/{sub['id']}")
                    st.rerun()
    with col2:
        st.subheader('Add New Subject')
        with st.form('add_subject_form', clear_on_submit=True):
            name = st.text_input('Name (e.g. Biology 101)')
            exam_date = st.date_input('Exam Date', min_value=date.today())
            if st.form_submit_button('Create Subject'):
                if name:
                    api_post('/subjects', {'name': name, 'exam_date': str(exam_date)})
                    st.rerun()
        if st.session_state.subject_id:
            st.subheader('Upload Notes')
            st.write(f'Add materials for **{subject_map[st.session_state.subject_id]}**')
            uploaded_files = st.file_uploader('PDF or TXT files', type=['pdf', 'txt'], accept_multiple_files=True)
            if st.button('Upload & Extract Topics'):
                for f in uploaded_files:
                    with st.spinner(f'Extracting topics from {f.name}...'):
                        res = api_post(f'/subjects/{st.session_state.subject_id}/upload', files={'file': (f.name, f.getvalue(), f.type)})
                        if res:
                            st.success(f"Extracted {res['topic_count']} topics from {f.name}")

def render_study_session():
    st.header('Study Session')
    if not st.session_state.subject_id:
        st.warning('Please select or create a subject first.')
        return
    if not st.session_state.session_id:
        if st.button('Start New Session'):
            with st.spinner('Analyzing your progress...'):
                res = api_post('/session/start', {'subject_id': st.session_state.subject_id})
                if res:
                    st.session_state.session_id = res['session_id']
                    st.session_state.current_action = res.get('current_action')
                    st.rerun()
        return
    elif st.button('End Session'):
        st.session_state.session_id = None
        st.session_state.current_action = None
        st.rerun()
    action = st.session_state.current_action
    if not action:
        st.info('Loading next step...')
        return
    action_type = action.get('action_type', 'quiz')
    content = action.get('content', {})
    topic = action.get('topic_name', 'General')
    st.subheader(f"{action_type.replace('_', ' ').title()} - {topic}")
    with st.expander('Agent Reasoning (Why this topic?)'):
        st.write(action.get('reasoning', 'No reasoning provided.'))
    if action_type in ['quiz', 'easier_question']:
        q_data = content.get('questions', [content])[0] if isinstance(content, dict) else content
        question = q_data.get('question', str(q_data))
        options = q_data.get('options', [])
        st.write(f'**{question}**')
        with st.form('quiz_form'):
            answer = st.radio('Select an answer:', options) if options else st.text_input('Your answer:')
            submitted = st.form_submit_button('Submit')
            if submitted:
                with st.spinner('Grading...'):
                    res = api_post(f'/session/{st.session_state.session_id}/answer', {'session_id': st.session_state.session_id, 'question': question, 'student_answer': answer or '', 'topic_id': action.get('topic_id', 0)})
                    if res:
                        if res.get('is_correct'):
                            st.success('Correct!')
                        else:
                            st.error(f"Incorrect. The right answer is {res.get('correct_answer')}")
                        st.write(f"**Explanation:** {res.get('explanation')}")
                        if res.get('misconception_detected'):
                            st.warning(f"Note: {res['misconception_detected']}")
        if st.button('Next'):
            with st.spinner('Deciding next move...'):
                next_res = api_get(f'/session/{st.session_state.session_id}/next')
                st.session_state.current_action = next_res.get('action') if next_res else None
                st.rerun()
    elif action_type == 'explain':
        if isinstance(content, dict):
            st.write(f"**{content.get('title', 'Explanation')}**")
            st.info(content.get('correct_understanding', ''))
            st.write(content.get('explanation', ''))
            if content.get('verification_question'):
                st.write(f"*Check your understanding:* {content['verification_question']}")
        else:
            st.write(str(content))
        if st.button('Got it'):
            next_res = api_get(f'/session/{st.session_state.session_id}/next')
            st.session_state.current_action = next_res.get('action') if next_res else None
            st.rerun()
    else:
        st.write('Review the material below:')
        st.json(content)
        if st.button('Continue'):
            next_res = api_get(f'/session/{st.session_state.session_id}/next')
            st.session_state.current_action = next_res.get('action') if next_res else None
            st.rerun()

def render_dashboard():
    st.header('Dashboard')
    if not st.session_state.subject_id:
        st.warning('Please select a subject.')
        return
    data = api_get(f'/subjects/{st.session_state.subject_id}/dashboard')
    if not data:
        st.info('No data available yet.')
        return
    sub = data['subject']
    col1, col2, col3 = st.columns(3)
    col1.metric('Average Mastery', f"{sub['avg_mastery']:.1f}%")
    col2.metric('Topics Studied', sub['topic_count'])
    col3.metric('Days to Exam', sub['days_until_exam'])
    st.divider()
    mastery = data.get('mastery_data', [])
    if mastery:
        df = pd.DataFrame(mastery)
        st.subheader('Topic Mastery')
        fig = px.bar(df, x='mastery_score', y='topic_name', orientation='h', text='mastery_score', color='mastery_score', color_continuous_scale=['#ff4b4b', '#ffa500', '#00c853'], range_x=[0, 100])
        fig.update_traces(texttemplate='%{text:.0f}%', textposition='outside')
        fig.update_layout(yaxis_title='', xaxis_title='Mastery %', height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader('Active Misconceptions')
        misc_data = []
        for t in mastery:
            for m in t.get('misconceptions', []):
                misc_data.append({'Topic': t['topic_name'], 'Misconception': m})
        if misc_data:
            st.table(pd.DataFrame(misc_data))
        else:
            st.write("You don't have any unresolved misconceptions right now.")

def render_study_plan():
    st.header('Study Plan')
    if not st.session_state.subject_id:
        st.warning('Please select a subject.')
        return
    plan = api_get(f'/subjects/{st.session_state.subject_id}/plan')
    if st.button('Generate / Replan Schedule'):
        with st.spinner('AI is generating your optimized schedule...'):
            api_post(f'/subjects/{st.session_state.subject_id}/plan/replan')
            st.rerun()
    if not plan:
        st.info('No study plan generated yet.')
        return
    df = pd.DataFrame(plan)
    if not df.empty:
        display_df = df[['date', 'topic_name', 'action_type', 'status']].copy()
        display_df.columns = ['Date', 'Topic', 'Activity', 'Status']
        st.dataframe(display_df, use_container_width=True, hide_index=True)

def render_mock_exam():
    st.header('Mock Exam')
    if not st.session_state.subject_id:
        st.warning('Please select a subject.')
        return
    if st.session_state.mock_paper is None:
        st.write('Generate a custom exam focused on your weakest topics.')
        if st.button('Generate Exam'):
            with st.spinner('Generating exam (this takes a moment)...'):
                paper = api_post(f'/subjects/{st.session_state.subject_id}/mock')
                if paper and paper.get('questions'):
                    st.session_state.mock_paper = paper
                    st.rerun()
                else:
                    st.error('Failed to generate exam.')
    else:
        paper = st.session_state.mock_paper
        st.subheader(paper.get('title', 'Practice Exam'))
        st.write(f"**Duration:** {paper.get('duration_minutes', 60)} mins | **Marks:** {paper.get('total_marks', 100)}")
        with st.form('mock_exam_form'):
            for q in paper.get('questions', []):
                st.write('---')
                st.write(f"**Q{q['number']}. {q['question']}** ({q.get('marks', 1)} marks)")
                if q.get('question_type') == 'mcq' and q.get('options'):
                    st.radio('Answer:', q['options'], key=f"q_{q['number']}", label_visibility='collapsed')
                else:
                    st.text_area('Answer:', key=f"q_{q['number']}", label_visibility='collapsed')
            st.write('---')
            if st.form_submit_button('Submit Exam'):
                st.success('Exam submitted! (Grading feature coming soon)')
        if st.button('Clear Exam'):
            st.session_state.mock_paper = None
            st.rerun()
if current_page == 'Subjects':
    render_subjects()
elif current_page == 'Study Session':
    render_study_session()
elif current_page == 'Dashboard':
    render_dashboard()
elif current_page == 'Study Plan':
    render_study_plan()
elif current_page == 'Mock Exam':
    render_mock_exam()