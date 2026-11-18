import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

def show_dashboard(df):
    if df.empty:
        st.info('No predictions have been logged yet. Start the camera to collect data.')
        return
    df = df.copy()
    df['captured_at'] = pd.to_datetime(df['captured_at'], errors='coerce')
    total = len(df)
    top = df['dominant_emotion'].mode().iat[0] if not df['dominant_emotion'].mode().empty else 'N/A'
    avg = df['confidence'].mean()
    a, b, c = st.columns(3)
    a.metric('Predictions logged', f'{total:,}')
    b.metric('Most frequent expression', str(top).title())
    c.metric('Mean predicted probability', f'{avg:.1%}')
    st.subheader('Expression distribution')
    st.bar_chart(df['dominant_emotion'].value_counts())
    st.subheader('Probability over time')
    timed = df.dropna(subset=['captured_at']).sort_values('captured_at')
    if not timed.empty:
        st.line_chart(timed.set_index('captured_at')['confidence'])
    st.subheader('Probability histogram')
    fig, ax = plt.subplots()
    ax.hist(df['confidence'].dropna(), bins=10)
    ax.set_xlabel('Predicted class probability'); ax.set_ylabel('Frequency')
    st.pyplot(fig); plt.close(fig)
    st.subheader('Recent records')
    st.dataframe(df.head(200), use_container_width=True)
