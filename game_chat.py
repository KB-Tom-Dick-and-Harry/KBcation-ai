import streamlit as st


from dotenv import load_dotenv


from game import get_ai_response

st.set_page_config(page_title="게임 챗봇", page_icon="🎲")

st.title("🎲 퀴즈 게임 챗봇")
st.caption("경제 관련 게임을 생성 해드립니다.")

load_dotenv()

if 'message_list' not in st.session_state:
    st.session_state.message_list = []

for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])




if user_question := st.chat_input(placeholder="게임을 생성 해주세요"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    with st.spinner("게임을 생성하는 중입니다"):
        ai_response = get_ai_response(user_question)
        with st.chat_message("ai"):
            ai_message = st.write_stream(ai_response)
            st.session_state.message_list.append({"role": "ai", "content": ai_message})
