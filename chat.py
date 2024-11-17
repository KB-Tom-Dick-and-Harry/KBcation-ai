import streamlit as st
from dotenv import load_dotenv
from llm import get_ai_response, extract_transaction_data
from config import db_config

# 페이지 설정
st.set_page_config(page_title="카드 추천 챗봇", page_icon="💳")

st.title("💳 카드 추천 챗봇")
st.caption("고객님의 거래 내역을 분석하여 맞춤형 카드를 추천해드립니다!")

# 환경 변수 로드
load_dotenv()

# 사이드바에 사용자 ID 입력
member_id = st.sidebar.number_input("사용자 ID", min_value=1, value=1)

# 거래 내역을 데이터베이스에서 추출
if 'transaction_df' not in st.session_state:
    try:
        transaction_df = extract_transaction_data(member_id, db_config)
        st.session_state.transaction_df = transaction_df
        
        if transaction_df.empty:
            st.warning("거래 내역이 없습니다.")
    except Exception as e:
        st.error(f"거래 내역 조회 중 오류가 발생했습니다: {str(e)}")
        st.session_state.transaction_df = None

# 거래 내역이 있는 경우에만 표시
if st.session_state.transaction_df is not None and not st.session_state.transaction_df.empty:
    with st.expander("거래 내역 보기"):
        st.dataframe(st.session_state.transaction_df)

# 메시지 리스트 초기화
if 'message_list' not in st.session_state:
    st.session_state.message_list = []

# 이전 메시지 출력
for message in st.session_state.message_list:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자의 질문을 입력받기
if user_question := st.chat_input(placeholder="거래 내역을 바탕으로 궁금한 사항을 입력해 주세요!"):
    # 사용자 메시지 추가
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.message_list.append({"role": "user", "content": user_question})

    # AI 응답 생성
    with st.chat_message("assistant"):
        with st.spinner("응답을 생성하고 있습니다..."):
            try:
                response_stream = get_ai_response(user_question, session_id=f"user_{member_id}")
                message_placeholder = st.empty()
                full_response = []
                
                # 스트리밍 응답 처리
                for chunk in response_stream:
                    if chunk:
                        full_response.append(str(chunk))
                        message_placeholder.markdown("".join(full_response))
                
                final_response = "".join(full_response)
                st.session_state.message_list.append({"role": "assistant", "content": final_response})
                
            except Exception as e:
                error_message = f"응답 생성 중 오류가 발생했습니다: {str(e)}"
                st.error(error_message)
                st.session_state.message_list.append({"role": "assistant", "content": error_message})

# 대화 내역 초기화 버튼
if st.sidebar.button("대화 내역 초기화"):
    st.session_state.message_list = []
    st.experimental_rerun()