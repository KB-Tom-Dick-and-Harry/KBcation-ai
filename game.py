from dotenv import load_dotenv
import pdfplumber
import pandas as pd
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, FewShotChatMessagePromptTemplate
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_chroma import Chroma
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from config import answer_examples , db_config

# Load environment variables
load_dotenv()

# Store for chat histories
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_llm(model='gpt-4o'):
    return ChatOpenAI(model=model)
    # return ChatOllama(model="gemma2")

def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    database = Chroma(
        collection_name='chroma-kb-game',
        persist_directory="./chroma",
        embedding_function=embedding
    )
    return database.as_retriever(search_kwargs={'k': 4})

def get_history_retriever():
    llm = get_llm()
    retriever = get_retriever()
    
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    return history_aware_retriever

def get_dictionary_chain():
    dictionary = ["game -> 게임"]
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}
        
        
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    
    return dictionary_chain

def get_rag_chain():
    llm = get_llm()
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{answer}"),
        ]
    )
    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=answer_examples,
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """
        [시스템]
        당신은 KB 국민은행의 경제 교육 게임 출제자입니다. 
        벡터 DB에서 검색된 유사한 문제들과 금융 정보를 참고하여 새로운 경제 퀴즈를 만들어주세요.

        [규칙]
        1. 문제는 경제, 금융과 관련된 내용이어야 합니다
        2. 4개의 객관식 보기를 제시해야 합니다
        3. 정답은 1개만 존재해야 합니다
        4. 모든 보기는 반드시 하나의 명사(단일 낱말)여야 합니다
           예시) 주식, 채권, 예금, 적금, 환율, 금리, 은행, 보험
        5. 난이도는 일반인도 충분히 풀 수 있는 수준이어야 합니다

        [보기 작성 예시]
        문제: 돈을 빌려주고 받는 대가를 뜻하는 말은?
        ① 금리
        ② 환율
        ③ 주가
        ④ 물가

        [벡터 DB 검색 결과]
        {context}

        [유의사항]
        1. 벡터 DB의 검색 결과를 참고하되, 단순 복사가 아닌 창의적인 변형이 필요합니다
        2. 모든 보기는 반드시 하나의 명사로만 구성되어야 합니다
        3. 긴 복합어나 구절은 사용하지 마세요

        [질문]
        {input}

        [응답 형식]
        0. 문제: (여기에 경제 관련 문제를 작성)
        1. ① (단일 명사)
        2. ② (단일 명사)
        3. ③ (단일 명사)
        4. ④ (단일 명사)
        5. 정답: (1~4 중 정답 번호)
        6. 해설: (문제와 정답에 대한 상세한 설명)

        [답변]:
        """),
        few_shot_prompt,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    history_aware_retriever = get_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    ).pick('answer')

    return conversational_rag_chain

def get_ai_response(user_message):
    rag_chain = get_rag_chain()
    
    # '게임생성' 입력이 들어오면 바로 퀴즈 생성
    if user_message.strip() == '게임생성':
        ai_response = rag_chain.stream(
            {
                "input": "경제 관련 퀴즈를 생성해주세요"
            },
            config={
                "configurable": {"session_id": "abc123"}
            },
        )
    else:
        # 다른 입력의 경우 기존 로직 유지
        dictionary_chain = get_dictionary_chain()
        game_chain = {"input": dictionary_chain} | rag_chain
        ai_response = game_chain.stream(
            {
                "input": user_message
            },
            config={
                "configurable": {"session_id": "abc123"}
            },
        )

    return ai_response