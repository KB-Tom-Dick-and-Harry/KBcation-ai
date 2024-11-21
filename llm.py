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

# def extract_transaction_data(pdf_path):
#     with pdfplumber.open(pdf_path) as pdf:
#         first_page = pdf.pages[0]
#         table = first_page.extract_table()

#     columns = table.pop(0)
#     transaction_df = pd.DataFrame(table, columns=columns)
#     return transaction_df

def extract_transaction_data(member_id, db_config):
    # SQLAlchemy 연결 문자열 생성
    connection_url = f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
    engine = create_engine(connection_url)
    
    query = """
        SELECT 
            consumption_id,
            category,
            consumption_details,
            date,
            member_id,
            spending_amount
        FROM consumption 
        WHERE member_id = %s 
        ORDER BY date DESC 
        LIMIT 10
    """
    
    # SQLAlchemy 엔진을 사용해 DataFrame으로 변환
    transaction_df = pd.read_sql(query, engine, params=(member_id,))
    
    return transaction_df

def get_retriever():
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    database = Chroma(
        collection_name='chroma-kb',
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
    dictionary = ["카드 -> 신용카드", "금융 상품 -> 카드","카드 추천 -> 신용카드추천"]
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요
        사전: {dictionary}
        
        질문: {{input}}
    """)

    dictionary_chain = prompt | llm | StrOutputParser()
    
    return dictionary_chain

def get_category_chain(transaction_data):
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        거래 내역을 분석하여 카테고리별로 분류하고 주요 패턴을 요약해주세요.
        예시 카테고리: 식비, 교통비, 쇼핑 등
        
        거래 내역:
        {transaction_details}
        """),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    return RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="output"
    )


def get_rag_chain(transaction_data, category_summary):
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
        당신은 KB 국민은행의 AI 금융 상담사입니다. 고객의 거래 내역을 바탕으로 적합한 KB 신용카드를 벡터 DB에서 찾아서 추천해주세요.
        주요 카테고리 별 지출 패턴을 참고하여 가장 적합한 KB 신용카드를 벡터 DB에서 찾아서 추천하고 혜택을 설명해주세요.
        타사의 카드를 추천하면 안됩니다. KB 국민카드에서 판매중인 신용카드만 추천해야합니다. 체크카드를 추천해주면 안됩니다.
        추천 카드명은 벡터 DB에서 정확한 카드명을 적어야 합니다.
        혜택은 추천카드만의 혜택을 설명해야합니다. 존재하지 않는 혜택을 언급해서는 안됩니다.
        주요 소비 패턴에 2줄정도로 간단하게 고객님의 소비패턴을 설명합니다(카드추천의 근거).
        최대한 사용자의 소비패턴과 관련성이 높은 카드를 추천하여야 합니다.
        관련성을 높이기 위해 존재하지 않는 혜택을 언급하거나 만들어내서는 안됩니다.
        KB국민 티타늄카드은 추천하면 안됩니다.
        BANK_TRANSACTION은 언급은 안됩니다.(소비자가 이해를 못합니다.)
        3개월치 소비패턴을 분석해서 카드추천을 해주고 예상 혜택금액 및 이점을 설명해줘.
        
        [참고할 정보]
        {context}

        [고객 정보]
        {category_summary}
         
        [거래 내역]
        {transaction_details}

        [질문]
        {input}

        [응답 형식]
        주요 소비 패턴 요약 :
        추천 카드 : 
        카드 혜택 :
        월 예상 혜택 금액 :

        [답변]:
        """),
        few_shot_prompt,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain =  RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    ).pick('answer')

    return conversational_rag_chain

def get_rag_chain(transaction_df, category_summary):
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
        당신은 KB 국민은행의 AI 금융 상담사입니다. 고객의 거래 내역을 바탕으로 적합한 KB 신용카드를 벡터 DB에서 찾아서 추천해주세요.
        주요 카테고리 별 지출 패턴을 참고하여 가장 적합한 KB 신용카드를 벡터 DB에서 찾아서 추천하고 혜택을 설명해주세요.
        타사의 카드를 추천하면 안됩니다. KB 국민카드에서 판매중인 신용카드만 추천해야합니다. 체크카드를 추천해주면 안됩니다.
        추천 카드명은 벡터 DB에서 정확한 카드명을 적어야 합니다.
        혜택은 추천카드만의 혜택을 설명해야합니다. 존재하지 않는 혜택을 언급해서는 안됩니다.
        주요 소비 패턴에 2줄정도로 간단하게 고객님의 소비패턴을 설명합니다(카드추천의 근거).
        최대한 사용자의 소비패턴과 관련성이 높은 카드를 추천하여야 합니다.
        관련성을 높이기 위해 존재하지 않는 혜택을 언급하거나 만들어내서는 안됩니다.
        KB국민 티타늄카드은 추천하면 안됩니다.
        BANK_TRANSACTION은 언급은 안됩니다.(소비자가 이해를 못합니다.)
        3개월치 소비패턴을 분석해서 카드추천을 해주고 예상 혜택금액 및 이점을 설명해줘.

        [참고할 정보]
        {context}

        [고객 정보]
        {category_summary}
         
        [거래 내역]
        {transaction_details}

        [질문]
        {input}

        [응답 형식]
        주요 소비 패턴 요약 :
        추천 카드 : 
        카드 혜택 :
        월 예상 혜택 금액 :

        [답변]:
        """),
        few_shot_prompt,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain =  RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    ).pick('answer')

    return conversational_rag_chain

def get_general_chain():
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

    system_prompt = (
        "KB 국민은행의 AI 금융 상담사입니다. 일반적인 문의에 답변해드리겠습니다."
        "아래에 제공된 문서를 활용해서 답변해주시고"
        "답변을 알 수 없다면 모른다고 답변해주세요"
        "2-3 문장정도의 짧은 내용의 답변을 원합니다"
        "\n\n"
        "{context}"
    )
    
    general_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        few_shot_prompt,
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    
    history_aware_retriever = get_history_retriever()
    question_answer_chain = create_stuff_documents_chain(llm, general_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    conversational_rag_chain=  RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    ).pick('answer')
    return conversational_rag_chain

def get_ai_response(user_question, session_id="abc123"):
    financial_keywords = ["카드추천", "신용카드추천", "신용카드 추천"]
    
    if any(keyword in user_question for keyword in financial_keywords):
        # 1. 거래 데이터 처리
        transaction_df = extract_transaction_data(member_id=1, db_config=db_config)
        
        # DataFrame을 JSON 형태로 변환
        transaction_data = transaction_df.to_dict(orient="records")  # 딕셔너리 리스트로 변환하여 사용
        
        # 나머지 코드
        dictionary_chain = get_dictionary_chain()
        formatted_question = dictionary_chain.invoke({"input": user_question})

        # transaction_data를 전달하여 JSON 데이터만 사용
        category_chain = get_category_chain(transaction_data)

        category_result = category_chain.invoke(
            {
                "transaction_details": transaction_data,
                "input": formatted_question
            },
            config={
                "configurable": {"session_id": session_id}
            }
        )

        # RAG 체인 생성 및 스트리밍 응답
        rag_chain = get_rag_chain(transaction_data, category_result)  # transaction_df 대신 transaction_data 사용
        financial_chain = {"input": formatted_question, "category_summary": category_result}
        
        # Stream 응답 생성
        ai_response = rag_chain.stream(
            {
                "input": financial_chain,
                "category_summary": category_result,
                "context": "",
                "transaction_details": transaction_data  # JSON 데이터 사용
            },
            config={
                "configurable": {"session_id": session_id}
            }
        )
    else:
        # 일반 질문 처리
        dictionary_chain = get_dictionary_chain()
        rag_chain = get_general_chain()
        general_chain = {"input": dictionary_chain} | rag_chain
        # Stream 응답 생성
        ai_response = general_chain.stream(
            {"input": user_question},
            config={
                "configurable": {"session_id": session_id}
            }
        )
    
    return ai_response

