from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from llm import get_ai_response
import mysql.connector
from mysql.connector import Error
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from config import db_config, answer_examples
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str
    memberId: int

def create_db_connection():
    """db_config을 사용하여 데이터베이스 연결 설정"""
    try:
        connection = mysql.connector.connect(
            host=db_config["host"],
            port=db_config["port"],
            user=db_config["user"],
            password=db_config["password"],
            database=db_config["database"]
        )
        logger.info("MySQL 데이터베이스 연결 성공")
        return connection
    except Error as e:
        logger.error(f"MySQL 연결 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=f"데이터베이스 연결 오류: {str(e)}")

def collect_response(stream):
    """스트림 응답을 단일 문자열로 수집"""
    full_response = []
    try:
        for chunk in stream:
            if chunk:
                full_response.append(str(chunk))
    except Exception as e:
        logger.error(f"스트림 수집 오류: {str(e)}")
        raise
    
    return "".join(full_response)

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        logger.info(f"채팅 요청 수신: {request}")
        
        # 금융 관련 키워드 확인
        financial_keywords = ["소비", "지출", "추천", "패턴", "내역", "분석", "사용", "금액", "카드추천", "신용카드"]
        is_financial_query = any(keyword in request.question for keyword in financial_keywords)
        
        # 세션 ID 생성
        session_id = f"user_{request.memberId}_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"
        
        # 일반 대화 처리
        if not is_financial_query:
            stream_response = get_ai_response(request.question, session_id)
            response = collect_response(stream_response)
            return {"answer": response}
        
        # 금융 관련 대화 처리
        conn = create_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        try:
            # 소비 데이터 조회
            cursor.execute("""
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
            """, (request.memberId,))
            
            consumption_data = cursor.fetchall()
            
            # 소비 데이터가 없는 경우 처리
            if not consumption_data:
                return {
                    "answer": "죄송합니다. 현재 분석할 수 있는 소비 내역이 없어 맞춤형 추천이 어렵습니다. "
                             "소비 내역이 쌓이면 더 정확한 분석과 추천이 가능합니다. "
                             "일반적인 금융 상품이나 서비스에 대해 알고 싶으신 점이 있다면 말씀해 주세요."
                }
            
            # AI 응답 생성
            stream_response = get_ai_response(request.question, session_id)
            response = collect_response(stream_response)
            
            return {"answer": response}
            
        finally:
            cursor.close()
            conn.close()
            logger.info("데이터베이스 연결 종료")
        
    except Exception as e:
        logger.error(f"오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("llm_server:app", host="0.0.0.0", port=8000, reload=True)