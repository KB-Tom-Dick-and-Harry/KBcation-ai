from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from game import get_ai_response
from fastapi.middleware.cors import CORSMiddleware
import logging
import traceback

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GameQuestion(BaseModel):
    quiz: str
    answer_options: List[str]
    correct_answer: str
    answer_explanation: str

class GameResponse(BaseModel):
    success: bool
    data: Optional[GameQuestion] = None
    error: Optional[str] = None

def parse_game_response(response_text: str) -> GameQuestion:
    try:
        lines = response_text.strip().split('\n')
        quiz = lines[0].replace('0. 문제: ', '').strip()
        options = [
            lines[1].replace('1. ① ', '').strip(),
            lines[2].replace('2. ② ', '').strip(),
            lines[3].replace('3. ③ ', '').strip(),
            lines[4].replace('4. ④ ', '').strip()
        ]
        
        # 정답 텍스트에서 번호만 추출
        answer_text = lines[5].replace('5. 정답: ', '').strip()
        if '①' in answer_text:
            correct_answer_num = 1
        elif '②' in answer_text:
            correct_answer_num = 2
        elif '③' in answer_text:
            correct_answer_num = 3
        elif '④' in answer_text:
            correct_answer_num = 4
        else:
            answer_num = answer_text[0]  # 첫 번째 문자만 사용
            correct_answer_num = int(answer_num)
            
        explanation = lines[6].replace('6. 해설: ', '').strip()
        
        return GameQuestion(
            quiz=quiz,
            answer_options=options,
            correct_answer=options[correct_answer_num - 1],
            answer_explanation=explanation
        )
    except Exception as e:
        logger.error(f"Error parsing response: {str(e)}")
        logger.error(f"Response text: {response_text}")
        logger.error(traceback.format_exc())
        raise ValueError(f"Failed to parse game response: {str(e)}")

@app.post("/api/game/generate")
async def generate_game() -> GameResponse:
    try:
        # AI 모델을 사용하여 게임 생성
        ai_response = get_ai_response("게임생성")
        response_text = "".join([chunk for chunk in ai_response])
        
        logger.info(f"AI Response received: {response_text}")
        
        # 응답 파싱
        game_question = parse_game_response(response_text)
        return GameResponse(success=True, data=game_question)
        
    except Exception as e:
        logger.error(f"Error generating game: {str(e)}")
        logger.error(traceback.format_exc())
        return GameResponse(success=False, error=str(e))

if __name__ == "__main__":
    uvicorn.run("game_server:app", host="0.0.0.0", port=8001, reload=True)