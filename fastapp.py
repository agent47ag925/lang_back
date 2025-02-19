#이 코드를 실행하기 전 fastapi와 uvicorn 설치 먼저 실행해야 함
# pip install fastapi uvicorn

#이 fastapi 파일을 단독으로 실행하기 위한 코드
#uvicorn 파일이름:앱이름 --리로드옵션션
#uvicorn fastapp:app --reload
import os
import uvicorn
from fastapi import FastAPI
from fastapi import File, UploadFile, Form, Depends
from pydantic import BaseModel #인풋데이터의 형식을 고정정
from typing import Optional
#BaseModel 상속후, 내 모델을 만듦 -> 내 모델이 인풋의 형식 고정정

#랭체인으로 답변을 받을 수 있도록 모듈화 코드를 가지고 옴
import LangModule as LM

#데이터베이스의 내용을 가져올 수 있도록
import LocalDB as DB

import shutil

#일반적인 채팅 형태의 인풋이 들어오는 경우
class user_input(BaseModel):
    inputs : str    #사용자가 쿼리한 내용
    history : list  #사용자의 이전 대화내역

#첨부파일(.jpg, .png같은 그림파일, .pdf)가 같이 들어오는 경우
class user_attach(BaseModel):
    inputs : str    #사용자가 쿼리한 내용
    extension : str #첨부된 파일의 확장자
    dbuse : str    #DB를 활용한 챗을 진행할지 여부

#음성채팅을 입력한 경우
class user_voice(BaseModel):
    inputs : str    #사용자가 쿼리한 내용

    #데이터베이스를 사용하는 경우
class user_db(BaseModel):
    inputs : str    #사용자가 쿼리한 내용
    dbtable : Optional[str]

app = FastAPI()

#일반채팅
@app.post('/chat')
def chat(input:user_input):
    print(input)
    #input -> 유저의 질문
    #history -> main_chat[]에다 쓰려고

    #메모리가 없음
    response = LM.default_chat(input.inputs)
    print("REST RETURN :" , response)
    return response

#첨부파일을 가지고 있는 채팅
@app.post('/attached')
async def attached(inputs: str = Form(...),
    extension: str = Form(...),
    files: Optional[UploadFile] = File(None)):

    try:
        file_content = await files.read()
        print("FFF", file_content)
    
    except Exception as e:
        print(f"에러 발생 : {str(e)}")

    finally:
                #그림 파일 일때를 분리
        if extension == 'txt':
            response = LM.rag_chat(inputs, file_content.decode('utf-8'), 'txt')
        
        else:
            response = LM.rag_chat(inputs, file_content, 'rag')

    #print(response)
    return response

@app.post('/db')
def db(inputs:user_db):
    print(inputs.inputs)
    print(inputs.dbtable)

    if inputs.dbtable == None:
        response = LM.default_chat(inputs.inputs)
    else:
        response = LM.db_chat(inputs.inputs, inputs.dbtable)
        response = response['output']

    return response


#음성채팅
@app.post('/voice')
def voice(input:user_voice):
    response = LM.default_chat(input.inputs)
    return response

#헤로쿠 배포 시 이슈 있어서 진입점을 따로 만듦듦
if __name__ == '__main__':
    port = int(os.environ.get('API_PORT', 8001))
    uvicorn.run(app, host='0.0.0.0', port=port)