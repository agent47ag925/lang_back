# langchain==0.3.18
# langchain-core==0.3.35
# langchain-text-splitters==0.3.6
# langsmith==0.3.8
import os
from dotenv import load_dotenv

#openai api를 활용하여 채팅 
from openai import OpenAI

#모델과의 대화를 주도
from langchain_community.chat_models import ChatOpenAI
#대화 시 프롬프트를 미리 세팅, 전달할
from langchain.prompts import ChatPromptTemplate

#RAG관련된 라이브러리
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import retrieval_qa
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader

import tiktoken
#문서를 가져왔을 때, 단어를 나누는 작업(토큰화) -> 문자를 나누는 작업을 도와주는 역할
from langchain.text_splitter import RecursiveCharacterTextSplitter

#유사도를 측정하는 알고리즘 중 하나
from langchain_community.retrievers import BM25Retriever

#db관련 라이브러리
import LocalDB as DB
import pandas as pd
from tabulate import tabulate
from langchain_experimental.agents import create_pandas_dataframe_agent

def titoken_len(text):
  tokenizer = tiktoken.get_encoding('cl100k_base')
  tokens = tokenizer.encode(text)
  return len(tokens)

def default_chat(query):
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('API_KEY') 
    Chat = ChatOpenAI(model='gpt-4o-mini', temperature=0.5)

    template_string = '''
                        입력되는 내용 : {inputs}

                        사용자가 inputs란 내용으로 너에게 질문을 할거야.
                        너는 상냥하고 따뜻한, 긍정적인 어조로 사용자의 질문에 답변해줘.

                        혹시라도, 입력되는 inputs에
                        욕설이 있으면 제거하고 *로 대체해서 보여줘.

                        예시 : 나쁜 새끼... -> 나쁜 **
                        '''
    
    prompt_template = ChatPromptTemplate.from_template(template_string)

    #실제로 포맷된 메시지를 전달하기 위한 작업
    chat_message = prompt_template.format_messages(inputs=query)

    response = Chat(chat_message)
    return response.content


#합치지는 마세요!
#동작하지 않음음
def rag_chat(query, files, ext):
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('API_KEY') 
    client = OpenAI()

    if ext == 'txt':
        loader = TextLoader(files)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500,
                                               chunk_overlap = 20,
                                               length_function = titoken_len)
        
        docs = text_splitter.split_documents(docs)
        bm25 = BM25Retriever.from_documents(docs)
        docs = bm25.invoke(query)
        #print(docs)

        message = [{'role':'system', 'content':f'''
                리트리버를 통해 추가된 docs 내용을 참고하여 답변을 완성하고, 따뜻한 말투로 답변해줘.
                docs : {[x.page_content for x in docs]}                                
                '''},
                {'role':'user', 'content': query}]
    
        #print(message)

    #pdf가 추가된 경우
    else: 
        loader = PyPDFLoader(files)

        #페이지 단위로 pdf를 분할
        texts = loader.load_and_split()
        embedding = OpenAIEmbeddings()
        vectordb = Chroma.from_documents(documents=texts, embedding=embedding)

        documents = vectordb._collection.get()['documents']

        chat =  ChatOpenAI(temperature = 0.1, model='gpt-3.5-turbo')
        multi_retriever = MultiQueryRetriever.from_llm(retriever=vectordb.as_retriever(search_kwargs={'k':2}),
                                               llm = chat)
        
        docs = multi_retriever.invoke(query)
        #print(docs)
        message = [{'role':'system', 'content':f'''
                리트리버를 통해 추가된 docs 내용을 참고하여 답변을 완성하고, 따뜻한 말투로 답변해줘.
                docs : {[x.page_content for x in docs]}                                
                '''},
                {'role':'user', 'content': query}]
    
    response = client.chat.completions.create(model = 'gpt-3.5-turbo',
                            messages = message,
                            temperature = 0.1) 
    
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def db_chat(query, table):
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('API_KEY') 
    dataframe = DB.show_data(table)
    print(dataframe)

    agent = create_pandas_dataframe_agent(
        ChatOpenAI(temperature=0, model='gpt-4o-mini'),
        dataframe,
        agent_type='tool-calling',
        allow_dangerous_code=True
    )

    response = agent.invoke({query})
    return response



#if __name__ == '__main__':
    #rag_chat txt 버전테스트
    # query = '등장인물 이름이 누구누구나와? 무슨 상황이야?'
    # files =  'C:/Users/jeong/Desktop/OpenAISupport/Project/Morris_Water_1897.txt'
    # ext = 'txt'

    #rag_chat pdf 버전테스트트
    # query = '문서의 내용 요약해줄래?'
    # files =  'C:/Users/jeong/Desktop/OpenAISupport/Project/MBTI.pdf'
    # ext = 'pdf'

    # r = rag_chat(query=query, files=files, ext=ext)
    # print(f"RESULT : {r}")

    #db_chat 함수 개별 테스트
    # r = db_chat('가장 높은 예적금 금액이 얼마야?', 'finance')
    # print(r)
