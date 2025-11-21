import streamlit as st
import tiktoken
from loguru import logger

# LangChain 코어 모듈
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


# LangChain OpenAI 통합 모듈
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import get_openai_callback

# LangChain 커뮤니티 통합 모듈
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import UnstructuredPowerPointLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# 기타 모듈
from langchain_text_splitters import RecursiveCharacterTextSplitter

# from streamlit_chat import message

def main():
    st.set_page_config(
    page_title="DirChat",   #앱 이름
    page_icon=":books:")    #앱 아이콘

    st.title("_Private Data :red[QA Chat]_ :books:")    #앱 제목

    if "conversation" not in st.session_state:
        st.session_state.conversation = None        # conversation 사용을 위한 초기화

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None        # chat_history 사용을 위한 초기화

    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None     # processComplete 사용을 위한 초기화

    with st.sidebar:    # 구성 요소 안에 하위 구성요소가 필요할 때 with문 사용
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
    if process:   # Process 버튼이 눌렸을 때
        if not openai_api_key: # OpenAI API 키가 입력되지 않았을 때
            st.info("Please add your OpenAI API key to continue.") # 알람
            st.stop()
        files_text = get_text(uploaded_files)   # 업로드 된 파일을 텍스트로 변환
        text_chunks = get_text_chunks(files_text)       # 텍스트를 청크 단위로 분할
        vetorestore = get_vectorstore(text_chunks)      # 분할된 텍스트 청크를 벡터스토어로 변환
     
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) # 체인 생성

        st.session_state.processComplete = True

    if 'messages' not in st.session_state:  # 채팅 화면을 구현하기 위한 코드, 채팅 창에 이전 메시지들이 남아 있도록 함
        st.session_state['messages'] = [{"role": "assistant", 
                                        "content": "안녕하세요! 주어진 문서에 대해 궁금하신 것이 있으면 언제든 물어봐주세요!"}]  #초기 메시지 설정

    for message in st.session_state.messages:   # 메세지마다 with문으로 묶어 출력
        with st.chat_message(message["role"]):  # 메세지의 역할에 따라 content 생성
            st.markdown(message["content"])

    history = StreamlitChatMessageHistory(key="chat_messages")

    # Chat logic
    if query := st.chat_input("질문을 입력해주세요."): # 사용자가 질문을 입력한다면
        st.session_state.messages.append({"role": "user", "content": query}) # 질문이 있으면 role을 user로 하고 content에 질문을 넣어 메시지에 추가

        with st.chat_message("user"): 
            st.markdown(query)  # 채팅장에 사용자 메세지 출력

        with st.chat_message("assistant"):
            chain = st.session_state.conversation

            with st.spinner("Thinking..."): # 로딩창
                result = chain.invoke({"question": query}) #LLM 체인에 질문 전달 후 받은 결과
                with get_openai_callback() as cb:
                    st.session_state.chat_history = result['chat_history'] # 대화 기록 저장
                response = result['answer'] # LLM이 생성한 답변
                source_documents = result['source_documents'] # 참고 문서

                st.markdown(response)
                with st.expander("참고 문서 확인"): # 접었다 펼칠 수 있는 참고 문서 영역
                    st.markdown(source_documents[0].metadata['source'], help = source_documents[0].page_content) # 첫번째 metadata의 source 출력, help는 참고한 문서의 청크
                    st.markdown(source_documents[1].metadata['source'], help = source_documents[1].page_content)
                    st.markdown(source_documents[2].metadata['source'], help = source_documents[2].page_content)
                    


# Add assistant message to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

def tiktoken_len(text): # text의 토큰 길이 계산
    tokenizer = tiktoken.get_encoding("cl100k_base") # tiktoken의 cl100k_base 인코딩 사용
    tokens = tokenizer.encode(text)
    return len(tokens)

def get_text(docs): # 업로드된 문서를 텍스트로 변환

    doc_list = []
    
    for doc in docs:
        file_name = doc.name  # doc 객체의 이름을 파일 이름으로 사용
        with open(file_name, "wb") as file:  # 파일을 doc.name으로 저장
            file.write(doc.getvalue())
            logger.info(f"Uploaded {file_name}")
        if '.pdf' in doc.name: # PDF 파일인 경우
            loader = PyPDFLoader(file_name)
            documents = loader.load_and_split()
        elif '.docx' in doc.name: # DOCX 파일인 경우
            loader = Docx2txtLoader(file_name)
            documents = loader.load_and_split()
        elif '.pptx' in doc.name: # PPTX 파일인 경우
            loader = UnstructuredPowerPointLoader(file_name)
            documents = loader.load_and_split()

        doc_list.extend(documents)
    return doc_list


def get_text_chunks(text): # 텍스트를 청크 단위로 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, # 청크 크기
        chunk_overlap=100, # 청크 겹침
        length_function=tiktoken_len  # 토큰 길이 계산 함수
    )
    chunks = text_splitter.split_documents(text) # 문서를 청크 단위로 분할
    return chunks


def get_vectorstore(text_chunks): # 텍스트 청크를 벡터스토어로 변환
    embeddings = HuggingFaceEmbeddings(
                                        model_name="jhgan/ko-sroberta-multitask", # 한국어 모델 사용
                                        model_kwargs={'device': 'cpu'},
                                        encode_kwargs={'normalize_embeddings': True}
                                        )  
    vectordb = FAISS.from_documents(text_chunks, embeddings) # FAISS 벡터스토어 생성
    return vectordb

def get_conversation_chain(vetorestore,openai_api_key): # 대화 체인 생성
    llm = ChatOpenAI(openai_api_key=openai_api_key, model_name = 'gpt-4o',temperature=0)
    conversation_chain = ConversationalRetrievalChain.from_llm(  
            llm=llm, 
            chain_type="stuff", 
            retriever=vetorestore.as_retriever(search_type = 'mmr'), 
            memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer'), # 대화 메모리 설정
            get_chat_history=lambda h: h,
            return_source_documents=True,
            verbose = True
        )

    return conversation_chain



if __name__ == '__main__':
    main()
