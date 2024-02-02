import os
import utils
import streamlit as st
from streaming import StreamHandler
import urllib3
import certifi
import json
import re
from bs4 import BeautifulSoup
from bs4.element import Comment
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import DocArrayInMemorySearch
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

st.set_page_config(page_title="ChatPDF", page_icon="📄")
st.header('Chat with Federal Registry Rules')

#viewing the tags in the html doc
def tags(element):
  if element.parent.name in ['style', 'head', 'script', 'meta', 'title', '[document]']:
    return False
  if isinstance(element, Comment):
    return False
  return True

#look at text in html and filter by the tags
# def get_html_text(body):
#   soup = BeautifulSoup(body, 'html.parser')
#   texts = soup.findAll(string=True)
#   visible_texts = filter(tags, texts)
#   return u" ".join(t.strip() for t in visible_texts)
def get_html_text(body):
  soup = BeautifulSoup(body, 'html.parser')
  texts = soup.get_text()
  return texts

class CustomDataChatbot:

    def __init__(self):
        utils.configure_openai_api_key()
        self.openai_model = "gpt-3.5-turbo"

    @st.spinner('Analyzing documents..')
    def setup_qa_chain(self, doc_number):
        # Load documents
        docs = []
        http = urllib3.PoolManager(
            cert_reqs='CERT_REQUIRED',
            ca_certs = certifi.where()
            )
        
        header = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) '
        'AppleWebKit/537.11 (KHTML, like Gecko) '
        'Chrome/23.0.1271.64 Safari/537.11',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
        'Accept-Encoding': 'none',
        'Accept-Language': 'en-US,en;q=0.8',
        'Connection': 'keep-alive'}

        #getting fed reg doc
        st.write("Fetching Federal Register Rule...")
        output = []
        api_url = f"https://www.federalregister.gov/api/v1/documents/{doc_number}.json?fields[]=html_url&fields[]=title"
        r = http.request('GET', api_url)
        json_data = json.loads(r.data.decode('utf-8'))
        output.append(json_data["html_url"])
        output.append(json_data['title'])

        url = output[0]
        title = output[1]
        #getting text
        req = http.request("GET", url=url, headers=header)
        page = req.data
        st.write("Turning Page into Text...")
        text_page = get_html_text(page)

        index = text_page.find(title)
        index_end = text_page.find("Published Document Home Home Sections Money Environment World Science")
        text_page = text_page[index:index_end]

        str_remove = "Please enable JavaScript if it is disabled in your browser or access the information through the links provided below."
        if str_remove in text_page:
            text_page = text_page.replace(text_page.split(str_remove, 1)[0] + str_remove, "", 1)
        text_page = re.sub(r'\s+', ' ', text_page)

        
        st.write("Chunking Text...")
        # text_splitter =  CharacterTextSplitter(
        #     separator= " ",
        #     chunk_size = 500,
        #     chunk_overlap = 200,
        #     length_function = len,
        # )
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        text_splitter = SemanticChunker(embeddings)
        html_texts = text_splitter.split_text(text_page)
        # Create embeddings and store in vectordb
        st.write("Vectorizing Text...")

        st.write("Storing Vectors...")
        vectordb = Chroma.from_texts(html_texts, embeddings)

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k':4, 'fetch_k':7}
        )

        # Setup memory for contextual conversation        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )
        st.write("Creating Chain...")
        # Setup LLM and QA chain
        llm = ChatOpenAI(model_name=self.openai_model, temperature=0, streaming=True)
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, verbose=True)
        st.write("Chain Finished!")
        return qa_chain

    @utils.enable_chat_history
    def main(self):

        if "qa_chain" not in st.session_state:
           st.session_state.qa_chain = None
        # User Inputs
        with st.sidebar:
            
            uploaded_doc = st.text_input("Enter the Document Number", placeholder="Ex: 2024-12345")
            if st.button("Process"):
               
               st.session_state.qa_chain= self.setup_qa_chain(uploaded_doc)
            if not uploaded_doc:
                st.error("Please enter a Document Number!")
                st.stop()

        user_query = st.chat_input(placeholder="Ask me anything!")

        if uploaded_doc and user_query:
            

            utils.display_msg(user_query, 'user')

            with st.chat_message("assistant"):
                st_cb = StreamHandler(st.empty())
                response = st.session_state.qa_chain.run(user_query, callbacks=[st_cb])
                response = response.replace(user_query, '')
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    obj = CustomDataChatbot()
    obj.main()
