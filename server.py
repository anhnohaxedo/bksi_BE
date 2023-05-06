from fastapi import FastAPI
from db_helper import DB_helper
# from index import
from document_model import DocumentModel
from bksi_search_engine import cosine_sim_func, candidates, preprocess
app = FastAPI()

@app.get("/create_index")
async def create_index(documents):
    return {"documents_list": documents}

@app.get("/update_index")
async def update_index(documents: DocumentModel):
    return documents

@app.get("/delete_index")
async def delete_index(index_id: int):
    return {"message": index_id}

@app.get("/delete_all_index")
async def delete_index():
    return {"message": "delete all"}

@app.get("/get_docs")
async def get_docs(query: str, status_code = 200):
    doc_list = cosine_sim_func(preprocess(query), candidates(preprocess(query)))
    return {"documents_id_list": doc_list}

@app.get("/classify")
async def classify(query: str):
    return {"message": query}

@app.get("/test")
async def test(status_code = 200):
    doc_list = cosine_sim_func(preprocess('chương trình song bằng'), candidates(preprocess('chương trình song bằng')))
    return {"documents_id_list": doc_list}

@app.get("/")
async def test():
    return {'message': 'hello'}