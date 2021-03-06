from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Optional
import uvicorn
from pydantic import BaseModel

from nao import NumericallyAugmentedElectra
import nao
app = FastAPI()

config = nao.config
model = NumericallyAugmentedElectra(config)

class Item(BaseModel):
    passage: str
    question : str

@app.get("/")
async def read_root():
    return {"msg": "World"}

@app.post("/message")
async def message(item: Item):
    passage = item.passage
    question = item.question



    result = model.query_sender(passage = passage, question = question)
    print(result)
    return JSONResponse(result)



# 46777
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=46777)
