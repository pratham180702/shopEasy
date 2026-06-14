from pydantic import BaseModel


class GetQuery(BaseModel):
    query : str
    session_id: str

class ReturnQuery(BaseModel):
    response: str