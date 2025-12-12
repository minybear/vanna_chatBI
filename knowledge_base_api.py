from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from app import vn

router = APIRouter()

class TrainingDataRequest(BaseModel):
    id: Optional[str] = None
    type: str # 'ddl', 'documentation', 'sql'
    content: str
    question: Optional[str] = None

class BatchDeleteRequest(BaseModel):
    ids: List[str]

@router.get("/get_training_data")
def get_training_data():
    df = vn.get_training_data()
    if df is None or df.empty:
        return []
    return df.to_dict(orient='records')

@router.post("/training_data")
def add_training_data(data: TrainingDataRequest):
    try:
        if data.type == 'ddl':
            vn.train(ddl=data.content)
        elif data.type == 'documentation':
            vn.train(documentation=data.content)
        elif data.type == 'sql':
            if not data.question:
                raise HTTPException(status_code=400, detail="SQL training requires a question")
            vn.train(question=data.question, sql=data.content)
        else:
            raise HTTPException(status_code=400, detail="Invalid training type")
        return {"status": "success", "message": "Training data added"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/training_data/{id}")
def delete_training_data(id: str):
    try:
        success = vn.remove_training_data(id=id)
        if success:
            return {"status": "success", "message": "Training data removed"}
        else:
            raise HTTPException(status_code=404, detail="Training data not found or failed to remove")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/training_data/batch_delete")
def batch_delete_training_data(request: BatchDeleteRequest):
    try:
        deleted_count = 0
        errors = []
        for id in request.ids:
            try:
                success = vn.remove_training_data(id=id)
                if success:
                    deleted_count += 1
                else:
                    errors.append(f"Failed to remove {id}")
            except Exception as e:
                errors.append(f"Error removing {id}: {str(e)}")
        
        return {
            "status": "success", 
            "message": f"Deleted {deleted_count} items", 
            "deleted_count": deleted_count,
            "errors": errors
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
