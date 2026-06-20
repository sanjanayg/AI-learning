from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from schemas import ExtractTextResponse
from services.extraction_service import ExtractionService
from services.chunking_service import RAGPipelineService

extraction_router = APIRouter(tags=["Extraction"])
chunking_router = APIRouter(tags=["Chunking"])

# Dependency Injection for our service class instance
def get_extraction_service() -> ExtractionService:
    return ExtractionService()

@extraction_router.post("/extract-text", response_model=ExtractTextResponse)
async def extract_text(
    file: UploadFile = File(...),
    service: ExtractionService = Depends(get_extraction_service)
):
    try:
        return await service.extract_text_from_file(file)
        
    except HTTPException:
        # Re-raise known API errors untouched
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Text extraction pipeline failed: {str(e)}"
        )

chunking_service=RAGPipelineService()

@chunking_router.post("/extract-text-convert-chunks")
async def extract_text(file: UploadFile = File(...)):
    
    try:
        return await chunking_service.extract_and_chunk(file)
        
    except HTTPException:
        # Re-raise known API errors untouched
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Text extraction pipeline failed: {str(e)}"
        )


@chunking_router.post("/semantic-chunking")
async def semantic_chunking_api(file: UploadFile = File(...)):
    
    try:
        return await chunking_service.semantic_chunking(file)
        
    except HTTPException:
        # Re-raise known API errors untouched
        raise
    except Exception as e:
        
        raise HTTPException(
            status_code=500,
            detail=f"Text extraction pipeline failed: {str(e)}"
        )