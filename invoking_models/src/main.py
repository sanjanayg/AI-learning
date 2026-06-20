# from fastapi import FastAPI, UploadFile, File, HTTPException

# from schemas import ExtractTextResponse
# from file_service import FileService
# from pdf_service import PDFService
# from llm_service import LLMService


# app = FastAPI(
#     title="Image and PDF Text Extraction API",
#     version="1.0.0"
# )

# ALLOWED_IMAGE_TYPES = [
#     "image/jpeg",
#     "image/png",
#     "image/webp"
# ]

# @app.post("/extract-text", response_model=ExtractTextResponse)
# async def extract_text(file: UploadFile = File(...)):
#     try:
#         file_bytes, mime_type = await FileService.read_and_validate(file)

#         llm_service = LLMService()

#         if mime_type in FileService.ALLOWED_IMAGE_TYPES:
#             base64_image = FileService.encode_to_base64(file_bytes)

#             extracted_text = llm_service.extract_text_from_image(
#                 base64_image=base64_image,
#                 mime_type=mime_type
#             )

#             return ExtractTextResponse(
#                 success=True,
#                 file_type="image",
#                 extraction_method="llm_ocr",
#                 extracted_text=extracted_text
#             )

#         if mime_type == FileService.PDF_TYPE:
#             pdf_analysis = PDFService.analyze_pdf(file_bytes)

#             pdf_type = pdf_analysis["pdf_type"]

#             if pdf_type == "text_pdf":
#                 extracted_text = PDFService.extract_text_from_pdf(file_bytes)

#                 return ExtractTextResponse(
#                     success=True,
#                     file_type="pdf",
#                     extraction_method="direct_text_extraction",
#                     extracted_text=extracted_text
#                 )

#             page_images = PDFService.convert_pdf_to_images(file_bytes)

#             extracted_pages = []

#             for page in page_images:
#                 base64_image = FileService.encode_to_base64(
#                     page["image_bytes"]
#                 )

#                 page_text = llm_service.extract_text_from_image(
#                     base64_image=base64_image,
#                     mime_type=page["mime_type"]
#                 )

#                 extracted_pages.append(
#                     f"\n--- Page {page['page_no']} ---\n{page_text}"
#                 )

#             return ExtractTextResponse(
#                 success=True,
#                 file_type="pdf",
#                 extraction_method=f"{pdf_type}_llm_ocr",
#                 extracted_text="\n".join(extracted_pages).strip()
#             )

#         raise HTTPException(
#             status_code=400,
#             detail="Unsupported file type"
#         )

#     except HTTPException:
#         raise

#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Text extraction failed: {str(e)}"
#         )
    

from fastapi import FastAPI
from api.endpoints import extraction_router, chunking_router

app = FastAPI(
    title="Image and PDF Text Extraction API",
    description="Production-grade document text parsing and LLM OCR orchestrator.",
    version="1.0.0"
)

app.include_router(extraction_router)
app.include_router(chunking_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)