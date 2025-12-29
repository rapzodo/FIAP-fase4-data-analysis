from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import PyPDF2
import json

class PDFParserToolInput(BaseModel):
    pdf_path: str = Field(..., description="Path to the PDF file to parse")

class PDFParserTool(BaseTool):
    name: str = "pdf_parser_tool"
    description: str = "Extracts text content from PDF files for analysis. Returns the full text content of the PDF document."
    args_schema: type[BaseModel] = PDFParserToolInput

    def _run(self, pdf_path: str) -> str:
        try:
            text_content = []

            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)

                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    text_content.append(f"--- Page {page_num + 1} ---\n{text}\n")

            full_text = "\n".join(text_content)

            return json.dumps({
                "success": True,
                "num_pages": num_pages,
                "content": full_text
            }, indent=2)

        except Exception as e:
            return json.dumps({
                "success": False,
                "error": str(e)
            }, indent=2)

