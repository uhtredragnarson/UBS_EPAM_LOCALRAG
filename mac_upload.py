import sys
import os
import re
import json
import PyPDF2
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QTextEdit


class FileUploader(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.output = QTextEdit(self)
        self.output.setReadOnly(True)

        self.pdf_button = QPushButton("Upload PDF", self)
        self.pdf_button.clicked.connect(self.convert_pdf_to_text)
        layout.addWidget(self.pdf_button)

        self.txt_button = QPushButton("Upload Text File", self)
        self.txt_button.clicked.connect(self.upload_txtfile)
        layout.addWidget(self.txt_button)

        self.json_button = QPushButton("Upload JSON File", self)
        self.json_button.clicked.connect(self.upload_jsonfile)
        layout.addWidget(self.json_button)

        layout.addWidget(self.output)

        self.setLayout(layout)
        self.setWindowTitle("Upload .pdf, .txt, or .json")
        self.setGeometry(300, 300, 400, 200)

    def log_output(self, message):
        self.output.append(message)

    def process_text(self, text, overlap=200):
        text = re.sub(r'\s+', ' ', text).strip()
        sentences = re.split(r'(?<=[.!?]) +', text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 < 1000:  # +1 for the space
                current_chunk += (sentence + " ").strip()
            else:
                chunks.append(current_chunk.strip())
                current_chunk = current_chunk[-overlap:] + sentence + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        with open("temp.txt", "a", encoding="utf-8") as temp_file:
            for chunk in chunks:
                temp_file.write(chunk + "\n")
        self.log_output("Content appended to temp.txt with each chunk on a separate line.")

    def convert_pdf_to_text(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a PDF file", "", "PDF Files (*.pdf)")
        if file_path:
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                num_pages = len(pdf_reader.pages)
                text = ''
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    if page.extract_text():
                        text += page.extract_text() + " "
            self.process_text(text)

    def upload_txtfile(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a text file", "", "Text Files (*.txt)")
        if file_path:
            with open(file_path, 'r', encoding="utf-8") as txt_file:
                text = txt_file.read()
            self.process_text(text)

    def upload_jsonfile(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select a JSON file", "", "JSON Files (*.json)")
        if file_path:
            with open(file_path, 'r', encoding="utf-8") as json_file:
                data = json.load(json_file)
            text = json.dumps(data, ensure_ascii=False)
            self.process_text(text)


def main():
    app = QApplication(sys.argv)
    ex = FileUploader()
    ex.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
