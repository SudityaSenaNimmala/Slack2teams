# PDF Knowledge Base

This directory contains PDF files that will be used as additional knowledge sources for the CF-Chatbot.

## How to Add PDFs

1. Simply place your PDF files in this directory
2. The system will automatically process them when the application starts
3. Supported formats: `.pdf`

## Processing Details

- PDFs are processed using both PyMuPDF and PyPDF2 for maximum text extraction coverage
- Text is split into chunks of 1000 characters with 200 character overlap
- Each PDF chunk includes metadata about its source file
- PDFs are combined with the existing CloudFuze blog content in the vector database

## File Organization

- Place all PDF files directly in this directory
- Subdirectories are not currently supported
- The system will process all `.pdf` files found in this directory

## Troubleshooting

If a PDF fails to process:
- Check that the PDF is not password-protected
- Ensure the PDF contains extractable text (not just images)
- Check the console output for specific error messages

## Note

The system will automatically detect if this directory exists and contains PDF files. If no PDFs are found, it will fall back to using only the web content.
