import streamlit as st
from PIL import Image
import pytesseract
from pytesseract import Output
import os
import cv2
import torch
import tempfile
import numpy as np
import fitz  # PyMuPDF

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#@st.cache(allow_output_mutation=True)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable CUDA

# Set the device to CPU
device = torch.device('cpu')

def add_bg_from_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background-image: url("https://www.linkpicture.com/q/DP_for_Site.png");
             background-attachment: fixed;
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

def perform_ocr(image):
    # Perform OCR using Tesseract
    image = np.array(image)
    
    # Convert image to RGB if it's not already
    if len(image.shape) == 2:  # If grayscale
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # If RGBA
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    extracted_data = pytesseract.image_to_data(image, output_type=Output.DICT)
    
    # Get bounding box coordinates and text
    bounding_boxes = extracted_data["level"]
    texts = extracted_data["text"]
    
    # Visualize the text with bounding boxes
    image_with_boxes = image.copy()
    for i in range(len(bounding_boxes)):
        if bounding_boxes[i] == 5 and texts[i].strip():  # Only process non-empty text
            (x, y, w, h) = (
                extracted_data["left"][i],
                extracted_data["top"][i],
                extracted_data["width"][i],
                extracted_data["height"][i]
            )
            cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image_with_boxes, texts[i], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return image_with_boxes

def process_pdf(pdf_path):
    try:
        # Open the PDF
        pdf_document = fitz.open(pdf_path)
        images = []
        
        # Convert each page to an image
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        
        pdf_document.close()
        return images
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

def main():
    add_bg_from_url() 
    st.title("OCR with Bounding Box Visualization")
    st.write("Upload a PNG or PDF document to perform OCR and visualize the results.")
    
    uploaded_file = st.file_uploader("Upload Document", type=["png", "pdf"])
    
    st.markdown(
        """
        <style>
        body {
            color: white;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    if uploaded_file is not None:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        
        if file_extension == ".png":
            try:
                # Read and display the image
                print(uploaded_file.name)
                
                file_name = uploaded_file.name
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Perform OCR
                ocr_result = perform_ocr(image)
                original_title = '<p style="font-family:sans-serif; color:White; font-size: 35px;"><b>OCR Results Bounding Boxes</b></p>'
                st.markdown(original_title, unsafe_allow_html=True)
                st.image(ocr_result, use_column_width=True)
                
                # Display extracted text
                text = pytesseract.image_to_string(image)
                st.markdown('<p style="font-family:sans-serif; color:White; font-size: 35px;"><b>Extracted Text:</b></p>', unsafe_allow_html=True)
                st.text(text)
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
        
        elif file_extension == ".pdf":
            try:
                # Convert PDF to images
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                    temp_path = temp_file.name
                    temp_file.write(uploaded_file.read())
                
                images = process_pdf(temp_path)
                
                if images:
                    for i, image in enumerate(images):
                        st.image(image, caption=f"Page {i+1}", use_column_width=True)
                        
                        # Perform OCR and visualize with bounding boxes
                        ocr_result = perform_ocr(image)
                        original_title = '<p style="font-family:sans-serif; color:White; font-size: 35px;"><b>OCR Results Bounding Boxes</b></p>'
                        st.markdown(original_title, unsafe_allow_html=True)
                        st.image(ocr_result, use_column_width=True)
                        
                        # Display extracted text
                        text = pytesseract.image_to_string(image)
                        st.markdown(f'<p style="font-family:sans-serif; color:White; font-size: 35px;"><b>Extracted Text (Page {i+1}):</b></p>', unsafe_allow_html=True)
                        st.text(text)
                
                # Clean up temporary file
                os.unlink(temp_path)
                
            except Exception as e:
                st.error(f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    main()