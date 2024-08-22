### **Functionality Testing Report**

#### **Overview**

This report details the functionality testing of the Real Estate QnA system. The tests covered various data types, including PDFs, Excel sheets, DOCX documents, and images. The tests focused on the system's ability to correctly process and retrieve relevant information based on the user's queries.

#### **Test Cases**

1. **PDF Processing**  
   * **Test Query**: "What are the property tax rates for owner-occupied residential properties?"  
   * **Expected Outcome**: The system should accurately retrieve and display the property tax rates relevant to owner-occupied residential properties from the provided PDF files.  
   * **Result**: The test passed successfully, with the system correctly identifying and extracting the relevant tax rate information.  
2. **Excel Processing**  
   * **Test Query**: "What is the total quantity of items across all inventory sheets?"  
   * **Expected Outcome**: The system should sum the quantities of items across all sheets and provide the total.  
   * **Result**: The test passed, and the system accurately calculated the total quantity of items across the inventory sheets.  
3. **DOCX Processing**  
   * **Test Query**: "What is the lead registration policy in the Sales SOP and policies document?"  
   * **Expected Outcome**: The system should correctly identify and return the lead registration policy from the provided DOCX document.  
   * **Result**: The test passed, with the system successfully retrieving the correct information regarding the lead registration policy.  
4. **Image Processing**  
   * **Test Query**: "What is the size and layout of the 1-bedroom \+ study unit in Tembusu Grand?"  
   * **Expected Outcome**: The system should extract and provide details about the size and layout of the 1-bedroom \+ study unit from the images provided.  
   * **Result**: The test passed, with the system correctly processing the images and extracting the relevant layout details.

#### 

#### 

#### **Observations**

* **Accuracy**: The system demonstrated average accuracy in extracting relevant information from the provided data types. It successfully handled different formats, correctly interpreting and responding to user queries.  
* **OCR Performance**: The image processing, relying on OCR (Tesseract), worked well but had occasional challenges with low-quality images or images with complex layouts. The performance could vary based on image quality.  
* **Model Limitations**:  
  * **Text Length**: The LLaMA model, like other language models, has a token limit that can affect processing long documents or large chunks of data. Some responses may be truncated or incomplete if the input exceeds the model's capacity.  
  * **Contextual Understanding**: While LLaMA performs well on direct queries, it might struggle with more nuanced or context-dependent questions, especially if the required information is scattered across multiple documents.

#### 

#### **Performance Analysis**

* **Memory and Processing Time**: The system handled moderate-sized files efficiently but could experience delays with very large documents or images. Optimizing memory usage or implementing more advanced chunking techniques could improve performance. The delay in time of response is also contributed by the level of hardware that I have in my system, so if we have better hardware the time of response will be better.

#### 

#### **Potential Improvements**

* **Advanced Models**:  
  * **GPT-4 or T5 Models**: Using models like GPT-4 or T5 could enhance contextual understanding, especially for complex queries. These models generally have better handling of longer contexts and can provide more accurate results.  
  * **Specialized OCR Engines**: Integrating more advanced OCR engines or using pre-trained models on specific data (e.g., architectural drawings) could improve accuracy in image processing.  
* **Data Retrieval Techniques**:  
  * **Hybrid Retrieval Methods**: Implementing a combination of keyword-based and vector-based retrieval methods could improve the system's ability to handle a wider range of queries, particularly those that require more nuanced understanding.  
  * **Document Chunking**: Refining the chunking strategies, especially for large documents, could help in efficiently utilizing the model's token limit and improving the accuracy of responses.

#### **Conclusion**

The Real Estate QnA system performed well across all test cases, successfully extracting and returning relevant information from PDFs, DOCX files, images, and Excel sheets. The identified limitations suggest opportunities for future enhancement, particularly in handling longer contexts and improving image processing accuracy. The system's overall performance indicates that it is robust and capable of meeting the requirements of the assignment, with potential for further refinement.

