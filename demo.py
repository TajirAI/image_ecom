import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from PIL import Image
import numpy as np
import easyocr
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load environment variables and the OpenAI API key
load_dotenv()

openai_api_key = os.getenv("GPT_API_KEY")

if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

def classify_question(question: str) -> str:
    prompt_template = """
You are an assistant that categorizes user queries regarding our ecommerce platform into one of the following categories:
1. General Query: Questions about products, features, pricing, etc.
2. Order Request: Requests to place an order.
3. Order Update: Inquiries regarding the status of an existing order.
Based on the user question provided below, reply with exactly one of the following words: "General Query", "Order Request", or "Order Update". Do not include any extra text.

User question: {question}
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["question"])
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0, openai_api_key=openai_api_key)
    response = model.call_as_llm(prompt.format(question=question))
    return response.strip()

def load_products_vectorstore(csv_path: str = "products.csv", vectorstore_path: str = "products_faiss_index"):
    if os.path.exists(vectorstore_path):
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        return vectorstore
    else:
        df = pd.read_csv(csv_path)
        docs = []
        for _, row in df.iterrows():
            content = (
                f"ID: {row['ID']}\n"
                f"Product: {row['Name']}\n"
                f"Price: {row['Price']}\n"
                f"Category: {row['Category']}\n"
                f"Description: {row['Description']}"
            )
            docs.append(Document(page_content=content))
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(docs, embedding=embeddings)
        vectorstore.save_local(vectorstore_path)
        return vectorstore

def process_general_query(question: str) -> str:
    vectorstore = load_products_vectorstore()
    retriever = vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(question)
    
    # Load CSV to get all unique product categories.
    df = pd.read_csv("products.csv")
    categories = df["Category"].unique().tolist()
    categories_str = ", ".join(str(category) for category in categories)
    
    # Retrieve conversation history from memory.
    memory_vars = st.session_state["memory"].load_memory_variables({})
    history = memory_vars.get("chat_history", "")
    
    prompt_template = """
You are a helpful assistant that uses the following conversation context along with product details to answer the customer's query.
If a product is available with us, please fetch its specifications from the internet and give them to the client. But if a product is not in the context,
DO NOT SHARE ITS SPECIFICATIONS; just say that it's not available.
If the user generally asks for products or categories, you should provide all product categories.
Conversation History:
{history}

Product Details:
{context}

All Product Categories: {categories}

Customer Query: {question}

Answer:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["history", "context", "question", "categories"])
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=openai_api_key)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    response = chain({
        "input_documents": docs, 
        "question": question, 
        "categories": categories_str,
        "history": history
    }, return_only_outputs=True)
    return response["output_text"]

def process_order_request(question: str) -> str:
    # Check for cancellation command.
    if question.strip().lower() in ["cancel", "cancel order", "stop"]:
         st.session_state["order_in_progress"] = False
         st.session_state["order_details"] = {}
         return "Order has been cancelled."
    
    # If order details are already confirmed, finalize and store the order.
    if st.session_state.get("order_details", {}).get("order_status") == "confirmed":
         confirmed_details = st.session_state["order_details"]
         order_info = {
             "Name": confirmed_details.get("Name"),
             "Address": confirmed_details.get("Address"),
             "Product": confirmed_details.get("Product"),
             "Quantity": confirmed_details.get("Quantity"),
             "Status": "received"
         }
         order_df = pd.DataFrame([order_info])
         if os.path.exists("orders.csv"):
             order_df.to_csv("orders.csv", mode='a', header=False, index=False)
         else:
             order_df.to_csv("orders.csv", index=False)
         st.session_state["order_in_progress"] = False
         st.session_state["order_details"] = {}
         return f"Order confirmed! Details: {order_info}"
    
    # Check if the current user input is a confirmation phrase.
    if question.strip().lower() in ["go ahead", "confirm", "okay thanks"]:
         required_keys = ["Name", "Address", "Product", "Quantity", "order_status"]
         if all(k in st.session_state.get("order_details", {}) for k in required_keys):
              confirmed_details = st.session_state["order_details"]
              if confirmed_details.get("order_status") == "confirmed":
                  order_info = {
                      "Name": confirmed_details.get("Name"),
                      "Address": confirmed_details.get("Address"),
                      "Product": confirmed_details.get("Product"),
                      "Quantity": confirmed_details.get("Quantity"),
                      "Status": "received"
                  }
                  order_df = pd.DataFrame([order_info])
                  if os.path.exists("orders.csv"):
                      order_df.to_csv("orders.csv", mode='a', header=False, index=False)
                  else:
                      order_df.to_csv("orders.csv", index=False)
                  st.session_state["order_in_progress"] = False
                  st.session_state["order_details"] = {}
                  return f"Order confirmed! Details: {order_info}"
         # If details are incomplete, we fall through to ask the LLM for more information.
    
    # Retrieve conversation history.
    memory_vars = st.session_state["memory"].load_memory_variables({})
    history = memory_vars.get("chat_history", "")
    
    prompt_template = """
You are an assistant that helps complete a customer's order interactively. Based on the conversation history below and the customer's latest input, extract the following order details if provided:
- Customer's Name
- Customer's Address
- Product Name (if already discussed in previous conversation, use that information)
- Quantity

If some details are missing, ask a follow-up question in a natural conversational manner to get that information.
If all details are provided, output a confirmation in JSON format with the following keys:
"Name": customer's name,
"Address": customer's address,
"Product": product name,
"Quantity": quantity,
"order_status": "confirmed"

If the customer types "cancel" at any point, cancel the order process.

Conversation History:
{history}

Current Customer Input:
{question}

Existing Order Details (if any):
{order_details}

Respond:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["history", "question", "order_details"])
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=openai_api_key)
    llm_response = model.call_as_llm(prompt.format(
        history=history,
        question=question,
        order_details=json.dumps(st.session_state.get("order_details", {}))
    ))
    
    try:
         response_json = json.loads(llm_response)
         if response_json.get("order_status") == "confirmed":
              order_info = {
                  "Name": response_json.get("Name", ""),
                  "Address": response_json.get("Address", ""),
                  "Product": response_json.get("Product", ""),
                  "Quantity": response_json.get("Quantity", ""),
                  "Status": "received"
              }
              order_df = pd.DataFrame([order_info])
              if os.path.exists("orders.csv"):
                  order_df.to_csv("orders.csv", mode='a', header=False, index=False)
              else:
                  order_df.to_csv("orders.csv", index=False)
              st.session_state["order_in_progress"] = False
              st.session_state["order_details"] = {}
              return f"Order confirmed! Details: {order_info}"
         else:
              st.session_state["order_details"].update(response_json)
              st.session_state["order_in_progress"] = True
              return llm_response
    except json.JSONDecodeError:
         st.session_state["order_in_progress"] = True
         return llm_response

def process_order_update(question: str) -> str:
    # Check for cancellation command.
    if question.strip().lower() in ["cancel", "cancel update", "stop"]:
         st.session_state["order_update_in_progress"] = False
         st.session_state["order_update_details"] = {}
         return "Order update process has been cancelled."

    # Retrieve conversation history.
    memory_vars = st.session_state["memory"].load_memory_variables({})
    history = memory_vars.get("chat_history", "")
    
    # Retrieve any existing update details from session.
    update_details = st.session_state.get("order_update_details", {})

    prompt_template = """
You are an assistant that helps customers check the status of their orders. Based on the conversation history below and the customer's latest input, extract the following detail if provided:
- Customer's Name

If the detail is missing, ask a follow-up question in a natural conversational manner to obtain the missing information.
If the required detail is provided, output a confirmation in JSON format with the following keys:
"Name": customer's name,
"lookup_status": "ready"

If the customer types "cancel" at any point, cancel the update process.

Conversation History:
{history}

Current Customer Input:
{question}

Existing Order Update Details (if any):
{update_details}

Respond:
"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["history", "question", "update_details"])
    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=openai_api_key)
    llm_response = model.call_as_llm(prompt.format(history=history, question=question, update_details=json.dumps(update_details)))
    
    try:
        response_json = json.loads(llm_response)
        if response_json.get("lookup_status") == "ready" and response_json.get("Name"):
            customer_name = response_json.get("Name").strip()
            # Read orders.csv to look up the order(s)
            if not os.path.exists("orders.csv"):
                st.session_state["order_update_in_progress"] = False
                st.session_state["order_update_details"] = {}
                return "No orders found in our records."
            orders_df = pd.read_csv("orders.csv")
            # Filter by name (case-insensitive)
            matched_orders = orders_df[orders_df["Name"].str.lower() == customer_name.lower()]
            if matched_orders.empty:
                st.session_state["order_update_in_progress"] = False
                st.session_state["order_update_details"] = {}
                return f"No orders found for the name {customer_name}."
            else:
                # Instead of returning the full order details, return only the status(es)
                statuses = matched_orders["Status"].unique().tolist()
                if len(matched_orders) == 1:
                    return f"Your order status is: {statuses[0]}"
                else:
                    status_message = ', '.join(statuses)
                    return f"Multiple orders found for {customer_name}. The statuses are: {status_message}"
        else:
            st.session_state["order_update_details"].update(response_json)
            st.session_state["order_update_in_progress"] = True
            return llm_response
    except json.JSONDecodeError:
        st.session_state["order_update_in_progress"] = True
        return llm_response

def process_user_query(question: str) -> str:
    category = classify_question(question)
    if category == "General Query":
        return process_general_query(question)
    elif category == "Order Request":
        return process_order_request(question)
    elif category == "Order Update":
        return process_order_update(question)
    else:
        return f"Unable to classify the question. Received classification: {category}"

def get_product_details_from_image(image_path, csv_path="products.csv"):
    """
    Extracts a product ID from an image using EasyOCR and looks up the corresponding product details in a CSV file.
    
    The CSV is expected to have the following columns:
    ID, Name, Price, Category, Description, Photo
    
    Parameters:
        image_path (str): The path to the image file.
        csv_path (str): The path to the CSV file containing product details.
        
    Returns:
        dict or None: A dictionary containing product details if a matching ID is found, otherwise None.
    """
    # Initialize the OCR reader and process the image.
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_path)
    
    # Find text starting with "Id:" and extract the number after it.
    id_text = None
    for bbox, text, confidence in results:
        if text.startswith("ID:"):
            id_text = text[3:].strip()  # Extract only the number after "Id:"
            break

    if not id_text:
        print("No ID detected in the image.")
        return None

    # Convert the extracted ID to an integer.
    try:
        product_id = int(id_text)
    except ValueError:
        print("Invalid product ID format:", id_text)
        return None

    # Load the CSV file.
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print("CSV file not found at:", csv_path)
        return None

    # Look for a matching product ID in the CSV.
    product_row = df[df["ID"] == product_id]
    if product_row.empty:
        print(f"Product with ID {product_id} not found in the CSV.")
        return None

    # Convert the found product details to a dictionary.
    product_details = product_row.to_dict(orient="records")[0]
    print("Product found:")
    for key, value in product_details.items():
        print(f"{key}: {value}")
    
    return product_details

def extract_text_from_image(image):
    # Convert the PIL image to a numpy array.
    image_np = np.array(image)
    # Initialize EasyOCR reader.
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image_np)
    # Return only the extracted text strings.
    return [text for bbox, text, confidence in results]

def lookup_product_details(extracted_texts, csv_path="products.csv"):
    product_id = None
    # Look for text starting with "ID:" and extract the number.
    for text in extracted_texts:
        if text.startswith("ID:"):
            id_text = text[3:].strip()
            try:
                product_id = int(id_text)
            except ValueError:
                st.error(f"Invalid product ID format: {id_text}")
            break

    if product_id is None:
        st.info("No product ID found in the image.")
        return None

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"CSV file not found at: {csv_path}")
        return None

    product_row = df[df["ID"] == product_id]
    if product_row.empty:
        st.warning(f"Product with ID {product_id} not found in the CSV.")
        return None

    # Convert the found row to a dictionary.
    return product_row.to_dict(orient="records")[0]

def main():
    st.set_page_config(page_title="TajirAI EcomAssist", layout="wide")
    st.header("TajirAI EcomAssist")
    
    # Chat input.
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="image_uploader", label_visibility="hidden")
    if uploaded_file is not None:
        print("uploaded_file", uploaded_file)
        # Open the image using PIL without displaying it.
        image = Image.open(uploaded_file)
        print("image", image)
        
        with st.spinner("Generating Response..."):
            extracted_texts = extract_text_from_image(image)
            product_details = lookup_product_details(extracted_texts)
        
        # Display the product details using the same chat message style as other responses.
        with st.chat_message("assistant"):
            if product_details:
                st.write("**Product Details**")
                for key, value in product_details.items():
                    st.write(f"**{key}:** {value}")
            else:
                st.write("Product details could not be retrieved.")
    
    # Initialize conversation memory if not already present.
    if "memory" not in st.session_state:
        st.session_state["memory"] = ConversationSummaryBufferMemory(
            llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5, openai_api_key=openai_api_key),
            memory_key="chat_history"
        )
    
    # Initialize chat messages list.
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    
    # Initialize order request state.
    if "order_in_progress" not in st.session_state:
        st.session_state["order_in_progress"] = False
    if "order_details" not in st.session_state:
        st.session_state["order_details"] = {}

    # Initialize order update state.
    if "order_update_in_progress" not in st.session_state:
        st.session_state["order_update_in_progress"] = False
    if "order_update_details" not in st.session_state:
        st.session_state["order_update_details"] = {}
    
    # Display previous chat messages.
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_query = st.chat_input(placeholder="Need help with shopping, orders, tracking?")
    if user_query:
        # Append and display user message.
        st.session_state["messages"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)
        
        # Process the user query.
        response = process_user_query(user_query)
        
        # Update conversation memory with the new interaction.
        st.session_state["memory"].save_context({"input": user_query}, {"output": response})
        
        # Append and display assistant response.
        st.session_state["messages"].append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)

if __name__ == "__main__":
    main()

