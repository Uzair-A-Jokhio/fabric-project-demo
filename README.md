# Fabric Pattern Classifier API - Quick Start

This guide provides a short way to run the API for non-Python users, assuming Python is already installed.

1.  **Prepare Project Folder:**
    ```
     main.py
     models/
     color_model/
    ```

2.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    ```

3.  **Activate Virtual Environment:**
    * **Windows:**
      ```
      .\venv\Scripts\activate
      ```
    * **macOS/Linux:**
      ```
      source venv/bin/activate
      ```

4.  **Install Libraries:**
    ```bash
    pip install -r requirements.txt
    ```

5.  **Run FastAPI Server:**
    ```bash
    uvicorn main:app --reload
    ```

6.  **Use the API:**
    * **Welcome Page:** Open your browser to `http://localhost:8000/`.
    * **Documetation:** Use the Swagger UI (`http://localhost:8000/docs`) for API documenation.

**Keep the terminal window with the server running.**

7. **To Deactivate**
   ```
   deactivate
   ```
