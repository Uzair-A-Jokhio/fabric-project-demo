# Fabric Pattern Classifier API - Quick Start

This guide provides a short way to run the API for non-Python users, assuming Python is already installed.

1.  **Prepare Project Folder:**
    * Create a folder (e.g., `fabric_api`).
    * Inside, create a folder named `models`.
    * Place your model file (`demo_3_classifier.h5`) in `models/`.
    * Save the API Python code (provided earlier) as `main.py` in `fabric_api/`.

    ```
     main.py
     models/
        └── demo_3_classifier.h5
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
    * **Prediction:** Use the provided HTML frontend (update `API_URL` to `http://localhost:8000/predict/`) to upload images and get predictions.

**Keep the terminal window with the server running.**

7. **To Deactivate**
   ```
   deactivate
   ```
