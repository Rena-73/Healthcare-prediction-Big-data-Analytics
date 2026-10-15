HEALTHCARE BIG DATA ANALYTICS — START HERE
==========================================

This is a beginner-friendly Windows project using:
- Python
- Apache Spark / PySpark (local mode)
- Spark MLlib (Logistic Regression)
- Streamlit dashboard
- Pandas and Plotly

IMPORTANT ABOUT HADOOP:
This ZIP runs Spark locally on your Windows laptop. It does NOT start a Hadoop HDFS cluster.
The project stores the source CSV and outputs as local Windows files. If your instructor explicitly requires HDFS, ask before submitting: HDFS needs a separate Hadoop setup. The project title mentions Hadoop and Spark, but this easy version demonstrates Spark and MLlib without a complicated HDFS installation.

WHAT YOU NEED:
1. Windows 10/11
2. Internet to install packages and download the dataset
3. Java JDK 17
4. Python 3.11 (64-bit)
5. VS Code (recommended)

AFTER EXTRACTING:
1. Install Java 17: https://adoptium.net/temurin/releases/?version=17
   During setup, enable options to set JAVA_HOME / update PATH if offered.
2. Install Python 3.11: https://www.python.org/downloads/release/python-3119/
   IMPORTANT: tick "Add python.exe to PATH".
3. Install VS Code: https://code.visualstudio.com/download
4. Download the dataset from:
   https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008
   Download/extract the dataset and copy diabetic_data.csv into this project's data folder.
   The final path must be:
   healthcare-bigdata-project\data\diabetic_data.csv
5. Open this project folder in VS Code.
6. Open Terminal > New Terminal.
7. Run these commands one by one:

   python --version
   java -version
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install --upgrade pip
   pip install -r requirements.txt

8. Run the complete processing and model training pipeline:

   python run_pipeline.py

   Wait until it finishes. The first Spark run may take several minutes.
   The generated files will be saved under artifacts.

9. Start the dashboard:

   .venv\Scripts\activate
   streamlit run app.py

10. Open the local URL shown in the terminal (usually http://localhost:8501).

EASIER OPTION:
After Python and Java are installed and the dataset is copied into data, you can double-click:
- setup_windows.bat (installs Python packages)
- run_pipeline.bat (processes data and trains the model)
- run_dashboard.bat (opens the dashboard)
If Windows asks whether to allow the script, only run it if you understand and trust the files. You can inspect all scripts in VS Code first.

WHAT THE PROJECT DOES:
- Reads historical hospital encounters from CSV using Spark.
- Cleans '?' missing-value markers and removes exact duplicate rows.
- Creates a target: readmitted within 30 days (<30) versus other valid outcomes (>30, NO).
- Uses Spark SQL for a few summary queries.
- Trains a Spark MLlib Logistic Regression model.
- Evaluates accuracy, precision, recall, F1, and ROC-AUC.
- Creates summary CSV files and an interactive Streamlit dashboard.

ACADEMIC / HEALTH NOTICE:
This is a learning prototype based on historical data, not a clinically validated medical tool. Do not use it for diagnosis or real patient-care decisions.

TROUBLESHOOTING:
- "python is not recognized": reinstall Python and tick Add Python to PATH, then reopen VS Code.
- "JAVA_HOME is not set": install Java 17 and set JAVA_HOME to the installed JDK folder, then restart VS Code.
- "No module named pyspark/streamlit": activate .venv and run pip install -r requirements.txt.
- "File not found": check that data\diabetic_data.csv exists exactly at that path.
- Spark/Java error: confirm java -version reports Java 17, close and reopen VS Code, then try again.
- The first model run can be slow and use memory. Close heavy applications while it runs.
- If Spark shows a Windows winutils/Hadoop warning, it may be a warning rather than a failure. If the script actually stops, save the complete error and ask for help.

DATA PRIVACY:
Use only the public educational dataset included in the UCI repository. Do not add real patient data.
