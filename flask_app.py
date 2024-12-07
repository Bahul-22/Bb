import os
from flask import Flask

import subprocess

app = Flask(__name__)

# Route to serve the Streamlit app
@app.route("/")
def streamlit_app():
    # Streamlit app will run as a subprocess
    streamlit_process = subprocess.Popen(
        ["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return (
        "Streamlit app is running! Visit "
        f"http://bbothps.pythonanywhere.com:8501 to access it."
    )


if __name__ == "__main__":
    app.run(debug=True)
