from flask import Flask, send_file

app = Flask(__name__)

CSV_FILE_PATH = "C:\\Users\\samne\\OneDrive\\ADVAY\\Food Waste\\combined_detection_log.csv"

@app.route("/download_csv", methods=["GET"])
def download_csv():
    """Serves the CSV file to be downloaded."""
    try:
        return send_file(CSV_FILE_PATH, as_attachment=True)
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
