from flask import Flask, request, render_template, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from prediction import predict_audio  # Importez la fonction de prédiction

app = Flask(__name__)
app.secret_key = "your_secret_key"  # Nécessaire pour les messages flash
UPLOAD_FOLDER = "uploads"  # Dossier pour stocker les fichiers uploadés
ALLOWED_EXTENSIONS = {"wav"}

# Configuration
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # Limite à 200 Mo
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Vérifier si le fichier est valide
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
import logging

logging.basicConfig(level=logging.DEBUG)
@app.route("/", methods=["GET", "POST"])
def upload_file():
    try:
        if request.method == "POST":
            logging.debug("POST request received.")
            # Vérifie si un fichier a été uploadé
            if "file" not in request.files:
                flash("Aucun fichier sélectionné.")
                return redirect(request.url)

            file = request.files["file"]

            if file.filename == "":
                flash("Aucun fichier sélectionné.")
                return redirect(request.url)

            if file and allowed_file(file.filename):
                # Sauvegarder le fichier uploadé
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(file_path)
                logging.debug(f"File saved at: {file_path}")

                # Appeler la fonction de prédiction
                try:
                    prediction_result = predict_audio(file_path)  # Appelez votre fonction
                    logging.debug(f"Prediction result: {prediction_result}")
                finally:
                    # Supprimez le fichier après la prédiction
                    try:
                        os.remove(file_path)
                        logging.debug(f"File removed: {file_path}")
                    except Exception as e:
                        logging.error(f"Failed to delete file: {file_path}, error: {e}")

                return render_template("upload.html", result=prediction_result)

        return render_template("upload.html", result=None)
    except Exception as e:
        logging.error(f"Unexpected server error: {str(e)}")
        return "Une erreur inattendue est survenue.", 500

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
