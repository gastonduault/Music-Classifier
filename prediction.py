from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import librosa
import torch

# Genre mapping corrected to a dictionary
genre_mapping = {
    0: "Electronic",
    1: "Rock",
    2: "Punk",
    3: "Experimental",
    4: "Hip-Hop",
    5: "Folk",
    6: "Chiptune / Glitch",
    7: "Instrumental",
    8: "Pop",
    9: "International",
}

model = Wav2Vec2ForSequenceClassification.from_pretrained("gastonduault/music-classifier")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large")

# Function for preprocessing audio for prediction
def preprocess_audio(audio_path):
    audio_array, sampling_rate = librosa.load(audio_path, sr=16000)
    return feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt", padding=True)

def predict_audio(file_path):
    """
    Cette fonction prend en entrée le chemin d'un fichier audio et retourne une prédiction.
    """
    try:
        inputs = preprocess_audio(file_path)
        with torch.no_grad():
            logits = model(**inputs).logits
            predicted_class = torch.argmax(logits, dim=-1).item()
        return f"{genre_mapping[predicted_class]}"
    except Exception as e:
        raise RuntimeError(f"Error during audio prediction: {str(e)}")
