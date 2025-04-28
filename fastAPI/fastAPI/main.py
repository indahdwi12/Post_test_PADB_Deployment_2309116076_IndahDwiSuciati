from fastapi import FastAPI
from pydantic import BaseModel, Field
import pickle
import pandas as pd


# Inisialisasi FastAPI
app = FastAPI(title="Anxiety Prediction API (XGBoost + Scaler)")

# Load model dan scaler dari file pickle
with open("posttest.pkl", "rb") as f:
    saved_objects = pickle.load(f)
    model = saved_objects['model']
    scaler = saved_objects['scaler']

# Schema input sesuai semua kolom fitur (kecuali target 'Anxiety')
class InputData(BaseModel):
    Age: float
    Primary_streaming_service: int = Field(..., alias="Primary streaming service")
    Hours_per_day: float = Field(..., alias="Hours per day")
    While_working: int = Field(..., alias="While working")
    Instrumentalist: int
    Composer: int
    Fav_genre: int = Field(..., alias="Fav genre")
    Exploratory: int
    Foreign_languages: int = Field(..., alias="Foreign languages")
    BPM: float

    Frequency_Classic: int = Field(..., alias="Frequency [Classical]")
    Frequency_Country: int = Field(..., alias="Frequency [Country]")
    Frequency_EDM: int = Field(..., alias="Frequency [EDM]")
    Frequency_Folk: int = Field(..., alias="Frequency [Folk]")
    Frequency_Gospel: int = Field(..., alias="Frequency [Gospel]")
    Frequency_Hip_hop: int = Field(..., alias="Frequency [Hip hop]")
    Frequency_Jazz: int = Field(..., alias="Frequency [Jazz]")
    Frequency_K_pop: int = Field(..., alias="Frequency [K pop]")
    Frequency_Latin: int = Field(..., alias="Frequency [Latin]")
    Frequency_Lofi: int = Field(..., alias="Frequency [Lofi]")
    Frequency_Metal: int = Field(..., alias="Frequency [Metal]")
    Frequency_Pop: int = Field(..., alias="Frequency [Pop]")
    Frequency_RnB: int = Field(..., alias="Frequency [R&B]")
    Frequency_Rap: int = Field(..., alias="Frequency [Rap]")
    Frequency_Rock: int = Field(..., alias="Frequency [Rock]")
    Frequency_Video_game_music: int = Field(..., alias="Frequency [Video game music]")

    Depression: float
    Insomnia: float
    OCD: float
    Music_effects: int = Field(..., alias="Music effects")
    AgeCategory: int

    class Config:
        validate_by_name = True


# Preprocessing input
def preprocess_input(data: InputData):
    df = pd.DataFrame([data.dict(by_alias=True)])  # Pakai alias agar nama kolom cocok
    df_scaled = scaler.transform(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    return df_scaled

# Endpoint root
@app.get("/")
def read_root():
    return {"message": "✅ Anxiety Prediction API is running"}

# Endpoint prediksi
@app.post("/predict")
def predict_anxiety(data: InputData):
    processed = preprocess_input(data)
    prediction = model.predict(processed)[0]
    return {
        "predicted_anxiety_score": round(float(prediction), 2)
    }