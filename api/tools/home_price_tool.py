import pickle
from langchain_core.tools import tool
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
model_path = os.path.join(MODEL_DIR, "home_price_model.pkl")

with open(model_path, "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]
    columns = data["columns"]

@tool
def predict_home_price(
    GrossSquareMeters: float,
    ItemStatus: str,
    room: float,
    hall: float,
    district: str
) -> dict:
    """
    Predicts the home price based on input features.

    Args:
        GrossSquareMeters (float): The size of the home in square meters. Must be a number only, without units like "m²".
        ItemStatus (str): Condition of the home. Must be either "Eşyalı" or "Boş".
        room (float): Number of rooms (e.g., in a 3+1 home, this is 3).
        hall (float): Number of halls (e.g., in a 3+1 home, this is 1).
        district (str): Name of any district in Istanbul. Must be lowercase and without Turkish characters.

    Returns:
        dict: {"predicted_price": float} — The estimated home price.

    Example:
        For a furnished 3+1 home of 160 m² in any district of Istanbul:

        predict_home_price(
            GrossSquareMeters=160,
            ItemStatus="Eşyalı",
            room=3,
            hall=1,
            district="arnavutkoy"  # District name in lowercase and without Turkish characters
        )
    """

    try:
        input_data = {
            "GrossSquareMeters": GrossSquareMeters,
            "room": room,
            "hall": hall
        }

        # Base dataframe
        df = pd.DataFrame([input_data])

        # Boş tüm sütunları 0 ile başlat
        for col in columns:
            df[col] = False

        # Manuel olarak ilgili one-hot sütununu 1 yap
        district_col = f"district_{district}"
        status_col = f"ItemStatus_{ItemStatus}"

        if district_col in columns:
            df[district_col] = True
        if status_col in columns:
            df[status_col] = True

        # Diğer sayısal sütunları da ekle
        df["GrossSquareMeters"] = GrossSquareMeters
        df["room"] = room
        df["hall"] = hall

        # Ölçekle
        df_scaled = scaler.transform(df)

        # Tahmin yap
        prediction = model.predict(df_scaled)

        return {"predicted_price": float(prediction)}

    except Exception as e:
        return {"error": str(e)}
