import pickle
from langchain_core.tools import tool
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
model_path = os.path.join(MODEL_DIR, "cluster_model.pkl")

with open(model_path, "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    encoders = data["encoder"]
    cluster_labels = data["cluster_labels"]
    columns = data["columns"]

@tool
def predict_home_type(
    price:float,
    GrossSquareMeters: float,
    ItemStatus: str,
    room: float,
    hall: float,
    district: str
) -> dict:
    """
    Classifies the home type based on its features.

    Args:
        price (float): The price of the home. Must be a number only, without currency symbols or other characters.
        GrossSquareMeters (float): Size of the home in square meters. Must be a number only, without units like "m²".
        ItemStatus (str): Condition of the home. Must be either "Eşyalı" or "Boş".
        room (float): Number of rooms (e.g., in a 3+1 home, this is 3).
        hall (float): Number of halls (e.g., in a 3+1 home, this is 1).
        district (str): Name of any district in Istanbul. Must be lowercase and without Turkish characters.

    Returns:
        dict: {"home_type": str} — One of the following 4 classes:  
            - "Suitable Apartments for Middle-Income Families"  
            - "Luxurious and Spacious Residences for High-Income Groups"  
            - "Mid-Segment, Spacious and Economical Homes"  
            - "Multi-Room Ultra-Luxury Living Spaces"

    Example:
        For a furnished 3-room, 1-hall home of 160 m² priced at 3,000,000 TRY in any district of Istanbul:

        predict_home_type(
            price=3000000,
            GrossSquareMeters=160,
            ItemStatus="Eşyalı",
            room=3,
            hall=1,
            district="balat"  # District name should be lowercase and without Turkish characters
        )
    """
    try:
        input_data = {
            "district": district,
            "price": price,
            "GrossSquareMeters": GrossSquareMeters,
            "ItemStatus":ItemStatus,
            "room": room,
            "hall": hall,
        }

        # Base dataframe
        df = pd.DataFrame([input_data])


        # Diğer sayısal sütunları da ekle
        df["district"]=district
        df["price"] = price
        df["GrossSquareMeters"] = GrossSquareMeters
        df["room"] = room
        df["hall"] = hall
        
        for col in df.select_dtypes(include='object').columns:
            if col in encoders:
                encoder = encoders[col]
                unknown = set(df[col]) - set(encoder.classes_)
                if unknown:
                    raise ValueError(f"Unknown labels in column '{col}': {unknown}")
                df[col] = encoder.transform(df[col])

        # Tahmin yap
        clusters = model.predict(df)
        df['cluster'] = clusters[0]
        df['cluster_label'] = df['cluster'].map(cluster_labels)

        return {"home_type": str(df["cluster_label"].iloc[0])}

    except Exception as e:
        return {"error": str(e)}
