from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Union
from agents import ask
from tools.home_price_tool import predict_home_price
from tools.anomaly_detection_tool import predict_anomalies
from tools.home_type_tool import predict_home_type
from tools.investment_tool import predict_investment
from tools.dataframe_tool import view_dataframe
import json
import warnings
warnings.filterwarnings("ignore")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "🎉 LangChain destekli FastAPI çalışıyor!"}

@app.get("/home_price")
def home_price(
    GrossSquareMeters: float = Query(..., description="Brüt metrekare"),
    ItemStatus: str = Query(..., description="İlan durumu (örneğin: sıfır, ikinci el)"),
    room: float = Query(..., description="Oda sayısı"),
    hall: float = Query(..., description="Salon sayısı"),
    district: str = Query(..., description="Semt/ilçe")
):
    input_data = {
        "GrossSquareMeters": GrossSquareMeters,
        "ItemStatus": ItemStatus,
        "room": room,
        "hall": hall,
        "district": district
    }
    return predict_home_price(input_data)

@app.get("/anomalies")
def anomalies(
    Price: float = Query(..., description="Fiyat"),
    GrossSquareMeters: float = Query(..., description="Brüt metrekare"),
    ItemStatus: str = Query(..., description="İlan durumu (örneğin: Eşyalı, Boş)"),
    room: float = Query(..., description="Oda sayısı"),
    hall: float = Query(..., description="Salon sayısı"),
    district: str = Query(..., description="Semt/ilçe")
):
    input_data = {
        "price":Price,
        "GrossSquareMeters": GrossSquareMeters,
        "ItemStatus":ItemStatus,
        "room": room,
        "hall": hall,
        "district": district
    }
    return predict_anomalies(input_data)

@app.get("/home_type")
def home_type(
    Price: float = Query(..., description="Fiyat"),
    GrossSquareMeters: float = Query(..., description="Brüt metrekare"),
    ItemStatus: str = Query(..., description="İlan durumu (örneğin: Eşyalı, Boş)"),
    room: float = Query(..., description="Oda sayısı"),
    hall: float = Query(..., description="Salon sayısı"),
    district: str = Query(..., description="Semt/ilçe")
):
    input_data = {
        "price":Price,
        "GrossSquareMeters": GrossSquareMeters,
        "ItemStatus":ItemStatus,
        "room": room,
        "hall": hall,
        "district": district
    }
    return predict_home_type(input_data)

@app.get("/is_eligible")
def is_eilgible(
    Price: float = Query(..., description="Fiyat"),
    GrossSquareMeters: float = Query(..., description="Brüt metrekare"),
    ItemStatus: str = Query(..., description="İlan durumu (örneğin: Eşyalı, Boş)"),
    room: float = Query(..., description="Oda sayısı"),
    hall: float = Query(..., description="Salon sayısı"),
    district: str = Query(..., description="Semt/ilçe")
):
    input_data = {
        "price":Price,
        "GrossSquareMeters": GrossSquareMeters,
        "ItemStatus":ItemStatus,
        "room": room,
        "hall": hall,
        "district": district
    }
    return predict_investment(input_data)


@app.get("/dataframe_check")
def dataframe_check(
    show_columns: Optional[List[str]] = Query(default=None),
    filter_condition: Optional[str] = None,
    sort_by: Optional[str] = None,
    ascending: bool = True,
    group_by: Optional[List[str]] = Query(default=None),
    aggregations: Optional[str] = None,
    show_head: Optional[int] = None,
    show_tail: Optional[int] = None,
    show_info: bool = False,
    show_na: bool = False,
    show_describe: bool = False
):
    aggregations_dict = None
    if aggregations:
        try:
            aggregations_dict = json.loads(aggregations)
        except json.JSONDecodeError:
            return {"error": "aggregations parametresi geçerli bir JSON olmalı. Örn: {\"salary\": \"mean\"}"}
    input_data = {
        "show_columns":show_columns,
        "filter_condition":filter_condition,
        "sort_by":sort_by,
        "ascending":ascending,
        "group_by":group_by,
        "aggregations":aggregations_dict,
        "show_head":show_head,
        "show_tail":show_tail,
        "show_info":show_info,
        "show_na":show_na,
        "show_describe":show_describe
    }
    return view_dataframe(input_data)


@app.get("/stream")
async def stream(question: str):
    async def event_stream():
        async for token in ask(question):
            print(token)
            yield json.dumps({"response": token}) + "\n"
    return StreamingResponse(event_stream(), media_type="application/json")
