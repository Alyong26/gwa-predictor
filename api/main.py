from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from api.modeling import (
        ARTIFACT_PATH,
        DATASET_PATH,
    )
except ModuleNotFoundError:
    from modeling import (  # type: ignore
        ARTIFACT_PATH,
        DATASET_PATH,
    )

try:
    from api.one_field_model import (
        ONE_FIELD_ARTIFACT_PATH,
        load_or_train_one_field_artifact,
    )
except ModuleNotFoundError:
    from one_field_model import (  # type: ignore
        ONE_FIELD_ARTIFACT_PATH,
        load_or_train_one_field_artifact,
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)


class PredictionInput(BaseModel):
    overall_gwa: float


one_field_artifact = load_or_train_one_field_artifact(ONE_FIELD_ARTIFACT_PATH, DATASET_PATH)
one_field_model = one_field_artifact["model"]
one_field_calibrator = one_field_artifact["calibrator"]
one_field_interval_radius = float(one_field_artifact["interval_radius"])
one_field_metrics = one_field_artifact["metrics"]


@app.get('/')
def home():
    return {
        'message': 'API Running',
        'dataset_path': str(DATASET_PATH),
        'artifact_path': str(ARTIFACT_PATH),
        'one_field_artifact_path': str(ONE_FIELD_ARTIFACT_PATH),
        'one_field_metrics': one_field_metrics,
    }


@app.post('/predict')
def predict(data: PredictionInput):
    if data.overall_gwa < 1.0 or data.overall_gwa > 3.0:
        raise HTTPException(
            status_code=400,
            detail='Please provide a GWA between 1.00 and 3.00.',
        )

    row = pd.DataFrame([{"current_gwa": float(data.overall_gwa)}])
    raw_prediction = float(one_field_model.predict(row)[0])
    predicted_gwa = float(one_field_calibrator.predict([raw_prediction])[0])
    lower_gwa = predicted_gwa - one_field_interval_radius
    upper_gwa = predicted_gwa + one_field_interval_radius

    predicted_gwa = max(1.0, min(5.0, predicted_gwa))
    lower_gwa = max(1.0, min(5.0, lower_gwa))
    upper_gwa = max(1.0, min(5.0, upper_gwa))

    return {
        'predicted_gwa': round(predicted_gwa, 2),
        'prediction_interval': {
            'confidence': 0.8,
            'lower_gwa': round(lower_gwa, 2),
            'upper_gwa': round(upper_gwa, 2),
        },
        'metrics': one_field_metrics,
        'input_fields': 1,
    }
