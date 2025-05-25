from flask import Flask, request, jsonify
from flask_cors import CORS

import pickle
import pandas as pd
import numpy as np

# Cargar el modelo de regresión entrenado
with open("xgb_reg_model2.pkl", "rb") as file:
    reg_model = pickle.load(file)

# Cargar el modelo de clasificación para predecir 'Above_Goal'
with open("xgb_classifier_model.pkl", "rb") as file:
    clf_model = pickle.load(file)

# Columnas esperadas por el modelo de regresión
expected_columns = [
    'NIVELSOCIOECONOMICO_DES', 'MTS2VENTAS_NUM', 'PUERTASREFRIG_NUM',
    'CAJONESESTACIONAMIENTO_NUM', 'LATITUD_NUM', 'LONGITUD_NUM',
    'ENTORNO_DES_Base', 'ENTORNO_DES_Hogar', 'ENTORNO_DES_Peatonal',
    'ENTORNO_DES_Receso', 'SEGMENTO_MAESTRO_DESC_Barrio Competido',
    'SEGMENTO_MAESTRO_DESC_Clásico', 'SEGMENTO_MAESTRO_DESC_Hogar Reunión',
    'SEGMENTO_MAESTRO_DESC_Oficinistas', 'SEGMENTO_MAESTRO_DESC_Parada Técnica',
    'LID_UBICACION_TIENDA_UT_CARRETERA_GAS', 'LID_UBICACION_TIENDA_UT_DENSIDAD',
    'LID_UBICACION_TIENDA_UT_GAS_URBANA', 'Above_Goal'
]

def predecir_venta_total(user_input):
    """
    user_input: dict sin la clave 'Above_Goal'
    """

    df = pd.DataFrame([user_input])

    # Variables categóricas para dummies
    df_cat = pd.get_dummies(df, columns=['ENTORNO_DES', 'SEGMENTO_MAESTRO_DESC', 'LID_UBICACION_TIENDA_UT'])

    # Columnas dummy esperadas por el modelo de clasificación
    clf_dummy_cols = [col for col in clf_model.get_booster().feature_names if col not in df.columns]

    for col in clf_dummy_cols:
        if col not in df_cat.columns:
            df_cat[col] = 0

    # Alinear columnas del clasificador
    clf_features = clf_model.get_booster().feature_names
    df_clf = df_cat.reindex(columns=clf_features, fill_value=0)

    # Predecir Above_Goal
    above_goal_pred = clf_model.predict(df_clf)[0]

    # Insertar columna en el dataframe original
    df_cat["Above_Goal"] = above_goal_pred

    # Asegurar todas las columnas para el modelo de regresión
    for col in expected_columns:
        if col not in df_cat.columns:
            df_cat[col] = 0

    df_cat = df_cat[expected_columns]

    # Predicción de ventas
    pred = reg_model.predict(df_cat)[0]

    return {
        "venta_total_estimado": float(round(pred, 2)), # Diezmiles de pesos
        "Above_Goal_calculado": int(above_goal_pred)
    }

# Ejemplo de uso
if __name__ == "__main__":
    sample_input = {
        'NIVELSOCIOECONOMICO_DES': 3,
        'MTS2VENTAS_NUM': 2,
        'PUERTASREFRIG_NUM': 4,
        'CAJONESESTACIONAMIENTO_NUM': 2,
        'LATITUD_NUM': 20.67,
        'LONGITUD_NUM': -100.31,
        'ENTORNO_DES': 'Base',
        'SEGMENTO_MAESTRO_DESC': 'Parada Técnica',
        'LID_UBICACION_TIENDA_UT': 'UT_GAS_URBANA'
    }

    resultado = predecir_venta_total(sample_input)
    print(resultado)

# Configuración de Flask
app = Flask(__name__)
CORS(app)
@app.route('/predictSales', methods=['POST'])
def predict_sales():
    data = request.get_json()

    # Validar que la clave 'Above_Goal' no esté en el input
    if 'Above_Goal' in data:
        return jsonify({"error": "No se debe incluir la clave 'Above_Goal' en el input."}), 400

    try:
        result = predecir_venta_total(data)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500