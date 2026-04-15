import gradio as gr
import joblib
import numpy as np

# Load model
model = joblib.load("titanic_model.pkl")

def predict_survival(pclass, sex, age, fare):
    # Encode sex
    sex = 1 if sex == "male" else 0

    input_data = np.array([[pclass, sex, age, fare]])
    prediction = model.predict(input_data)[0]

    if prediction == 1:
        return "Survived ✅"
    else:
        return "Did Not Survive ❌"

# UI
interface = gr.Interface(
    fn=predict_survival,
    inputs=[
        gr.Dropdown([1, 2, 3], label="Passenger Class"),
        gr.Radio(["male", "female"], label="Sex"),
        gr.Slider(0, 80, label="Age"),
        gr.Slider(0, 500, label="Fare")
    ],
    outputs="text",
    title="🚢 Titanic Survival Predictor",
    description="Enter passenger details to predict survival"
)

interface.launch()