import joblib
import pandas as pd

model = joblib.load("billet_model.pkl")

features = [
    "diagonal",
    "height_left",
    "height_right",
    "margin_low",
    "margin_up",
    "length"
]

mode = input("1 = single billet, 2 = CSV file : ")

if mode == "1":

    billet = pd.DataFrame([{
        "diagonal": float(input("diagonal: ")),
        "height_left": float(input("height_left: ")),
        "height_right": float(input("height_right: ")),
        "margin_low": float(input("margin_low: ")),
        "margin_up": float(input("margin_up: ")),
        "length": float(input("length: "))
    }])

    prediction = model.predict(billet)[0]

    print("Prediction :", prediction)


elif mode == "2":

    path = input("CSV file path: ").strip('"')

    data = pd.read_csv(path)

    predictions = model.predict(data[features])

    data["prediction"] = predictions

    print(data[["id", "prediction"]])

    data.to_csv("predictions.csv", index=False)

    print("Results saved in predictions.csv")