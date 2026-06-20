import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# Create FastAPI app
app = FastAPI()


# -----------------------
# 1. Train ML Model
# -----------------------

df = pd.read_csv("house_data.csv")

X = df.drop("price", axis=1)
y = df["price"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = LinearRegression()

model.fit(
    X_train,
    y_train
)


predictions = model.predict(X_test)

error = mean_absolute_error(
    y_test,
    predictions
)

print(f"Mean absolute error: {error}")


# -----------------------
# 2. Create Input Format
# -----------------------

class HouseData(BaseModel):

    area: int
    bedrooms: int
    bathrooms: int



# -----------------------
# 3. Home API
# -----------------------

@app.get("/")
def home():

    return {
        "message": "House Price Prediction API"
    }



# -----------------------
# 4. Prediction API
# -----------------------

@app.post("/predict")
def predict(data: HouseData):

    input_data = pd.DataFrame(
        [
            [
                data.area,
                data.bedrooms,
                data.bathrooms
            ]
        ],
        columns=[
            "area",
            "bedrooms",
            "bathrooms"
        ]
    )


    price = model.predict(input_data)


    return {

        "predicted_price": int(price[0])

    }