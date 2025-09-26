from bus_eta_model import predict_eta_from_api_data

origin = "MG Road, Bangalore"
destination = "Whitefield, Bangalore"

prediction = predict_eta_from_api_data(origin, destination)

print("\n=== API Data ===")
print(prediction["api_data"])

print("\n=== ETA Prediction ===")
print(prediction["eta_prediction"])