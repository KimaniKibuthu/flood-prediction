import sys
import os
import grpc
import asyncio

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from protos
from protos import flood_prediction_pb2
from protos import flood_prediction_pb2_grpc

async def test_grpc_endpoints():
    # Create a gRPC channel
    async with grpc.aio.insecure_channel('localhost:50051') as channel:
        # Create a stub (client)
        stub = flood_prediction_pb2_grpc.FloodPredictionServiceStub(channel)
        
        try:
            # Test GetLocations
            print("\nTesting GetLocations...")
            locations_request = flood_prediction_pb2.LocationsRequest()
            locations_response = await stub.GetLocations(locations_request)
            print("Available locations:", locations_response.locations)

            # Test PredictFlood
            print("\nTesting PredictFlood...")
            predict_request = flood_prediction_pb2.FloodPredictionRequest(
                location="Dhaka"  # Using Dhaka as an example
            )
            prediction_response = await stub.PredictFlood(predict_request)
            print("Prediction Results:")
            print(f"Current Flood Probability: {prediction_response.current_flood_probability}")
            print(f"Forecast Flood Probability: {prediction_response.forecast_flood_probability}")
            print(f"Estimated Rainfall: {prediction_response.estimated_rainfall}")
            print(f"Daily River Discharge: {prediction_response.daily_river_discharge}")

        except grpc.RpcError as e:
            print(f"gRPC Error: {e.details()}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_grpc_endpoints())