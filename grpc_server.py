# import grpc
# from concurrent import futures
# import asyncio

# # Import the generated protobuf files
# from protos.flood_prediction_pb2 import FloodPredictionRequest as GrpcFloodRequest
# from protos.flood_prediction_pb2 import FloodPredictionResponse as GrpcFloodResponse
# from protos.flood_prediction_pb2 import LocationsRequest, LocationsResponse
# from protos.flood_prediction_pb2_grpc import (
#     FloodPredictionServiceServicer,
#     add_FloodPredictionServiceServicer_to_server
# )

# # Import your existing FastAPI components
# from src.utils import logger
# from fastapi import HTTPException
# from flood_prediction_api import FloodPredictionRequest, locations, app

# class FloodPredictionServicer(FloodPredictionServiceServicer):
#     def __init__(self, fastapi_app):
#         """
#         Initialize with your FastAPI app
#         """
#         self.fastapi_app = fastapi_app

#     async def PredictFlood(self, request, context):
#         try:
#             # Convert gRPC request to your FastAPI request format
#             fastapi_request = FloodPredictionRequest(
#                 location=request.location
#             )
            
#             # Get your actual FastAPI endpoint function
#             fastapi_endpoint = self.fastapi_app.flood_prediction
            
#             # Call your existing FastAPI endpoint
#             result = await fastapi_endpoint(fastapi_request)
            
#             # Convert FastAPI response to gRPC response
#             return GrpcFloodResponse(
#                 current_flood_probability=float(result.current_flood_probability),
#                 forecast_flood_probability=float(result.forecast_flood_probability),
#                 estimated_rainfall=float(result.estimated_rainfall),
#                 daily_river_discharge=float(result.daily_river_discharge[0])
#             )
            
#         except HTTPException as e:
#             context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e.detail))
#         except Exception as e:
#             logger.error(f"Error in PredictFlood: {str(e)}")
#             context.abort(grpc.StatusCode.INTERNAL, str(e))

#     async def GetLocations(self, request, context):
#         try:
#             return LocationsResponse(locations=locations)
#         except Exception as e:
#             logger.error(f"Error in GetLocations: {str(e)}")
#             context.abort(grpc.StatusCode.INTERNAL, str(e))

# async def serve(fastapi_app, port: int = 50051):
#     """Start the gRPC server"""
#     server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
    
#     # Add your service to the gRPC server
#     add_FloodPredictionServiceServicer_to_server(
#         FloodPredictionServicer(fastapi_app), 
#         server
#     )
    
#     # Listen on port 50051
#     server.add_insecure_port(f'[::]:{port}')
#     await server.start()
#     logger.info(f"gRPC server running on port {port}")
#     await server.wait_for_termination()

# if __name__ == '__main__':
#     import uvicorn
    
#     # Run both servers
#     async def run_servers():
#         # Start gRPC server on port 50051
#         grpc_task = serve(app)
        
#         # Start FastAPI server on port 8000
#         fastapi_task = asyncio.create_task(
#             uvicorn.Server(
#                 config=uvicorn.Config(
#                     app=app, 
#                     host="0.0.0.0", 
#                     port=8000,
#                     loop="asyncio"
#                 )
#             ).serve()
#         )
        
#         # Run both servers concurrently
#         await asyncio.gather(grpc_task, fastapi_task)
    
#     # Start everything
#     asyncio.run(run_servers())
import grpc
from concurrent import futures
import asyncio
import uvicorn
from watchfiles import awatch

from protos.flood_prediction_pb2 import FloodPredictionRequest as GrpcFloodRequest
from protos.flood_prediction_pb2 import FloodPredictionResponse as GrpcFloodResponse
from protos.flood_prediction_pb2 import LocationsRequest, LocationsResponse
from protos.flood_prediction_pb2_grpc import (
    FloodPredictionServiceServicer,
    add_FloodPredictionServiceServicer_to_server
)

from src.utils import logger
from fastapi import HTTPException
from flood_prediction_api import (
    flood_prediction,
    FloodPredictionRequest,
    locations,
    app
)

class FloodPredictionServicer(FloodPredictionServiceServicer):
    def __init__(self, fastapi_app):
        self.fastapi_app = fastapi_app
        self.flood_prediction = flood_prediction

    async def PredictFlood(self, request, context):
        try:
            # Log incoming request
            logger.info(f"Received prediction request for location: {request.location}")
            
            # Create FastAPI request
            fastapi_request = FloodPredictionRequest(location=request.location)
            logger.info(f"Created FastAPI request: {fastapi_request}")
            
            # Call FastAPI endpoint
            result = await self.flood_prediction(fastapi_request)
            logger.info(f"FastAPI result type: {type(result)}")
            logger.info(f"FastAPI result: {result}")
            logger.info(f"FastAPI result attributes: {dir(result)}")
            
            # Log individual fields
            logger.info(f"current_flood_probability: {result.current_flood_probability} (type: {type(result.current_flood_probability)})")
            logger.info(f"forecast_flood_probability: {result.forecast_flood_probability} (type: {type(result.forecast_flood_probability)})")
            logger.info(f"estimated_rainfall: {result.estimated_rainfall} (type: {type(result.estimated_rainfall)})")
            logger.info(f"daily_river_discharge: {result.daily_river_discharge} (type: {type(result.daily_river_discharge)})")
            
            # Create gRPC response
            try:
                response = GrpcFloodResponse(
                    current_flood_probability=float(result.current_flood_probability),
                    forecast_flood_probability=float(result.forecast_flood_probability),
                    estimated_rainfall=float(result.estimated_rainfall),
                    daily_river_discharge=float(result.daily_river_discharge)  # Removed [0] indexing
                )
                logger.info(f"Created gRPC response successfully: {response}")
                return response
            except Exception as e:
                logger.error(f"Error creating gRPC response: {str(e)}")
                raise
            
        except HTTPException as e:
            logger.error(f"HTTP Exception: {str(e.detail)}")
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e.detail))
        except Exception as e:
            logger.error(f"Error in PredictFlood: {str(e)}, {type(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))
            return None

    async def GetLocations(self, request, context):
        try:
            return LocationsResponse(locations=locations)
        except Exception as e:
            logger.error(f"Error in GetLocations: {str(e)}")
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

class GRPCServer:
    def __init__(self, app, port=50051):
        self.app = app
        self.port = port
        self.server = None

    async def start(self):
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
        add_FloodPredictionServiceServicer_to_server(
            FloodPredictionServicer(self.app), 
            self.server
        )
        self.server.add_insecure_port(f'[::]:{self.port}')
        await self.server.start()
        logger.info(f"gRPC server running on port {self.port}")

    async def stop(self):
        if self.server:
            await self.server.stop(0)
            logger.info("gRPC server stopped")

async def run_servers_with_reload():
    grpc_server = GRPCServer(app)
    await grpc_server.start()
    
    config = uvicorn.Config(
        app=app, 
        host="0.0.0.0", 
        port=8000, 
        loop="asyncio", 
        reload=True
    )
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == '__main__':
    asyncio.run(run_servers_with_reload())