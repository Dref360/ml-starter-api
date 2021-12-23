# FastAPI starter for Machine Learning

This is a small project I made to solve an issue I had:

Users can submit different requests with multiple configurations and different arguments.
We want to cache said predictions when possible.

This project will:
* Use FastAPI to create a clean API
* Use SQLModel to store predictions based on the config and on the request.
* Use HuggingFace Transformers to run the prediction.

Hopefully, this can help others.

## How to use:

1. Run the app:
   1. `make compose`
   2. go to 0.0.0.0:8080/docs
   3. Execute the `/predictions` API with different requests.
2. Run test
   1. `make test`
3. **DEV** Autoformatting
   1. `make black`


