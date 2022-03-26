# FastAPI starter for Machine Learning

This is a small project I made to solve an issue I had:

Users can submit different requests with multiple configurations and different arguments.
We want to cache said predictions when possible.

This project will:
* Use FastAPI to create a REST API
* Use SQLModel to store predictions based on the config and on the request.
* Use HuggingFace Transformers to run the prediction.

Hopefully, this can help others.

**DISCLAIMER**
I am aware that `model_runner.py` should be split in multiple "Task" objects. This is out of scope for this project.
If you want to use this project, let me know, and we can work on refactoring that part together. It won't be too hard.

**Future work**
1. Expand set of capabilities
   1. Work on entire datasets
   2. Uncertainty estimation
2. Progress bar on progress
3. Async routes

## How to use:

1. Run the app:
   1. `make DEVICE=cpu compose` (optionally `DEVICE=gpu` to use a GPU)
   2. go to 0.0.0.0:8080/docs
   3. Execute the `/predictions` API with different requests.
2. Run test
   1. `make test`
3. **DEV** Autoformatting
   1. `make black`


