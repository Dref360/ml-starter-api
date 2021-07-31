import argparse

import uvicorn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8080)
    parser.add_argument('--debug', action='store_true')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    uvicorn.run(
        app="ml_starter_api.app_loader:app",
        host="0.0.0.0",
        port=int(args.port),
        reload=args.debug,
        debug=args.debug,
        workers=1
    )
