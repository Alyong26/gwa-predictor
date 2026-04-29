try:
    from api.main import app
except ModuleNotFoundError:
    from main import app  # type: ignore
