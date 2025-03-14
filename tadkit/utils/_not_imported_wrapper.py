class NotImportedWrapper:
    """Exception handler in case Confiance.AI dependencies are not available."""

    def __init__(self, exc):
        self.exc = exc

    def __getattribute__(self, __name):
        raise super().__getattribute__("exc")

    def __call__(self):
        raise super().__getattribute__("exc")
