def get(name):
    import importlib
    return importlib.import_module("models.%s" % name).Network()

