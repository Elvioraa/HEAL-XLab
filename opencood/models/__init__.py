def build_model(hypes):
    """Compatibility wrapper around the repository's dynamic model builder."""
    from opencood.tools.train_utils import create_model

    return create_model(hypes)
