class ModelTypeMixin:
    @property
    def model_type(self):
        raise NotImplementedError
