import gin.torch

def gin_get(param, default=None):
  try:
    return gin.query_parameter(param)
  except ValueError as e:
    if default is not None: return default
    raise
