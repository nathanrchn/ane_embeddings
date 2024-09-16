import coremltools as ct
from shutil import copytree

coreml_model = ct.models.MLModel("ane-snowflake-arctic-embed-s.mlpackage")

compiled_model_path = coreml_model.get_compiled_model_path()
print(compiled_model_path)
copytree(compiled_model_path, "localy/ane-snowflake-arctic-embed-s/model.mlmodelc", dirs_exist_ok=True)