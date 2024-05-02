import numpy as np
import coremltools as ct
import coremltools.models as ctm
import coremltools.optimize.coreml as cto

model = ctm.MLModel("AneEmbeddings.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)

qconfig = cto.OptimizationConfig(cto.OpLinearQuantizerConfig(dtype=np.int8))
quantized_model = cto.linear_quantize_weights(model, config=qconfig)
quantized_model.save("AneEmbeddings.mlpackage")