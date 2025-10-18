# pip install onnx onnxruntime
import onnx
import onnxruntime as ort
from onnx import mapping

path = "rtmpose-m_256x192.onnx"

# Load raw ONNX
model = onnx.load(path)
graph = model.graph

print("IR version:", model.ir_version)
print("Opset imports:", [(op.domain or "ai.onnx", op.version) for op in model.opset_import])
print("Producer:", model.producer_name, model.producer_version)
print("Model metadata:", {p.key: p.value for p in model.metadata_props})
print("Graph doc_string:", graph.doc_string)

def show_vi(tag, vi):
    t = vi.type.tensor_type
    onnx_dtype = t.elem_type
    np_dtype = mapping.TENSOR_TYPE_TO_NP_TYPE.get(onnx_dtype, None)
    dims = [(d.dim_param if d.dim_param else d.dim_value) for d in t.shape.dim]
    print(f"[{tag}] name={vi.name}  onnx_dtype={onnx_dtype} np_dtype={np_dtype}  shape={dims}")

for vi in graph.input:
    show_vi("INPUT", vi)

for vi in graph.output:
    show_vi("OUTPUT", vi)

# onnxruntime view (often clearer for shapes with dynamic dims)
sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
print("\nORT inputs:")
for i in sess.get_inputs():
    print(dict(name=i.name, type=i.type, shape=i.shape))
print("ORT outputs:")
for o in sess.get_outputs():
    print(dict(name=o.name, type=o.type, shape=o.shape))
