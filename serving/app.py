import gradio as gr
import torch, onnxruntime
from transformers import T5Tokenizer
from fastT5 import get_onnx_model

tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")
model = get_onnx_model("trained_model/GuwenNet", "trained_model/GuwenNet-quantized")

def generate_classic(text):
    input_ids = tokenizer("转古文：" + text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, max_length=100)
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

def launch_gradio():
    iface = gr.Interface(
        fn=generate_classic,
        inputs=gr.inputs.Textbox(lines=2, default="先帝开创的事业没有完成一半，却中途去世了。现在天下分裂成三个国家。蜀汉民力困乏，这实在是危急存亡的时候啊。"),
        outputs="text")
    iface.launch(server_name="0.0.0.0")

if __name__ == '__main__':
    launch_gradio()
