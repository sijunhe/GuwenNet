import gradio as gr
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")
t5_config = T5Config(
    vocab_size=32128,
    d_model=768,
    d_kv=64,
    d_ff=2048,
    num_layers=12,
    num_decoder_layers=12,
    num_heads=12,
    relative_attention_num_buckets=32,
    dropout_rate=0.1,
    layer_norm_epsilon=1e-6,
    initializer_factor=1.0,
    feed_forward_proj="gated-gelu",
    is_encoder_decoder=True,
    use_cache=True,
    pad_token_id=0,
    eos_token_id=1,
    decoder_start_token_id=0,
    tie_word_embeddings=False,
    torch_dtype="float32",
    gradient_checkpointing=False)
model = T5ForConditionalGeneration(t5_config)
model.load_state_dict(torch.load("trained_model/20220327_kaggle/pytorch_model.bin", map_location=torch.device('cpu')))
model.eval()

def generate_classic(text):
    input_ids = tokenizer("转古文：" + text, return_tensors="pt").input_ids
    output_ids = model.generate(input_ids, num_beams=2, max_length=100)
    return tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

def launch_gradio():
    iface = gr.Interface(
        fn=generate_classic,
        inputs=gr.inputs.Textbox(lines=2, default="先帝开创的事业没有完成一半，却中途去世了。"),
        outputs="text")
    iface.launch(server_name="0.0.0.0")

if __name__ == '__main__':
    launch_gradio()
