import os 
import time
import torch
import pynvml
import requests
import numpy as np
from PIL import Image
import multiprocessing
import matplotlib.pyplot as plt
from transformers import ViTImageProcessor, ViTForImageClassification

export = False

q = multiprocessing.Queue()
def get_power(q):
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    start_proc = False
    measure = []
    while True:
        try:
            packet = q.get_nowait()
        except:
            pass
        else:
            if(packet == "Start"):        
                start_proc = True
            if(packet == "Stop"):
                break
        if(start_proc):
            measure.append(pynvml.nvmlDeviceGetPowerUsage(handle)/1e3)
        time.sleep(0.001)
    measure = np.array(measure)
    np.savetxt("gpu_power.log", measure)
    timeaxis = np.linspace(0, len(measure)/0.001, len(measure))
    plt.plot(timeaxis, measure, 'b-')
    plt.grid()
    plt.savefig("./power_plot_gpu.png")


fxn = multiprocessing.Process(target = get_power, args=(q,))
fxn.start()

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224') # 86.56M parameters
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=image, return_tensors="pt")

iteration = 50
q.put("Start")
time.sleep(0.5)
inputs["pixel_values"] = inputs["pixel_values"].cuda()
model = model.eval().cuda()
with torch.no_grad():
    start = time.time()
    for _ in range(iteration):
        outputs = model(**inputs)
    end = time.time()
    total_time = (end-start)/iteration
    print("Time = {:.8f}".format(total_time))
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1)
    print(predicted_class_idx)
    print("Predicted class:", model.config.id2label[predicted_class_idx.item()])
time.sleep(0.5)
q.put("Stop")
fxn.join()


if(export):
    if not os.path.exists("./vit/embedding_layer/"):
        os.makedirs("./vit/embedding_layer/")
    np.save("vit/input_raw.npy", inputs['pixel_values'].cpu().detach().numpy())
    np.save("vit/output.npy", logits.detach().cpu().numpy())
    np.save("vit/embedding_layer/positional_embeddings.npy", model.vit.embeddings.position_embeddings.cpu().detach().numpy())
    np.save("vit/embedding_layer/patch_embeddings_w.npy", model.vit.embeddings.patch_embeddings.projection.weight.cpu().detach().numpy())
    np.save("vit/embedding_layer/patch_embeddings_b.npy", model.vit.embeddings.patch_embeddings.projection.bias.cpu().detach().numpy())
    np.save("vit/embedding_layer/cls_tokens.npy", model.vit.embeddings.cls_token.cpu().detach().numpy())

    for i in range(len(model.vit.encoder.layer)):
        if not os.path.exists("./vit/layer_{}/".format(i)):
            os.makedirs("./vit/layer_{}/".format(i))
        np.save("./vit/layer_{}/layernorm_before_w.npy".format(i), model.vit.encoder.layer[i].layernorm_before.weight.cpu().detach().numpy())
        np.save("./vit/layer_{}/layernorm_before_b.npy".format(i), model.vit.encoder.layer[i].layernorm_before.bias.cpu().detach().numpy())

        qw = model.vit.encoder.layer[i].attention.attention.query.weight.cpu().detach().numpy()
        kw = model.vit.encoder.layer[i].attention.attention.key.weight.cpu().detach().numpy()
        vw = model.vit.encoder.layer[i].attention.attention.value.weight.cpu().detach().numpy()
        qb = model.vit.encoder.layer[i].attention.attention.query.bias.cpu().detach().numpy()
        kb = model.vit.encoder.layer[i].attention.attention.key.bias.cpu().detach().numpy()
        vb = model.vit.encoder.layer[i].attention.attention.value.bias.cpu().detach().numpy()

        qkv_w = np.concatenate((qw,kw,vw), axis=0)
        qkv_b = np.concatenate((qb, kb, vb), axis=0)
        np.save("./vit/layer_{}/qkv_w.npy".format(i), qkv_w.T)
        np.save("./vit/layer_{}/qkv_b.npy".format(i), qkv_b[np.newaxis,...])

        np.save("./vit/layer_{}/attn_out_w.npy".format(i), model.vit.encoder.layer[i].attention.output.dense.weight.cpu().detach().numpy().T)
        np.save("./vit/layer_{}/attn_out_b.npy".format(i), model.vit.encoder.layer[i].attention.output.dense.bias.cpu().detach().numpy()[np.newaxis,...])

        np.save("./vit/layer_{}/layernorm_after_w.npy".format(i), model.vit.encoder.layer[i].layernorm_after.weight.cpu().detach().numpy())
        np.save("./vit/layer_{}/layernorm_after_b.npy".format(i), model.vit.encoder.layer[i].layernorm_after.bias.cpu().detach().numpy()[np.newaxis,...])

        np.save("./vit/layer_{}/intermediate_w.npy".format(i), model.vit.encoder.layer[i].intermediate.dense.weight.cpu().detach().numpy().T)
        np.save("./vit/layer_{}/intermediate_b.npy".format(i), model.vit.encoder.layer[i].intermediate.dense.bias.cpu().detach().numpy()[np.newaxis,...])

        np.save("./vit/layer_{}/output_w.npy".format(i), model.vit.encoder.layer[i].output.dense.weight.cpu().detach().numpy().T)
        np.save("./vit/layer_{}/output_b.npy".format(i), model.vit.encoder.layer[i].output.dense.bias.cpu().detach().numpy()[np.newaxis,...])

    if not os.path.exists("./vit/last_layernorm"):
        os.makedirs("./vit/last_layernorm")
    np.save("./vit/last_layernorm/llnw.npy".format(i), model.vit.layernorm.weight.cpu().detach().numpy())
    np.save("./vit/last_layernorm/llnb.npy".format(i), model.vit.layernorm.bias.cpu().detach().numpy())

    if not os.path.exists("./vit/classifier"):
        os.makedirs("./vit/classifier")
    np.save("./vit/classifier/weight.npy".format(i), model.classifier.weight.cpu().detach().numpy().T)
    np.save("./vit/classifier/bias.npy".format(i), model.classifier.bias.cpu().detach().numpy()[np.newaxis,...])

