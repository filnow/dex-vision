from ultralytics import FastSAM

model = FastSAM('../models/FastSAM-x.pt')
model.export(format="openvino", half=True, imgsz=640)
