from ultralytics import YOLO

model = YOLO("models/yolov8s.pt")

model.train(
    data="data/traffic-sign-detection-yolov8/data.yaml", # data path
    epochs=10, # number of epochs
    imgsz=640, # image size
    batch=24, # batch size
    name = "traffic_signs_model", # output result folder name
    lr0 = 0.01, # initial learning rate
    optimizer = "SGD", # optimizer method
    weight_decay = 0.0005, #  a regularization technique that adds a penalty to the loss function to prevent overfitting
    momentum = 0.935, # a hyperparameter that helps accelerate gradients vectors in the right directions, thus leading to faster converging
    patience = 50, # number of epochs with no improvement after which training will be stopped
    workers = 0, # number of workers for data loading
    device = 0, # device to use for training (0 for GPU, -1 for CPU)
    save = True, # whether to save the model after training
    save_period = 5, # number of epochs between saving the model
    val = True, # whether to perform validation during training
    verbose = True # whether to print training progress and metrics
)

