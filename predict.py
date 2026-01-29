from ultralytics.models.yolo.detect import DetectionPredictor

def predict():
    args = dict(
        model="/data1/huaziyao/result/Fish/yolov8n/weights/best.pt",
        source="/data1/huaziyao/ImageSets/test",
    )

    predictor = DetectionPredictor(overrides=args)
    predictor.predict_cli()

if __name__ == "__main__":
    predict()
