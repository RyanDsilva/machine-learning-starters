from imageai.Detection import ObjectDetection

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath("./resnet50.h5")
detector.loadModel()
detections = detector.detectObjectsFromImage(
    input_image="test.jpg", output_image_path="result.jpg")

for obj in detections:
    print(obj["name"], " : ", obj["percentage_probability"])
