import sys
import paddle
import paddlehub as hub
module = hub.Module(name="emotion_detection_textcnn")
test_text = ["今天天气真好", "湿纸巾是干垃圾", "别来吵我"]
def predict(texts:str):
    texts=texts.split(",")
    results = module.emotion_classify(texts=texts)
    emotion_labels=[]
    for result in results:
        emotion_labels.append(result['emotion_label'])
    return emotion_labels
if __name__ == '__main__':
    print("emotion_analysis_start")
    print(predict(sys.argv[1]))