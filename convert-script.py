import coremltools


caffe_model = ("oxford102.caffemodel","deploy.prototxt")


labels = "flower-labels.txt"


corelml_model = coremltools.converters.caffe.convert(
caffe_model,
class_labels=labels,
image_input_names='data'
)

corelml_model.save('FlowerClassifier.mlmodel')
