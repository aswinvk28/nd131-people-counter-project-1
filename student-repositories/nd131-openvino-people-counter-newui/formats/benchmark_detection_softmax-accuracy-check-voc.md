Running with parameters:
    USE_CASE: object_detection
    FRAMEWORK: caffe
    WORKSPACE: /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/common/caffe
    DATASET_LOCATION: /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/pascal_voc_tfrecord/tfrecord-voc.record
    CHECKPOINT_DIRECTORY: 
    IN_GRAPH: /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.prototxt
    SOCKET_ID: 0
    MODEL_NAME: detection_softmax
    MODE: inference
    PRECISION: fp32
    BATCH_SIZE: 1
    NUM_CORES: 2
    BENCHMARK_ONLY: False
    ACCURACY_ONLY: True
    OUTPUT_RESULTS: False
    DISABLE_TCMALLOC: True
    TCMALLOC_LARGE_ALLOC_REPORT_THRESHOLD: 2147483648
    NOINSTALL: False
    OUTPUT_DIR: /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/tensorflow_object_detection_create_coco_tfrecord
    MPI_NUM_PROCESSES: None
    MPI_NUM_PEOCESSES_PER_SOCKET: 1

WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

Reading package lists...
E: Could not open lock file /var/lib/apt/lists/lock - open (13: Permission denied)
E: Unable to lock directory /var/lib/apt/lists/
W: Problem unlinking the file /var/cache/apt/pkgcache.bin - RemoveCaches (13: Permission denied)
W: Problem unlinking the file /var/cache/apt/srcpkgcache.bin - RemoveCaches (13: Permission denied)

WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)
E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), are you root?
E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)
E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), are you root?
update-alternatives: error: unable to create file '/var/lib/dpkg/alternatives/gcc.dpkg-tmp': Permission denied
update-alternatives: error: unable to create file '/var/lib/dpkg/alternatives/gcc.dpkg-tmp': Permission denied

WARNING: apt does not have a stable CLI interface. Use with caution in scripts.

E: Could not open lock file /var/lib/dpkg/lock-frontend - open (13: Permission denied)
E: Unable to acquire the dpkg frontend lock (/var/lib/dpkg/lock-frontend), are you root?
Requirement already up-to-date: pip in /home/aswin/anaconda3/lib/python3.7/site-packages (20.1.1)
Requirement already satisfied: requests in /home/aswin/anaconda3/lib/python3.7/site-packages (2.21.0)
Requirement already satisfied: urllib3<1.25,>=1.21.1 in /home/aswin/anaconda3/lib/python3.7/site-packages (from requests) (1.24.1)
Requirement already satisfied: idna<2.9,>=2.5 in /home/aswin/anaconda3/lib/python3.7/site-packages (from requests) (2.8)
Requirement already satisfied: certifi>=2017.4.17 in /home/aswin/anaconda3/lib/python3.7/site-packages (from requests) (2020.4.5.1)
Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /home/aswin/anaconda3/lib/python3.7/site-packages (from requests) (3.0.4)
Installing caffe requirements..\n
Check whether the caffe model is present..\n
Log output location: /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/tensorflow_object_detection_create_coco_tfrecord/benchmark_detection_softmax_inference_fp32_20200523_094755.log
benchmarking
/usr/bin/python3.6 common/caffe/run_tf_benchmark.py --framework=caffe --use-case=object_detection --model-name=detection_softmax --precision=fp32 --mode=inference --benchmark-dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks --intelai-models=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/detection_softmax --num-cores=2 --batch-size=1 --socket-id=0 --output-dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/tensorflow_object_detection_create_coco_tfrecord --annotations_dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/caffe2-pose-estimation/annotations/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages --accuracy-only   --verbose --model-source-dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet --in-graph=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.prototxt --in-weights=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.caffemodel --data-location=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/pascal_voc_tfrecord/tfrecord-voc.record --num-inter-threads=1 --num-intra-threads=1 --disable-tcmalloc=True                  
2020-05-23 09:47:57.839571: I tensorflow/core/platform/profile_utils/cpu_utils.cc:94] CPU Frequency: 2993110000 Hz
2020-05-23 09:47:57.839873: I tensorflow/compiler/xla/service/service.cc:150] XLA service 0x1245b00 executing computations on platform Host. Devices:
2020-05-23 09:47:57.839894: I tensorflow/compiler/xla/service/service.cc:158]   StreamExecutor device (0): <undefined>, <undefined>
WARNING: Logging before InitGoogleLogging() is written to STDERR
W0523 09:47:57.842334  8195 _caffe.cpp:139] DEPRECATION WARNING - deprecated use of Python interface
W0523 09:47:57.842356  8195 _caffe.cpp:140] Use this instead (with the named "weights" parameter):
W0523 09:47:57.842360  8195 _caffe.cpp:142] Net('/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.prototxt', 1, weights='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.caffemodel')
I0523 09:47:57.844449  8195 net.cpp:51] Initializing net from parameters: 
state {
  phase: TEST
  level: 0
}
layer {
  name: "data"
  type: "Input"
  top: "data"
  input_param {
    shape {
      dim: 1
      dim: 3
      dim: 224
      dim: 224
    }
  }
}
layer {
  name: "conv1_1"
  type: "Convolution"
  bottom: "data"
  top: "conv1_1"
  convolution_param {
    num_output: 64
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 3
    kernel_w: 3
    stride_h: 2
    stride_w: 2
    dilation: 1
  }
}
layer {
  name: "conv1_2"
  type: "ReLU"
  bottom: "conv1_1"
  top: "conv1_2"
}
layer {
  name: "pool1"
  type: "Pooling"
  bottom: "conv1_2"
  top: "pool1"
  pooling_param {
    pool: MAX
    kernel_h: 3
    kernel_w: 3
    stride_h: 2
    stride_w: 2
    pad_h: 0
    pad_w: 0
  }
}
layer {
  name: "fire2/squeeze1x1_1"
  type: "Convolution"
  bottom: "pool1"
  top: "fire2/squeeze1x1_1"
  convolution_param {
    num_output: 16
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire2/squeeze1x1_2"
  type: "ReLU"
  bottom: "fire2/squeeze1x1_1"
  top: "fire2/squeeze1x1_2"
}
layer {
  name: "fire2/expand1x1_1"
  type: "Convolution"
  bottom: "fire2/squeeze1x1_2"
  top: "fire2/expand1x1_1"
  convolution_param {
    num_output: 64
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire2/expand1x1_2"
  type: "ReLU"
  bottom: "fire2/expand1x1_1"
  top: "fire2/expand1x1_2"
}
layer {
  name: "fire2/expand3x3_1"
  type: "Convolution"
  bottom: "fire2/squeeze1x1_2"
  top: "fire2/expand3x3_1"
  convolution_param {
    num_output: 64
    bias_term: true
    group: 1
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire2/expand3x3_2"
  type: "ReLU"
  bottom: "fire2/expand3x3_1"
  top: "fire2/expand3x3_2"
}
layer {
  name: "fire2/concat"
  type: "Concat"
  bottom: "fire2/expand1x1_2"
  bottom: "fire2/expand3x3_2"
  top: "fire2/concat"
  concat_param {
    axis: 1
  }
}
layer {
  name: "fire3/squeeze1x1_1"
  type: "Convolution"
  bottom: "fire2/concat"
  top: "fire3/squeeze1x1_1"
  convolution_param {
    num_output: 16
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire3/squeeze1x1_2"
  type: "ReLU"
  bottom: "fire3/squeeze1x1_1"
  top: "fire3/squeeze1x1_2"
}
layer {
  name: "fire3/expand1x1_1"
  type: "Convolution"
  bottom: "fire3/squeeze1x1_2"
  top: "fire3/expand1x1_1"
  convolution_param {
    num_output: 64
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire3/expand1x1_2"
  type: "ReLU"
  bottom: "fire3/expand1x1_1"
  top: "fire3/expand1x1_2"
}
layer {
  name: "fire3/expand3x3_1"
  type: "Convolution"
  bottom: "fire3/squeeze1x1_2"
  top: "fire3/expand3x3_1"
  convolution_param {
    num_output: 64
    bias_term: true
    group: 1
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire3/expand3x3_2"
  type: "ReLU"
  bottom: "fire3/expand3x3_1"
  top: "fire3/expand3x3_2"
}
layer {
  name: "fire3/concat"
  type: "Concat"
  bottom: "fire3/expand1x1_2"
  bottom: "fire3/expand3x3_2"
  top: "fire3/concat"
  concat_param {
    axis: 1
  }
}
layer {
  name: "pool3"
  type: "Pooling"
  bottom: "fire3/concat"
  top: "pool3"
  pooling_param {
    pool: MAX
    kernel_h: 3
    kernel_w: 3
    stride_h: 2
    stride_w: 2
    pad_h: 0
    pad_w: 0
  }
}
layer {
  name: "fire4/squeeze1x1_1"
  type: "Convolution"
  bottom: "pool3"
  top: "fire4/squeeze1x1_1"
  convolution_param {
    num_output: 32
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire4/squeeze1x1_2"
  type: "ReLU"
  bottom: "fire4/squeeze1x1_1"
  top: "fire4/squeeze1x1_2"
}
layer {
  name: "fire4/expand1x1_1"
  type: "Convolution"
  bottom: "fire4/squeeze1x1_2"
  top: "fire4/expand1x1_1"
  convolution_param {
    num_output: 128
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire4/expand1x1_2"
  type: "ReLU"
  bottom: "fire4/expand1x1_1"
  top: "fire4/expand1x1_2"
}
layer {
  name: "fire4/expand3x3_1"
  type: "Convolution"
  bottom: "fire4/squeeze1x1_2"
  top: "fire4/expand3x3_1"
  convolution_param {
    num_output: 128
    bias_term: true
    group: 1
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire4/expand3x3_2"
  type: "ReLU"
  bottom: "fire4/expand3x3_1"
  top: "fire4/expand3x3_2"
}
layer {
  name: "fire4/concat"
  type: "Concat"
  bottom: "fire4/expand1x1_2"
  bottom: "fire4/expand3x3_2"
  top: "fire4/concat"
  concat_param {
    axis: 1
  }
}
layer {
  name: "fire5/squeeze1x1_1"
  type: "Convolution"
  bottom: "fire4/concat"
  top: "fire5/squeeze1x1_1"
  convolution_param {
    num_output: 32
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire5/squeeze1x1_2"
  type: "ReLU"
  bottom: "fire5/squeeze1x1_1"
  top: "fire5/squeeze1x1_2"
}
layer {
  name: "fire5/expand1x1_1"
  type: "Convolution"
  bottom: "fire5/squeeze1x1_2"
  top: "fire5/expand1x1_1"
  convolution_param {
    num_output: 128
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire5/expand1x1_2"
  type: "ReLU"
  bottom: "fire5/expand1x1_1"
  top: "fire5/expand1x1_2"
}
layer {
  name: "fire5/expand3x3_1"
  type: "Convolution"
  bottom: "fire5/squeeze1x1_2"
  top: "fire5/expand3x3_1"
  convolution_param {
    num_output: 128
    bias_term: true
    group: 1
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire5/expand3x3_2"
  type: "ReLU"
  bottom: "fire5/expand3x3_1"
  top: "fire5/expand3x3_2"
}
layer {
  name: "fire5/concat"
  type: "Concat"
  bottom: "fire5/expand1x1_2"
  bottom: "fire5/expand3x3_2"
  top: "fire5/concat"
  concat_param {
    axis: 1
  }
}
layer {
  name: "pool5"
  type: "Pooling"
  bottom: "fire5/concat"
  top: "pool5"
  pooling_param {
    pool: MAX
    kernel_h: 3
    kernel_w: 3
    stride_h: 2
    stride_w: 2
    pad_h: 0
    pad_w: 0
  }
}
layer {
  name: "fire6/squeeze1x1_1"
  type: "Convolution"
  bottom: "pool5"
  top: "fire6/squeeze1x1_1"
  convolution_param {
    num_output: 48
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire6/squeeze1x1_2"
  type: "ReLU"
  bottom: "fire6/squeeze1x1_1"
  top: "fire6/squeeze1x1_2"
}
layer {
  name: "fire6/expand1x1_1"
  type: "Convolution"
  bottom: "fire6/squeeze1x1_2"
  top: "fire6/expand1x1_1"
  convolution_param {
    num_output: 192
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire6/expand1x1_2"
  type: "ReLU"
  bottom: "fire6/expand1x1_1"
  top: "fire6/expand1x1_2"
}
layer {
  name: "fire6/expand3x3_1"
  type: "Convolution"
  bottom: "fire6/squeeze1x1_2"
  top: "fire6/expand3x3_1"
  convolution_param {
    num_output: 192
    bias_term: true
    group: 1
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire6/expand3x3_2"
  type: "ReLU"
  bottom: "fire6/expand3x3_1"
  top: "fire6/expand3x3_2"
}
layer {
  name: "fire6/concat"
  type: "Concat"
  bottom: "fire6/expand1x1_2"
  bottom: "fire6/expand3x3_2"
  top: "fire6/concat"
  concat_param {
    axis: 1
  }
}
layer {
  name: "fire7/squeeze1x1_1"
  type: "Convolution"
  bottom: "fire6/concat"
  top: "fire7/squeeze1x1_1"
  convolution_param {
    num_output: 48
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire7/squeeze1x1_2"
  type: "ReLU"
  bottom: "fire7/squeeze1x1_1"
  top: "fire7/squeeze1x1_2"
}
layer {
  name: "fire7/expand1x1_1"
  type: "Convolution"
  bottom: "fire7/squeeze1x1_2"
  top: "fire7/expand1x1_1"
  convolution_param {
    num_output: 192
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire7/expand1x1_2"
  type: "ReLU"
  bottom: "fire7/expand1x1_1"
  top: "fire7/expand1x1_2"
}
layer {
  name: "fire7/expand3x3_1"
  type: "Convolution"
  bottom: "fire7/squeeze1x1_2"
  top: "fire7/expand3x3_1"
  convolution_param {
    num_output: 192
    bias_term: true
    group: 1
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire7/expand3x3_2"
  type: "ReLU"
  bottom: "fire7/expand3x3_1"
  top: "fire7/expand3x3_2"
}
layer {
  name: "fire7/concat"
  type: "Concat"
  bottom: "fire7/expand1x1_2"
  bottom: "fire7/expand3x3_2"
  top: "fire7/concat"
  concat_param {
    axis: 1
  }
}
layer {
  name: "fire8/squeeze1x1_1"
  type: "Convolution"
  bottom: "fire7/concat"
  top: "fire8/squeeze1x1_1"
  convolution_param {
    num_output: 64
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire8/squeeze1x1_2"
  type: "ReLU"
  bottom: "fire8/squeeze1x1_1"
  top: "fire8/squeeze1x1_2"
}
layer {
  name: "fire8/expand1x1_1"
  type: "Convolution"
  bottom: "fire8/squeeze1x1_2"
  top: "fire8/expand1x1_1"
  convolution_param {
    num_output: 256
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire8/expand1x1_2"
  type: "ReLU"
  bottom: "fire8/expand1x1_1"
  top: "fire8/expand1x1_2"
}
layer {
  name: "fire8/expand3x3_1"
  type: "Convolution"
  bottom: "fire8/squeeze1x1_2"
  top: "fire8/expand3x3_1"
  convolution_param {
    num_output: 256
    bias_term: true
    group: 1
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire8/expand3x3_2"
  type: "ReLU"
  bottom: "fire8/expand3x3_1"
  top: "fire8/expand3x3_2"
}
layer {
  name: "fire8/concat"
  type: "Concat"
  bottom: "fire8/expand1x1_2"
  bottom: "fire8/expand3x3_2"
  top: "fire8/concat"
  concat_param {
    axis: 1
  }
}
layer {
  name: "fire9/squeeze1x1_1"
  type: "Convolution"
  bottom: "fire8/concat"
  top: "fire9/squeeze1x1_1"
  convolution_param {
    num_output: 64
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire9/squeeze1x1_2"
  type: "ReLU"
  bottom: "fire9/squeeze1x1_1"
  top: "fire9/squeeze1x1_2"
}
layer {
  name: "fire9/expand1x1_1"
  type: "Convolution"
  bottom: "fire9/squeeze1x1_2"
  top: "fire9/expand1x1_1"
  convolution_param {
    num_output: 256
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire9/expand1x1_2"
  type: "ReLU"
  bottom: "fire9/expand1x1_1"
  top: "fire9/expand1x1_2"
}
layer {
  name: "fire9/expand3x3_1"
  type: "Convolution"
  bottom: "fire9/squeeze1x1_2"
  top: "fire9/expand3x3_1"
  convolution_param {
    num_output: 256
    bias_term: true
    group: 1
    pad_h: 1
    pad_w: 1
    kernel_h: 3
    kernel_w: 3
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "fire9/expand3x3_2"
  type: "ReLU"
  bottom: "fire9/expand3x3_1"
  top: "fire9/expand3x3_2"
}
layer {
  name: "fire9/concat_1"
  type: "Concat"
  bottom: "fire9/expand1x1_2"
  bottom: "fire9/expand3x3_2"
  top: "fire9/concat_1"
  concat_param {
    axis: 1
  }
}
layer {
  name: "fire9/concat_2__fire9/concat_mask"
  type: "Dropout"
  bottom: "fire9/concat_1"
  top: "fire9/concat_2"
  dropout_param {
    dropout_ratio: 0.5
  }
}
layer {
  name: "conv10_1"
  type: "Convolution"
  bottom: "fire9/concat_2"
  top: "conv10_1"
  convolution_param {
    num_output: 1000
    bias_term: true
    group: 1
    pad_h: 0
    pad_w: 0
    kernel_h: 1
    kernel_w: 1
    stride_h: 1
    stride_w: 1
    dilation: 1
  }
}
layer {
  name: "conv10_2"
  type: "ReLU"
  bottom: "conv10_1"
  top: "conv10_2"
}
layer {
  name: "pool10"
  type: "Pooling"
  bottom: "conv10_2"
  top: "pool10"
  pooling_param {
    pool: AVE
    global_pooling: true
  }
}
layer {
  name: "softmaxout"
  type: "Softmax"
  bottom: "pool10"
  top: "softmaxout"
}
I0523 09:47:57.844911  8195 layer_factory.hpp:77] Creating layer data
I0523 09:47:57.844929  8195 net.cpp:84] Creating Layer data
I0523 09:47:57.844935  8195 net.cpp:380] data -> data
I0523 09:47:57.844959  8195 net.cpp:122] Setting up data
I0523 09:47:57.844962  8195 net.cpp:129] Top shape: 1 3 224 224 (150528)
I0523 09:47:57.844967  8195 net.cpp:137] Memory required for data: 602112
I0523 09:47:57.844971  8195 layer_factory.hpp:77] Creating layer conv1_1
I0523 09:47:57.844978  8195 net.cpp:84] Creating Layer conv1_1
I0523 09:47:57.844983  8195 net.cpp:406] conv1_1 <- data
I0523 09:47:57.844988  8195 net.cpp:380] conv1_1 -> conv1_1
I0523 09:47:57.845041  8195 net.cpp:122] Setting up conv1_1
I0523 09:47:57.845046  8195 net.cpp:129] Top shape: 1 64 111 111 (788544)
I0523 09:47:57.845052  8195 net.cpp:137] Memory required for data: 3756288
I0523 09:47:57.845063  8195 layer_factory.hpp:77] Creating layer conv1_2
I0523 09:47:57.845069  8195 net.cpp:84] Creating Layer conv1_2
I0523 09:47:57.845073  8195 net.cpp:406] conv1_2 <- conv1_1
I0523 09:47:57.845078  8195 net.cpp:380] conv1_2 -> conv1_2
I0523 09:47:57.845088  8195 net.cpp:122] Setting up conv1_2
I0523 09:47:57.845091  8195 net.cpp:129] Top shape: 1 64 111 111 (788544)
I0523 09:47:57.845095  8195 net.cpp:137] Memory required for data: 6910464
I0523 09:47:57.845098  8195 layer_factory.hpp:77] Creating layer pool1
I0523 09:47:57.845104  8195 net.cpp:84] Creating Layer pool1
I0523 09:47:57.845108  8195 net.cpp:406] pool1 <- conv1_2
I0523 09:47:57.845113  8195 net.cpp:380] pool1 -> pool1
I0523 09:47:57.845119  8195 net.cpp:122] Setting up pool1
I0523 09:47:57.845124  8195 net.cpp:129] Top shape: 1 64 55 55 (193600)
I0523 09:47:57.845127  8195 net.cpp:137] Memory required for data: 7684864
I0523 09:47:57.845130  8195 layer_factory.hpp:77] Creating layer fire2/squeeze1x1_1
I0523 09:47:57.845139  8195 net.cpp:84] Creating Layer fire2/squeeze1x1_1
I0523 09:47:57.845142  8195 net.cpp:406] fire2/squeeze1x1_1 <- pool1
I0523 09:47:57.845149  8195 net.cpp:380] fire2/squeeze1x1_1 -> fire2/squeeze1x1_1
I0523 09:47:57.845171  8195 net.cpp:122] Setting up fire2/squeeze1x1_1
I0523 09:47:57.845175  8195 net.cpp:129] Top shape: 1 16 55 55 (48400)
I0523 09:47:57.845180  8195 net.cpp:137] Memory required for data: 7878464
I0523 09:47:57.845185  8195 layer_factory.hpp:77] Creating layer fire2/squeeze1x1_2
I0523 09:47:57.845191  8195 net.cpp:84] Creating Layer fire2/squeeze1x1_2
I0523 09:47:57.845194  8195 net.cpp:406] fire2/squeeze1x1_2 <- fire2/squeeze1x1_1
I0523 09:47:57.845201  8195 net.cpp:380] fire2/squeeze1x1_2 -> fire2/squeeze1x1_2
I0523 09:47:57.845208  8195 net.cpp:122] Setting up fire2/squeeze1x1_2
I0523 09:47:57.845212  8195 net.cpp:129] Top shape: 1 16 55 55 (48400)
I0523 09:47:57.845216  8195 net.cpp:137] Memory required for data: 8072064
I0523 09:47:57.845219  8195 layer_factory.hpp:77] Creating layer fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split
I0523 09:47:57.845225  8195 net.cpp:84] Creating Layer fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split
I0523 09:47:57.845228  8195 net.cpp:406] fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split <- fire2/squeeze1x1_2
I0523 09:47:57.845233  8195 net.cpp:380] fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split -> fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split_0
I0523 09:47:57.845239  8195 net.cpp:380] fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split -> fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split_1
I0523 09:47:57.845250  8195 net.cpp:122] Setting up fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split
I0523 09:47:57.845254  8195 net.cpp:129] Top shape: 1 16 55 55 (48400)
I0523 09:47:57.845259  8195 net.cpp:129] Top shape: 1 16 55 55 (48400)
I0523 09:47:57.845264  8195 net.cpp:137] Memory required for data: 8459264
I0523 09:47:57.845268  8195 layer_factory.hpp:77] Creating layer fire2/expand1x1_1
I0523 09:47:57.845275  8195 net.cpp:84] Creating Layer fire2/expand1x1_1
I0523 09:47:57.845283  8195 net.cpp:406] fire2/expand1x1_1 <- fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split_0
I0523 09:47:57.845289  8195 net.cpp:380] fire2/expand1x1_1 -> fire2/expand1x1_1
I0523 09:47:57.845311  8195 net.cpp:122] Setting up fire2/expand1x1_1
I0523 09:47:57.845315  8195 net.cpp:129] Top shape: 1 64 55 55 (193600)
I0523 09:47:57.845320  8195 net.cpp:137] Memory required for data: 9233664
I0523 09:47:57.845327  8195 layer_factory.hpp:77] Creating layer fire2/expand1x1_2
I0523 09:47:57.845336  8195 net.cpp:84] Creating Layer fire2/expand1x1_2
I0523 09:47:57.845340  8195 net.cpp:406] fire2/expand1x1_2 <- fire2/expand1x1_1
I0523 09:47:57.845346  8195 net.cpp:380] fire2/expand1x1_2 -> fire2/expand1x1_2
I0523 09:47:57.845352  8195 net.cpp:122] Setting up fire2/expand1x1_2
I0523 09:47:57.845356  8195 net.cpp:129] Top shape: 1 64 55 55 (193600)
I0523 09:47:57.845361  8195 net.cpp:137] Memory required for data: 10008064
I0523 09:47:57.845364  8195 layer_factory.hpp:77] Creating layer fire2/expand3x3_1
I0523 09:47:57.845369  8195 net.cpp:84] Creating Layer fire2/expand3x3_1
I0523 09:47:57.845376  8195 net.cpp:406] fire2/expand3x3_1 <- fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split_1
I0523 09:47:57.845382  8195 net.cpp:380] fire2/expand3x3_1 -> fire2/expand3x3_1
I0523 09:47:57.845417  8195 net.cpp:122] Setting up fire2/expand3x3_1
I0523 09:47:57.845423  8195 net.cpp:129] Top shape: 1 64 55 55 (193600)
I0523 09:47:57.845428  8195 net.cpp:137] Memory required for data: 10782464
I0523 09:47:57.845433  8195 layer_factory.hpp:77] Creating layer fire2/expand3x3_2
I0523 09:47:57.845441  8195 net.cpp:84] Creating Layer fire2/expand3x3_2
I0523 09:47:57.845444  8195 net.cpp:406] fire2/expand3x3_2 <- fire2/expand3x3_1
I0523 09:47:57.845449  8195 net.cpp:380] fire2/expand3x3_2 -> fire2/expand3x3_2
I0523 09:47:57.845456  8195 net.cpp:122] Setting up fire2/expand3x3_2
I0523 09:47:57.845459  8195 net.cpp:129] Top shape: 1 64 55 55 (193600)
I0523 09:47:57.845466  8195 net.cpp:137] Memory required for data: 11556864
I0523 09:47:57.845469  8195 layer_factory.hpp:77] Creating layer fire2/concat
I0523 09:47:57.845475  8195 net.cpp:84] Creating Layer fire2/concat
I0523 09:47:57.845479  8195 net.cpp:406] fire2/concat <- fire2/expand1x1_2
I0523 09:47:57.845482  8195 net.cpp:406] fire2/concat <- fire2/expand3x3_2
I0523 09:47:57.845489  8195 net.cpp:380] fire2/concat -> fire2/concat
I0523 09:47:57.845502  8195 net.cpp:122] Setting up fire2/concat
I0523 09:47:57.845506  8195 net.cpp:129] Top shape: 1 128 55 55 (387200)
I0523 09:47:57.845510  8195 net.cpp:137] Memory required for data: 13105664
I0523 09:47:57.845515  8195 layer_factory.hpp:77] Creating layer fire3/squeeze1x1_1
I0523 09:47:57.845521  8195 net.cpp:84] Creating Layer fire3/squeeze1x1_1
I0523 09:47:57.845526  8195 net.cpp:406] fire3/squeeze1x1_1 <- fire2/concat
I0523 09:47:57.845531  8195 net.cpp:380] fire3/squeeze1x1_1 -> fire3/squeeze1x1_1
I0523 09:47:57.845554  8195 net.cpp:122] Setting up fire3/squeeze1x1_1
I0523 09:47:57.845558  8195 net.cpp:129] Top shape: 1 16 55 55 (48400)
I0523 09:47:57.845563  8195 net.cpp:137] Memory required for data: 13299264
I0523 09:47:57.845568  8195 layer_factory.hpp:77] Creating layer fire3/squeeze1x1_2
I0523 09:47:57.845578  8195 net.cpp:84] Creating Layer fire3/squeeze1x1_2
I0523 09:47:57.845582  8195 net.cpp:406] fire3/squeeze1x1_2 <- fire3/squeeze1x1_1
I0523 09:47:57.845587  8195 net.cpp:380] fire3/squeeze1x1_2 -> fire3/squeeze1x1_2
I0523 09:47:57.845592  8195 net.cpp:122] Setting up fire3/squeeze1x1_2
I0523 09:47:57.845595  8195 net.cpp:129] Top shape: 1 16 55 55 (48400)
I0523 09:47:57.845602  8195 net.cpp:137] Memory required for data: 13492864
I0523 09:47:57.845604  8195 layer_factory.hpp:77] Creating layer fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split
I0523 09:47:57.845616  8195 net.cpp:84] Creating Layer fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split
I0523 09:47:57.845620  8195 net.cpp:406] fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split <- fire3/squeeze1x1_2
I0523 09:47:57.845625  8195 net.cpp:380] fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split -> fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split_0
I0523 09:47:57.845630  8195 net.cpp:380] fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split -> fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split_1
I0523 09:47:57.845639  8195 net.cpp:122] Setting up fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split
I0523 09:47:57.845643  8195 net.cpp:129] Top shape: 1 16 55 55 (48400)
I0523 09:47:57.845647  8195 net.cpp:129] Top shape: 1 16 55 55 (48400)
I0523 09:47:57.845651  8195 net.cpp:137] Memory required for data: 13880064
I0523 09:47:57.845655  8195 layer_factory.hpp:77] Creating layer fire3/expand1x1_1
I0523 09:47:57.845662  8195 net.cpp:84] Creating Layer fire3/expand1x1_1
I0523 09:47:57.845667  8195 net.cpp:406] fire3/expand1x1_1 <- fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split_0
I0523 09:47:57.845674  8195 net.cpp:380] fire3/expand1x1_1 -> fire3/expand1x1_1
I0523 09:47:57.845695  8195 net.cpp:122] Setting up fire3/expand1x1_1
I0523 09:47:57.845700  8195 net.cpp:129] Top shape: 1 64 55 55 (193600)
I0523 09:47:57.845705  8195 net.cpp:137] Memory required for data: 14654464
I0523 09:47:57.845710  8195 layer_factory.hpp:77] Creating layer fire3/expand1x1_2
I0523 09:47:57.845717  8195 net.cpp:84] Creating Layer fire3/expand1x1_2
I0523 09:47:57.845721  8195 net.cpp:406] fire3/expand1x1_2 <- fire3/expand1x1_1
I0523 09:47:57.845726  8195 net.cpp:380] fire3/expand1x1_2 -> fire3/expand1x1_2
I0523 09:47:57.845731  8195 net.cpp:122] Setting up fire3/expand1x1_2
I0523 09:47:57.845736  8195 net.cpp:129] Top shape: 1 64 55 55 (193600)
I0523 09:47:57.845739  8195 net.cpp:137] Memory required for data: 15428864
I0523 09:47:57.845742  8195 layer_factory.hpp:77] Creating layer fire3/expand3x3_1
I0523 09:47:57.845748  8195 net.cpp:84] Creating Layer fire3/expand3x3_1
I0523 09:47:57.845753  8195 net.cpp:406] fire3/expand3x3_1 <- fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split_1
I0523 09:47:57.845758  8195 net.cpp:380] fire3/expand3x3_1 -> fire3/expand3x3_1
I0523 09:47:57.845790  8195 net.cpp:122] Setting up fire3/expand3x3_1
I0523 09:47:57.845798  8195 net.cpp:129] Top shape: 1 64 55 55 (193600)
I0523 09:47:57.845803  8195 net.cpp:137] Memory required for data: 16203264
I0523 09:47:57.845808  8195 layer_factory.hpp:77] Creating layer fire3/expand3x3_2
I0523 09:47:57.845814  8195 net.cpp:84] Creating Layer fire3/expand3x3_2
I0523 09:47:57.845819  8195 net.cpp:406] fire3/expand3x3_2 <- fire3/expand3x3_1
I0523 09:47:57.845824  8195 net.cpp:380] fire3/expand3x3_2 -> fire3/expand3x3_2
I0523 09:47:57.845830  8195 net.cpp:122] Setting up fire3/expand3x3_2
I0523 09:47:57.845834  8195 net.cpp:129] Top shape: 1 64 55 55 (193600)
I0523 09:47:57.845839  8195 net.cpp:137] Memory required for data: 16977664
I0523 09:47:57.845841  8195 layer_factory.hpp:77] Creating layer fire3/concat
I0523 09:47:57.845849  8195 net.cpp:84] Creating Layer fire3/concat
I0523 09:47:57.845852  8195 net.cpp:406] fire3/concat <- fire3/expand1x1_2
I0523 09:47:57.845856  8195 net.cpp:406] fire3/concat <- fire3/expand3x3_2
I0523 09:47:57.845861  8195 net.cpp:380] fire3/concat -> fire3/concat
I0523 09:47:57.845870  8195 net.cpp:122] Setting up fire3/concat
I0523 09:47:57.845873  8195 net.cpp:129] Top shape: 1 128 55 55 (387200)
I0523 09:47:57.845877  8195 net.cpp:137] Memory required for data: 18526464
I0523 09:47:57.845881  8195 layer_factory.hpp:77] Creating layer pool3
I0523 09:47:57.845886  8195 net.cpp:84] Creating Layer pool3
I0523 09:47:57.845890  8195 net.cpp:406] pool3 <- fire3/concat
I0523 09:47:57.845894  8195 net.cpp:380] pool3 -> pool3
I0523 09:47:57.845903  8195 net.cpp:122] Setting up pool3
I0523 09:47:57.845909  8195 net.cpp:129] Top shape: 1 128 27 27 (93312)
I0523 09:47:57.845913  8195 net.cpp:137] Memory required for data: 18899712
I0523 09:47:57.845917  8195 layer_factory.hpp:77] Creating layer fire4/squeeze1x1_1
I0523 09:47:57.845923  8195 net.cpp:84] Creating Layer fire4/squeeze1x1_1
I0523 09:47:57.845927  8195 net.cpp:406] fire4/squeeze1x1_1 <- pool3
I0523 09:47:57.845933  8195 net.cpp:380] fire4/squeeze1x1_1 -> fire4/squeeze1x1_1
I0523 09:47:57.845958  8195 net.cpp:122] Setting up fire4/squeeze1x1_1
I0523 09:47:57.845963  8195 net.cpp:129] Top shape: 1 32 27 27 (23328)
I0523 09:47:57.845966  8195 net.cpp:137] Memory required for data: 18993024
I0523 09:47:57.845971  8195 layer_factory.hpp:77] Creating layer fire4/squeeze1x1_2
I0523 09:47:57.845978  8195 net.cpp:84] Creating Layer fire4/squeeze1x1_2
I0523 09:47:57.845984  8195 net.cpp:406] fire4/squeeze1x1_2 <- fire4/squeeze1x1_1
I0523 09:47:57.845990  8195 net.cpp:380] fire4/squeeze1x1_2 -> fire4/squeeze1x1_2
I0523 09:47:57.845995  8195 net.cpp:122] Setting up fire4/squeeze1x1_2
I0523 09:47:57.845999  8195 net.cpp:129] Top shape: 1 32 27 27 (23328)
I0523 09:47:57.846004  8195 net.cpp:137] Memory required for data: 19086336
I0523 09:47:57.846007  8195 layer_factory.hpp:77] Creating layer fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split
I0523 09:47:57.846011  8195 net.cpp:84] Creating Layer fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split
I0523 09:47:57.846017  8195 net.cpp:406] fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split <- fire4/squeeze1x1_2
I0523 09:47:57.846022  8195 net.cpp:380] fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split -> fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split_0
I0523 09:47:57.846029  8195 net.cpp:380] fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split -> fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split_1
I0523 09:47:57.846035  8195 net.cpp:122] Setting up fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split
I0523 09:47:57.846038  8195 net.cpp:129] Top shape: 1 32 27 27 (23328)
I0523 09:47:57.846042  8195 net.cpp:129] Top shape: 1 32 27 27 (23328)
I0523 09:47:57.846047  8195 net.cpp:137] Memory required for data: 19272960
I0523 09:47:57.846063  8195 layer_factory.hpp:77] Creating layer fire4/expand1x1_1
I0523 09:47:57.846077  8195 net.cpp:84] Creating Layer fire4/expand1x1_1
I0523 09:47:57.846081  8195 net.cpp:406] fire4/expand1x1_1 <- fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split_0
I0523 09:47:57.846091  8195 net.cpp:380] fire4/expand1x1_1 -> fire4/expand1x1_1
I0523 09:47:57.846110  8195 net.cpp:122] Setting up fire4/expand1x1_1
I0523 09:47:57.846117  8195 net.cpp:129] Top shape: 1 128 27 27 (93312)
I0523 09:47:57.846123  8195 net.cpp:137] Memory required for data: 19646208
I0523 09:47:57.846130  8195 layer_factory.hpp:77] Creating layer fire4/expand1x1_2
I0523 09:47:57.846138  8195 net.cpp:84] Creating Layer fire4/expand1x1_2
I0523 09:47:57.846141  8195 net.cpp:406] fire4/expand1x1_2 <- fire4/expand1x1_1
I0523 09:47:57.846145  8195 net.cpp:380] fire4/expand1x1_2 -> fire4/expand1x1_2
I0523 09:47:57.846151  8195 net.cpp:122] Setting up fire4/expand1x1_2
I0523 09:47:57.846154  8195 net.cpp:129] Top shape: 1 128 27 27 (93312)
I0523 09:47:57.846159  8195 net.cpp:137] Memory required for data: 20019456
I0523 09:47:57.846163  8195 layer_factory.hpp:77] Creating layer fire4/expand3x3_1
I0523 09:47:57.846168  8195 net.cpp:84] Creating Layer fire4/expand3x3_1
I0523 09:47:57.846174  8195 net.cpp:406] fire4/expand3x3_1 <- fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split_1
I0523 09:47:57.846179  8195 net.cpp:380] fire4/expand3x3_1 -> fire4/expand3x3_1
I0523 09:47:57.846252  8195 net.cpp:122] Setting up fire4/expand3x3_1
I0523 09:47:57.846261  8195 net.cpp:129] Top shape: 1 128 27 27 (93312)
I0523 09:47:57.846266  8195 net.cpp:137] Memory required for data: 20392704
I0523 09:47:57.846271  8195 layer_factory.hpp:77] Creating layer fire4/expand3x3_2
I0523 09:47:57.846279  8195 net.cpp:84] Creating Layer fire4/expand3x3_2
I0523 09:47:57.846283  8195 net.cpp:406] fire4/expand3x3_2 <- fire4/expand3x3_1
I0523 09:47:57.846287  8195 net.cpp:380] fire4/expand3x3_2 -> fire4/expand3x3_2
I0523 09:47:57.846295  8195 net.cpp:122] Setting up fire4/expand3x3_2
I0523 09:47:57.846299  8195 net.cpp:129] Top shape: 1 128 27 27 (93312)
I0523 09:47:57.846303  8195 net.cpp:137] Memory required for data: 20765952
I0523 09:47:57.846307  8195 layer_factory.hpp:77] Creating layer fire4/concat
I0523 09:47:57.846314  8195 net.cpp:84] Creating Layer fire4/concat
I0523 09:47:57.846319  8195 net.cpp:406] fire4/concat <- fire4/expand1x1_2
I0523 09:47:57.846323  8195 net.cpp:406] fire4/concat <- fire4/expand3x3_2
I0523 09:47:57.846328  8195 net.cpp:380] fire4/concat -> fire4/concat
I0523 09:47:57.846334  8195 net.cpp:122] Setting up fire4/concat
I0523 09:47:57.846338  8195 net.cpp:129] Top shape: 1 256 27 27 (186624)
I0523 09:47:57.846343  8195 net.cpp:137] Memory required for data: 21512448
I0523 09:47:57.846345  8195 layer_factory.hpp:77] Creating layer fire5/squeeze1x1_1
I0523 09:47:57.846352  8195 net.cpp:84] Creating Layer fire5/squeeze1x1_1
I0523 09:47:57.846359  8195 net.cpp:406] fire5/squeeze1x1_1 <- fire4/concat
I0523 09:47:57.846364  8195 net.cpp:380] fire5/squeeze1x1_1 -> fire5/squeeze1x1_1
I0523 09:47:57.846391  8195 net.cpp:122] Setting up fire5/squeeze1x1_1
I0523 09:47:57.846395  8195 net.cpp:129] Top shape: 1 32 27 27 (23328)
I0523 09:47:57.846402  8195 net.cpp:137] Memory required for data: 21605760
I0523 09:47:57.846407  8195 layer_factory.hpp:77] Creating layer fire5/squeeze1x1_2
I0523 09:47:57.846415  8195 net.cpp:84] Creating Layer fire5/squeeze1x1_2
I0523 09:47:57.846417  8195 net.cpp:406] fire5/squeeze1x1_2 <- fire5/squeeze1x1_1
I0523 09:47:57.846422  8195 net.cpp:380] fire5/squeeze1x1_2 -> fire5/squeeze1x1_2
I0523 09:47:57.846429  8195 net.cpp:122] Setting up fire5/squeeze1x1_2
I0523 09:47:57.846432  8195 net.cpp:129] Top shape: 1 32 27 27 (23328)
I0523 09:47:57.846437  8195 net.cpp:137] Memory required for data: 21699072
I0523 09:47:57.846441  8195 layer_factory.hpp:77] Creating layer fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split
I0523 09:47:57.846446  8195 net.cpp:84] Creating Layer fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split
I0523 09:47:57.846451  8195 net.cpp:406] fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split <- fire5/squeeze1x1_2
I0523 09:47:57.846457  8195 net.cpp:380] fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split -> fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split_0
I0523 09:47:57.846463  8195 net.cpp:380] fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split -> fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split_1
I0523 09:47:57.846469  8195 net.cpp:122] Setting up fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split
I0523 09:47:57.846477  8195 net.cpp:129] Top shape: 1 32 27 27 (23328)
I0523 09:47:57.846482  8195 net.cpp:129] Top shape: 1 32 27 27 (23328)
I0523 09:47:57.846485  8195 net.cpp:137] Memory required for data: 21885696
I0523 09:47:57.846488  8195 layer_factory.hpp:77] Creating layer fire5/expand1x1_1
I0523 09:47:57.846496  8195 net.cpp:84] Creating Layer fire5/expand1x1_1
I0523 09:47:57.846503  8195 net.cpp:406] fire5/expand1x1_1 <- fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split_0
I0523 09:47:57.846509  8195 net.cpp:380] fire5/expand1x1_1 -> fire5/expand1x1_1
I0523 09:47:57.846527  8195 net.cpp:122] Setting up fire5/expand1x1_1
I0523 09:47:57.846531  8195 net.cpp:129] Top shape: 1 128 27 27 (93312)
I0523 09:47:57.846535  8195 net.cpp:137] Memory required for data: 22258944
I0523 09:47:57.846540  8195 layer_factory.hpp:77] Creating layer fire5/expand1x1_2
I0523 09:47:57.846547  8195 net.cpp:84] Creating Layer fire5/expand1x1_2
I0523 09:47:57.846551  8195 net.cpp:406] fire5/expand1x1_2 <- fire5/expand1x1_1
I0523 09:47:57.846556  8195 net.cpp:380] fire5/expand1x1_2 -> fire5/expand1x1_2
I0523 09:47:57.846561  8195 net.cpp:122] Setting up fire5/expand1x1_2
I0523 09:47:57.846565  8195 net.cpp:129] Top shape: 1 128 27 27 (93312)
I0523 09:47:57.846570  8195 net.cpp:137] Memory required for data: 22632192
I0523 09:47:57.846572  8195 layer_factory.hpp:77] Creating layer fire5/expand3x3_1
I0523 09:47:57.846581  8195 net.cpp:84] Creating Layer fire5/expand3x3_1
I0523 09:47:57.846586  8195 net.cpp:406] fire5/expand3x3_1 <- fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split_1
I0523 09:47:57.846592  8195 net.cpp:380] fire5/expand3x3_1 -> fire5/expand3x3_1
I0523 09:47:57.846664  8195 net.cpp:122] Setting up fire5/expand3x3_1
I0523 09:47:57.846673  8195 net.cpp:129] Top shape: 1 128 27 27 (93312)
I0523 09:47:57.846676  8195 net.cpp:137] Memory required for data: 23005440
I0523 09:47:57.846681  8195 layer_factory.hpp:77] Creating layer fire5/expand3x3_2
I0523 09:47:57.846689  8195 net.cpp:84] Creating Layer fire5/expand3x3_2
I0523 09:47:57.846693  8195 net.cpp:406] fire5/expand3x3_2 <- fire5/expand3x3_1
I0523 09:47:57.846699  8195 net.cpp:380] fire5/expand3x3_2 -> fire5/expand3x3_2
I0523 09:47:57.846706  8195 net.cpp:122] Setting up fire5/expand3x3_2
I0523 09:47:57.846710  8195 net.cpp:129] Top shape: 1 128 27 27 (93312)
I0523 09:47:57.846716  8195 net.cpp:137] Memory required for data: 23378688
I0523 09:47:57.846719  8195 layer_factory.hpp:77] Creating layer fire5/concat
I0523 09:47:57.846725  8195 net.cpp:84] Creating Layer fire5/concat
I0523 09:47:57.846731  8195 net.cpp:406] fire5/concat <- fire5/expand1x1_2
I0523 09:47:57.846735  8195 net.cpp:406] fire5/concat <- fire5/expand3x3_2
I0523 09:47:57.846741  8195 net.cpp:380] fire5/concat -> fire5/concat
I0523 09:47:57.846746  8195 net.cpp:122] Setting up fire5/concat
I0523 09:47:57.846750  8195 net.cpp:129] Top shape: 1 256 27 27 (186624)
I0523 09:47:57.846755  8195 net.cpp:137] Memory required for data: 24125184
I0523 09:47:57.846758  8195 layer_factory.hpp:77] Creating layer pool5
I0523 09:47:57.846763  8195 net.cpp:84] Creating Layer pool5
I0523 09:47:57.846768  8195 net.cpp:406] pool5 <- fire5/concat
I0523 09:47:57.846773  8195 net.cpp:380] pool5 -> pool5
I0523 09:47:57.846781  8195 net.cpp:122] Setting up pool5
I0523 09:47:57.846784  8195 net.cpp:129] Top shape: 1 256 13 13 (43264)
I0523 09:47:57.846788  8195 net.cpp:137] Memory required for data: 24298240
I0523 09:47:57.846791  8195 layer_factory.hpp:77] Creating layer fire6/squeeze1x1_1
I0523 09:47:57.846799  8195 net.cpp:84] Creating Layer fire6/squeeze1x1_1
I0523 09:47:57.846803  8195 net.cpp:406] fire6/squeeze1x1_1 <- pool5
I0523 09:47:57.846808  8195 net.cpp:380] fire6/squeeze1x1_1 -> fire6/squeeze1x1_1
I0523 09:47:57.846840  8195 net.cpp:122] Setting up fire6/squeeze1x1_1
I0523 09:47:57.846869  8195 net.cpp:129] Top shape: 1 48 13 13 (8112)
I0523 09:47:57.846884  8195 net.cpp:137] Memory required for data: 24330688
I0523 09:47:57.846897  8195 layer_factory.hpp:77] Creating layer fire6/squeeze1x1_2
I0523 09:47:57.846912  8195 net.cpp:84] Creating Layer fire6/squeeze1x1_2
I0523 09:47:57.846923  8195 net.cpp:406] fire6/squeeze1x1_2 <- fire6/squeeze1x1_1
I0523 09:47:57.846940  8195 net.cpp:380] fire6/squeeze1x1_2 -> fire6/squeeze1x1_2
I0523 09:47:57.846953  8195 net.cpp:122] Setting up fire6/squeeze1x1_2
I0523 09:47:57.846966  8195 net.cpp:129] Top shape: 1 48 13 13 (8112)
I0523 09:47:57.846978  8195 net.cpp:137] Memory required for data: 24363136
I0523 09:47:57.846997  8195 layer_factory.hpp:77] Creating layer fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split
I0523 09:47:57.847012  8195 net.cpp:84] Creating Layer fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split
I0523 09:47:57.847023  8195 net.cpp:406] fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split <- fire6/squeeze1x1_2
I0523 09:47:57.847036  8195 net.cpp:380] fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split -> fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split_0
I0523 09:47:57.847054  8195 net.cpp:380] fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split -> fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split_1
I0523 09:47:57.847069  8195 net.cpp:122] Setting up fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split
I0523 09:47:57.847081  8195 net.cpp:129] Top shape: 1 48 13 13 (8112)
I0523 09:47:57.847095  8195 net.cpp:129] Top shape: 1 48 13 13 (8112)
I0523 09:47:57.847106  8195 net.cpp:137] Memory required for data: 24428032
I0523 09:47:57.847118  8195 layer_factory.hpp:77] Creating layer fire6/expand1x1_1
I0523 09:47:57.847136  8195 net.cpp:84] Creating Layer fire6/expand1x1_1
I0523 09:47:57.847148  8195 net.cpp:406] fire6/expand1x1_1 <- fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split_0
I0523 09:47:57.847162  8195 net.cpp:380] fire6/expand1x1_1 -> fire6/expand1x1_1
I0523 09:47:57.847204  8195 net.cpp:122] Setting up fire6/expand1x1_1
I0523 09:47:57.847219  8195 net.cpp:129] Top shape: 1 192 13 13 (32448)
I0523 09:47:57.847231  8195 net.cpp:137] Memory required for data: 24557824
I0523 09:47:57.847244  8195 layer_factory.hpp:77] Creating layer fire6/expand1x1_2
I0523 09:47:57.847259  8195 net.cpp:84] Creating Layer fire6/expand1x1_2
I0523 09:47:57.847270  8195 net.cpp:406] fire6/expand1x1_2 <- fire6/expand1x1_1
I0523 09:47:57.847285  8195 net.cpp:380] fire6/expand1x1_2 -> fire6/expand1x1_2
I0523 09:47:57.847299  8195 net.cpp:122] Setting up fire6/expand1x1_2
I0523 09:47:57.847311  8195 net.cpp:129] Top shape: 1 192 13 13 (32448)
I0523 09:47:57.847323  8195 net.cpp:137] Memory required for data: 24687616
I0523 09:47:57.847334  8195 layer_factory.hpp:77] Creating layer fire6/expand3x3_1
I0523 09:47:57.847348  8195 net.cpp:84] Creating Layer fire6/expand3x3_1
I0523 09:47:57.847360  8195 net.cpp:406] fire6/expand3x3_1 <- fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split_1
I0523 09:47:57.847383  8195 net.cpp:380] fire6/expand3x3_1 -> fire6/expand3x3_1
I0523 09:47:57.847540  8195 net.cpp:122] Setting up fire6/expand3x3_1
I0523 09:47:57.847558  8195 net.cpp:129] Top shape: 1 192 13 13 (32448)
I0523 09:47:57.847570  8195 net.cpp:137] Memory required for data: 24817408
I0523 09:47:57.847584  8195 layer_factory.hpp:77] Creating layer fire6/expand3x3_2
I0523 09:47:57.847597  8195 net.cpp:84] Creating Layer fire6/expand3x3_2
I0523 09:47:57.847609  8195 net.cpp:406] fire6/expand3x3_2 <- fire6/expand3x3_1
I0523 09:47:57.847622  8195 net.cpp:380] fire6/expand3x3_2 -> fire6/expand3x3_2
I0523 09:47:57.847636  8195 net.cpp:122] Setting up fire6/expand3x3_2
I0523 09:47:57.847648  8195 net.cpp:129] Top shape: 1 192 13 13 (32448)
I0523 09:47:57.847661  8195 net.cpp:137] Memory required for data: 24947200
I0523 09:47:57.847672  8195 layer_factory.hpp:77] Creating layer fire6/concat
I0523 09:47:57.847687  8195 net.cpp:84] Creating Layer fire6/concat
I0523 09:47:57.847698  8195 net.cpp:406] fire6/concat <- fire6/expand1x1_2
I0523 09:47:57.847710  8195 net.cpp:406] fire6/concat <- fire6/expand3x3_2
I0523 09:47:57.847723  8195 net.cpp:380] fire6/concat -> fire6/concat
I0523 09:47:57.847738  8195 net.cpp:122] Setting up fire6/concat
I0523 09:47:57.847749  8195 net.cpp:129] Top shape: 1 384 13 13 (64896)
I0523 09:47:57.847761  8195 net.cpp:137] Memory required for data: 25206784
I0523 09:47:57.847772  8195 layer_factory.hpp:77] Creating layer fire7/squeeze1x1_1
I0523 09:47:57.847786  8195 net.cpp:84] Creating Layer fire7/squeeze1x1_1
I0523 09:47:57.847797  8195 net.cpp:406] fire7/squeeze1x1_1 <- fire6/concat
I0523 09:47:57.847811  8195 net.cpp:380] fire7/squeeze1x1_1 -> fire7/squeeze1x1_1
I0523 09:47:57.847862  8195 net.cpp:122] Setting up fire7/squeeze1x1_1
I0523 09:47:57.847883  8195 net.cpp:129] Top shape: 1 48 13 13 (8112)
I0523 09:47:57.847898  8195 net.cpp:137] Memory required for data: 25239232
I0523 09:47:57.847913  8195 layer_factory.hpp:77] Creating layer fire7/squeeze1x1_2
I0523 09:47:57.847926  8195 net.cpp:84] Creating Layer fire7/squeeze1x1_2
I0523 09:47:57.847939  8195 net.cpp:406] fire7/squeeze1x1_2 <- fire7/squeeze1x1_1
I0523 09:47:57.847954  8195 net.cpp:380] fire7/squeeze1x1_2 -> fire7/squeeze1x1_2
I0523 09:47:57.847968  8195 net.cpp:122] Setting up fire7/squeeze1x1_2
I0523 09:47:57.847980  8195 net.cpp:129] Top shape: 1 48 13 13 (8112)
I0523 09:47:57.847993  8195 net.cpp:137] Memory required for data: 25271680
I0523 09:47:57.848004  8195 layer_factory.hpp:77] Creating layer fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split
I0523 09:47:57.848017  8195 net.cpp:84] Creating Layer fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split
I0523 09:47:57.848029  8195 net.cpp:406] fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split <- fire7/squeeze1x1_2
I0523 09:47:57.848043  8195 net.cpp:380] fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split -> fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split_0
I0523 09:47:57.848058  8195 net.cpp:380] fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split -> fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split_1
I0523 09:47:57.848073  8195 net.cpp:122] Setting up fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split
I0523 09:47:57.848084  8195 net.cpp:129] Top shape: 1 48 13 13 (8112)
I0523 09:47:57.848096  8195 net.cpp:129] Top shape: 1 48 13 13 (8112)
I0523 09:47:57.848109  8195 net.cpp:137] Memory required for data: 25336576
I0523 09:47:57.848120  8195 layer_factory.hpp:77] Creating layer fire7/expand1x1_1
I0523 09:47:57.848140  8195 net.cpp:84] Creating Layer fire7/expand1x1_1
I0523 09:47:57.848152  8195 net.cpp:406] fire7/expand1x1_1 <- fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split_0
I0523 09:47:57.848170  8195 net.cpp:380] fire7/expand1x1_1 -> fire7/expand1x1_1
I0523 09:47:57.848209  8195 net.cpp:122] Setting up fire7/expand1x1_1
I0523 09:47:57.848223  8195 net.cpp:129] Top shape: 1 192 13 13 (32448)
I0523 09:47:57.848237  8195 net.cpp:137] Memory required for data: 25466368
I0523 09:47:57.848249  8195 layer_factory.hpp:77] Creating layer fire7/expand1x1_2
I0523 09:47:57.848263  8195 net.cpp:84] Creating Layer fire7/expand1x1_2
I0523 09:47:57.848274  8195 net.cpp:406] fire7/expand1x1_2 <- fire7/expand1x1_1
I0523 09:47:57.848287  8195 net.cpp:380] fire7/expand1x1_2 -> fire7/expand1x1_2
I0523 09:47:57.848301  8195 net.cpp:122] Setting up fire7/expand1x1_2
I0523 09:47:57.848312  8195 net.cpp:129] Top shape: 1 192 13 13 (32448)
I0523 09:47:57.848325  8195 net.cpp:137] Memory required for data: 25596160
I0523 09:47:57.848336  8195 layer_factory.hpp:77] Creating layer fire7/expand3x3_1
I0523 09:47:57.848352  8195 net.cpp:84] Creating Layer fire7/expand3x3_1
I0523 09:47:57.848364  8195 net.cpp:406] fire7/expand3x3_1 <- fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split_1
I0523 09:47:57.848377  8195 net.cpp:380] fire7/expand3x3_1 -> fire7/expand3x3_1
I0523 09:47:57.848523  8195 net.cpp:122] Setting up fire7/expand3x3_1
I0523 09:47:57.848541  8195 net.cpp:129] Top shape: 1 192 13 13 (32448)
I0523 09:47:57.848552  8195 net.cpp:137] Memory required for data: 25725952
I0523 09:47:57.848565  8195 layer_factory.hpp:77] Creating layer fire7/expand3x3_2
I0523 09:47:57.848579  8195 net.cpp:84] Creating Layer fire7/expand3x3_2
I0523 09:47:57.848592  8195 net.cpp:406] fire7/expand3x3_2 <- fire7/expand3x3_1
I0523 09:47:57.848604  8195 net.cpp:380] fire7/expand3x3_2 -> fire7/expand3x3_2
I0523 09:47:57.848620  8195 net.cpp:122] Setting up fire7/expand3x3_2
I0523 09:47:57.848634  8195 net.cpp:129] Top shape: 1 192 13 13 (32448)
I0523 09:47:57.848645  8195 net.cpp:137] Memory required for data: 25855744
I0523 09:47:57.848656  8195 layer_factory.hpp:77] Creating layer fire7/concat
I0523 09:47:57.848670  8195 net.cpp:84] Creating Layer fire7/concat
I0523 09:47:57.848681  8195 net.cpp:406] fire7/concat <- fire7/expand1x1_2
I0523 09:47:57.848693  8195 net.cpp:406] fire7/concat <- fire7/expand3x3_2
I0523 09:47:57.848711  8195 net.cpp:380] fire7/concat -> fire7/concat
I0523 09:47:57.848727  8195 net.cpp:122] Setting up fire7/concat
I0523 09:47:57.848738  8195 net.cpp:129] Top shape: 1 384 13 13 (64896)
I0523 09:47:57.848752  8195 net.cpp:137] Memory required for data: 26115328
I0523 09:47:57.848762  8195 layer_factory.hpp:77] Creating layer fire8/squeeze1x1_1
I0523 09:47:57.848776  8195 net.cpp:84] Creating Layer fire8/squeeze1x1_1
I0523 09:47:57.848788  8195 net.cpp:406] fire8/squeeze1x1_1 <- fire7/concat
I0523 09:47:57.848801  8195 net.cpp:380] fire8/squeeze1x1_1 -> fire8/squeeze1x1_1
I0523 09:47:57.848862  8195 net.cpp:122] Setting up fire8/squeeze1x1_1
I0523 09:47:57.848877  8195 net.cpp:129] Top shape: 1 64 13 13 (10816)
I0523 09:47:57.848891  8195 net.cpp:137] Memory required for data: 26158592
I0523 09:47:57.848903  8195 layer_factory.hpp:77] Creating layer fire8/squeeze1x1_2
I0523 09:47:57.848917  8195 net.cpp:84] Creating Layer fire8/squeeze1x1_2
I0523 09:47:57.848929  8195 net.cpp:406] fire8/squeeze1x1_2 <- fire8/squeeze1x1_1
I0523 09:47:57.848942  8195 net.cpp:380] fire8/squeeze1x1_2 -> fire8/squeeze1x1_2
I0523 09:47:57.848955  8195 net.cpp:122] Setting up fire8/squeeze1x1_2
I0523 09:47:57.848968  8195 net.cpp:129] Top shape: 1 64 13 13 (10816)
I0523 09:47:57.848980  8195 net.cpp:137] Memory required for data: 26201856
I0523 09:47:57.848991  8195 layer_factory.hpp:77] Creating layer fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split
I0523 09:47:57.849006  8195 net.cpp:84] Creating Layer fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split
I0523 09:47:57.849018  8195 net.cpp:406] fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split <- fire8/squeeze1x1_2
I0523 09:47:57.849031  8195 net.cpp:380] fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split -> fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split_0
I0523 09:47:57.849061  8195 net.cpp:380] fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split -> fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split_1
I0523 09:47:57.849076  8195 net.cpp:122] Setting up fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split
I0523 09:47:57.849087  8195 net.cpp:129] Top shape: 1 64 13 13 (10816)
I0523 09:47:57.849100  8195 net.cpp:129] Top shape: 1 64 13 13 (10816)
I0523 09:47:57.849112  8195 net.cpp:137] Memory required for data: 26288384
I0523 09:47:57.849123  8195 layer_factory.hpp:77] Creating layer fire8/expand1x1_1
I0523 09:47:57.849143  8195 net.cpp:84] Creating Layer fire8/expand1x1_1
I0523 09:47:57.849155  8195 net.cpp:406] fire8/expand1x1_1 <- fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split_0
I0523 09:47:57.849171  8195 net.cpp:380] fire8/expand1x1_1 -> fire8/expand1x1_1
I0523 09:47:57.849225  8195 net.cpp:122] Setting up fire8/expand1x1_1
I0523 09:47:57.849241  8195 net.cpp:129] Top shape: 1 256 13 13 (43264)
I0523 09:47:57.849252  8195 net.cpp:137] Memory required for data: 26461440
I0523 09:47:57.849265  8195 layer_factory.hpp:77] Creating layer fire8/expand1x1_2
I0523 09:47:57.849279  8195 net.cpp:84] Creating Layer fire8/expand1x1_2
I0523 09:47:57.849290  8195 net.cpp:406] fire8/expand1x1_2 <- fire8/expand1x1_1
I0523 09:47:57.849303  8195 net.cpp:380] fire8/expand1x1_2 -> fire8/expand1x1_2
I0523 09:47:57.849318  8195 net.cpp:122] Setting up fire8/expand1x1_2
I0523 09:47:57.849329  8195 net.cpp:129] Top shape: 1 256 13 13 (43264)
I0523 09:47:57.849342  8195 net.cpp:137] Memory required for data: 26634496
I0523 09:47:57.849354  8195 layer_factory.hpp:77] Creating layer fire8/expand3x3_1
I0523 09:47:57.849368  8195 net.cpp:84] Creating Layer fire8/expand3x3_1
I0523 09:47:57.849381  8195 net.cpp:406] fire8/expand3x3_1 <- fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split_1
I0523 09:47:57.849395  8195 net.cpp:380] fire8/expand3x3_1 -> fire8/expand3x3_1
I0523 09:47:57.849625  8195 net.cpp:122] Setting up fire8/expand3x3_1
I0523 09:47:57.849642  8195 net.cpp:129] Top shape: 1 256 13 13 (43264)
I0523 09:47:57.849654  8195 net.cpp:137] Memory required for data: 26807552
I0523 09:47:57.849668  8195 layer_factory.hpp:77] Creating layer fire8/expand3x3_2
I0523 09:47:57.849684  8195 net.cpp:84] Creating Layer fire8/expand3x3_2
I0523 09:47:57.849702  8195 net.cpp:406] fire8/expand3x3_2 <- fire8/expand3x3_1
I0523 09:47:57.849715  8195 net.cpp:380] fire8/expand3x3_2 -> fire8/expand3x3_2
I0523 09:47:57.849730  8195 net.cpp:122] Setting up fire8/expand3x3_2
I0523 09:47:57.849742  8195 net.cpp:129] Top shape: 1 256 13 13 (43264)
I0523 09:47:57.849754  8195 net.cpp:137] Memory required for data: 26980608
I0523 09:47:57.849766  8195 layer_factory.hpp:77] Creating layer fire8/concat
I0523 09:47:57.849779  8195 net.cpp:84] Creating Layer fire8/concat
I0523 09:47:57.849792  8195 net.cpp:406] fire8/concat <- fire8/expand1x1_2
I0523 09:47:57.849802  8195 net.cpp:406] fire8/concat <- fire8/expand3x3_2
I0523 09:47:57.849815  8195 net.cpp:380] fire8/concat -> fire8/concat
I0523 09:47:57.849829  8195 net.cpp:122] Setting up fire8/concat
I0523 09:47:57.849841  8195 net.cpp:129] Top shape: 1 512 13 13 (86528)
I0523 09:47:57.849853  8195 net.cpp:137] Memory required for data: 27326720
I0523 09:47:57.849864  8195 layer_factory.hpp:77] Creating layer fire9/squeeze1x1_1
I0523 09:47:57.849879  8195 net.cpp:84] Creating Layer fire9/squeeze1x1_1
I0523 09:47:57.849890  8195 net.cpp:406] fire9/squeeze1x1_1 <- fire8/concat
I0523 09:47:57.849903  8195 net.cpp:380] fire9/squeeze1x1_1 -> fire9/squeeze1x1_1
I0523 09:47:57.849979  8195 net.cpp:122] Setting up fire9/squeeze1x1_1
I0523 09:47:57.849995  8195 net.cpp:129] Top shape: 1 64 13 13 (10816)
I0523 09:47:57.850008  8195 net.cpp:137] Memory required for data: 27369984
I0523 09:47:57.850021  8195 layer_factory.hpp:77] Creating layer fire9/squeeze1x1_2
I0523 09:47:57.850037  8195 net.cpp:84] Creating Layer fire9/squeeze1x1_2
I0523 09:47:57.850049  8195 net.cpp:406] fire9/squeeze1x1_2 <- fire9/squeeze1x1_1
I0523 09:47:57.850062  8195 net.cpp:380] fire9/squeeze1x1_2 -> fire9/squeeze1x1_2
I0523 09:47:57.850076  8195 net.cpp:122] Setting up fire9/squeeze1x1_2
I0523 09:47:57.850087  8195 net.cpp:129] Top shape: 1 64 13 13 (10816)
I0523 09:47:57.850100  8195 net.cpp:137] Memory required for data: 27413248
I0523 09:47:57.850111  8195 layer_factory.hpp:77] Creating layer fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split
I0523 09:47:57.850132  8195 net.cpp:84] Creating Layer fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split
I0523 09:47:57.850144  8195 net.cpp:406] fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split <- fire9/squeeze1x1_2
I0523 09:47:57.850157  8195 net.cpp:380] fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split -> fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split_0
I0523 09:47:57.850172  8195 net.cpp:380] fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split -> fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split_1
I0523 09:47:57.850186  8195 net.cpp:122] Setting up fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split
I0523 09:47:57.850198  8195 net.cpp:129] Top shape: 1 64 13 13 (10816)
I0523 09:47:57.850210  8195 net.cpp:129] Top shape: 1 64 13 13 (10816)
I0523 09:47:57.850224  8195 net.cpp:137] Memory required for data: 27499776
I0523 09:47:57.850234  8195 layer_factory.hpp:77] Creating layer fire9/expand1x1_1
I0523 09:47:57.850250  8195 net.cpp:84] Creating Layer fire9/expand1x1_1
I0523 09:47:57.850262  8195 net.cpp:406] fire9/expand1x1_1 <- fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split_0
I0523 09:47:57.850276  8195 net.cpp:380] fire9/expand1x1_1 -> fire9/expand1x1_1
I0523 09:47:57.850320  8195 net.cpp:122] Setting up fire9/expand1x1_1
I0523 09:47:57.850335  8195 net.cpp:129] Top shape: 1 256 13 13 (43264)
I0523 09:47:57.850347  8195 net.cpp:137] Memory required for data: 27672832
I0523 09:47:57.850360  8195 layer_factory.hpp:77] Creating layer fire9/expand1x1_2
I0523 09:47:57.850375  8195 net.cpp:84] Creating Layer fire9/expand1x1_2
I0523 09:47:57.850387  8195 net.cpp:406] fire9/expand1x1_2 <- fire9/expand1x1_1
I0523 09:47:57.850400  8195 net.cpp:380] fire9/expand1x1_2 -> fire9/expand1x1_2
I0523 09:47:57.850417  8195 net.cpp:122] Setting up fire9/expand1x1_2
I0523 09:47:57.850430  8195 net.cpp:129] Top shape: 1 256 13 13 (43264)
I0523 09:47:57.850442  8195 net.cpp:137] Memory required for data: 27845888
I0523 09:47:57.850453  8195 layer_factory.hpp:77] Creating layer fire9/expand3x3_1
I0523 09:47:57.850471  8195 net.cpp:84] Creating Layer fire9/expand3x3_1
I0523 09:47:57.850483  8195 net.cpp:406] fire9/expand3x3_1 <- fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split_1
I0523 09:47:57.850498  8195 net.cpp:380] fire9/expand3x3_1 -> fire9/expand3x3_1
I0523 09:47:57.850726  8195 net.cpp:122] Setting up fire9/expand3x3_1
I0523 09:47:57.850744  8195 net.cpp:129] Top shape: 1 256 13 13 (43264)
I0523 09:47:57.850756  8195 net.cpp:137] Memory required for data: 28018944
I0523 09:47:57.850769  8195 layer_factory.hpp:77] Creating layer fire9/expand3x3_2
I0523 09:47:57.850785  8195 net.cpp:84] Creating Layer fire9/expand3x3_2
I0523 09:47:57.850797  8195 net.cpp:406] fire9/expand3x3_2 <- fire9/expand3x3_1
I0523 09:47:57.850811  8195 net.cpp:380] fire9/expand3x3_2 -> fire9/expand3x3_2
I0523 09:47:57.850824  8195 net.cpp:122] Setting up fire9/expand3x3_2
I0523 09:47:57.850836  8195 net.cpp:129] Top shape: 1 256 13 13 (43264)
I0523 09:47:57.850849  8195 net.cpp:137] Memory required for data: 28192000
I0523 09:47:57.850860  8195 layer_factory.hpp:77] Creating layer fire9/concat_1
I0523 09:47:57.850875  8195 net.cpp:84] Creating Layer fire9/concat_1
I0523 09:47:57.850888  8195 net.cpp:406] fire9/concat_1 <- fire9/expand1x1_2
I0523 09:47:57.850899  8195 net.cpp:406] fire9/concat_1 <- fire9/expand3x3_2
I0523 09:47:57.850912  8195 net.cpp:380] fire9/concat_1 -> fire9/concat_1
I0523 09:47:57.850926  8195 net.cpp:122] Setting up fire9/concat_1
I0523 09:47:57.850937  8195 net.cpp:129] Top shape: 1 512 13 13 (86528)
I0523 09:47:57.850950  8195 net.cpp:137] Memory required for data: 28538112
I0523 09:47:57.850961  8195 layer_factory.hpp:77] Creating layer fire9/concat_2__fire9/concat_mask
I0523 09:47:57.850981  8195 net.cpp:84] Creating Layer fire9/concat_2__fire9/concat_mask
I0523 09:47:57.850993  8195 net.cpp:406] fire9/concat_2__fire9/concat_mask <- fire9/concat_1
I0523 09:47:57.851008  8195 net.cpp:380] fire9/concat_2__fire9/concat_mask -> fire9/concat_2
I0523 09:47:57.851024  8195 net.cpp:122] Setting up fire9/concat_2__fire9/concat_mask
I0523 09:47:57.851037  8195 net.cpp:129] Top shape: 1 512 13 13 (86528)
I0523 09:47:57.851048  8195 net.cpp:137] Memory required for data: 28884224
I0523 09:47:57.851059  8195 layer_factory.hpp:77] Creating layer conv10_1
I0523 09:47:57.851075  8195 net.cpp:84] Creating Layer conv10_1
I0523 09:47:57.851088  8195 net.cpp:406] conv10_1 <- fire9/concat_2
I0523 09:47:57.851101  8195 net.cpp:380] conv10_1 -> conv10_1
I0523 09:47:57.852138  8195 net.cpp:122] Setting up conv10_1
I0523 09:47:57.852155  8195 net.cpp:129] Top shape: 1 1000 13 13 (169000)
I0523 09:47:57.852162  8195 net.cpp:137] Memory required for data: 29560224
I0523 09:47:57.852170  8195 layer_factory.hpp:77] Creating layer conv10_2
I0523 09:47:57.852181  8195 net.cpp:84] Creating Layer conv10_2
I0523 09:47:57.852185  8195 net.cpp:406] conv10_2 <- conv10_1
I0523 09:47:57.852191  8195 net.cpp:380] conv10_2 -> conv10_2
I0523 09:47:57.852200  8195 net.cpp:122] Setting up conv10_2
I0523 09:47:57.852202  8195 net.cpp:129] Top shape: 1 1000 13 13 (169000)
I0523 09:47:57.852207  8195 net.cpp:137] Memory required for data: 30236224
I0523 09:47:57.852210  8195 layer_factory.hpp:77] Creating layer pool10
I0523 09:47:57.852221  8195 net.cpp:84] Creating Layer pool10
I0523 09:47:57.852224  8195 net.cpp:406] pool10 <- conv10_2
I0523 09:47:57.852229  8195 net.cpp:380] pool10 -> pool10
I0523 09:47:57.852237  8195 net.cpp:122] Setting up pool10
I0523 09:47:57.852241  8195 net.cpp:129] Top shape: 1 1000 1 1 (1000)
I0523 09:47:57.852246  8195 net.cpp:137] Memory required for data: 30240224
I0523 09:47:57.852248  8195 layer_factory.hpp:77] Creating layer softmaxout
I0523 09:47:57.852254  8195 net.cpp:84] Creating Layer softmaxout
I0523 09:47:57.852257  8195 net.cpp:406] softmaxout <- pool10
I0523 09:47:57.852262  8195 net.cpp:380] softmaxout -> softmaxout
I0523 09:47:57.852272  8195 net.cpp:122] Setting up softmaxout
I0523 09:47:57.852277  8195 net.cpp:129] Top shape: 1 1000 1 1 (1000)
I0523 09:47:57.852280  8195 net.cpp:137] Memory required for data: 30244224
I0523 09:47:57.852284  8195 net.cpp:200] softmaxout does not need backward computation.
I0523 09:47:57.852288  8195 net.cpp:200] pool10 does not need backward computation.
I0523 09:47:57.852291  8195 net.cpp:200] conv10_2 does not need backward computation.
I0523 09:47:57.852295  8195 net.cpp:200] conv10_1 does not need backward computation.
I0523 09:47:57.852299  8195 net.cpp:200] fire9/concat_2__fire9/concat_mask does not need backward computation.
I0523 09:47:57.852303  8195 net.cpp:200] fire9/concat_1 does not need backward computation.
I0523 09:47:57.852308  8195 net.cpp:200] fire9/expand3x3_2 does not need backward computation.
I0523 09:47:57.852311  8195 net.cpp:200] fire9/expand3x3_1 does not need backward computation.
I0523 09:47:57.852315  8195 net.cpp:200] fire9/expand1x1_2 does not need backward computation.
I0523 09:47:57.852319  8195 net.cpp:200] fire9/expand1x1_1 does not need backward computation.
I0523 09:47:57.852324  8195 net.cpp:200] fire9/squeeze1x1_2_fire9/squeeze1x1_2_0_split does not need backward computation.
I0523 09:47:57.852327  8195 net.cpp:200] fire9/squeeze1x1_2 does not need backward computation.
I0523 09:47:57.852331  8195 net.cpp:200] fire9/squeeze1x1_1 does not need backward computation.
I0523 09:47:57.852335  8195 net.cpp:200] fire8/concat does not need backward computation.
I0523 09:47:57.852339  8195 net.cpp:200] fire8/expand3x3_2 does not need backward computation.
I0523 09:47:57.852344  8195 net.cpp:200] fire8/expand3x3_1 does not need backward computation.
I0523 09:47:57.852349  8195 net.cpp:200] fire8/expand1x1_2 does not need backward computation.
I0523 09:47:57.852352  8195 net.cpp:200] fire8/expand1x1_1 does not need backward computation.
I0523 09:47:57.852356  8195 net.cpp:200] fire8/squeeze1x1_2_fire8/squeeze1x1_2_0_split does not need backward computation.
I0523 09:47:57.852360  8195 net.cpp:200] fire8/squeeze1x1_2 does not need backward computation.
I0523 09:47:57.852365  8195 net.cpp:200] fire8/squeeze1x1_1 does not need backward computation.
I0523 09:47:57.852368  8195 net.cpp:200] fire7/concat does not need backward computation.
I0523 09:47:57.852372  8195 net.cpp:200] fire7/expand3x3_2 does not need backward computation.
I0523 09:47:57.852376  8195 net.cpp:200] fire7/expand3x3_1 does not need backward computation.
I0523 09:47:57.852381  8195 net.cpp:200] fire7/expand1x1_2 does not need backward computation.
I0523 09:47:57.852385  8195 net.cpp:200] fire7/expand1x1_1 does not need backward computation.
I0523 09:47:57.852391  8195 net.cpp:200] fire7/squeeze1x1_2_fire7/squeeze1x1_2_0_split does not need backward computation.
I0523 09:47:57.852396  8195 net.cpp:200] fire7/squeeze1x1_2 does not need backward computation.
I0523 09:47:57.852399  8195 net.cpp:200] fire7/squeeze1x1_1 does not need backward computation.
I0523 09:47:57.852403  8195 net.cpp:200] fire6/concat does not need backward computation.
I0523 09:47:57.852407  8195 net.cpp:200] fire6/expand3x3_2 does not need backward computation.
I0523 09:47:57.852412  8195 net.cpp:200] fire6/expand3x3_1 does not need backward computation.
I0523 09:47:57.852416  8195 net.cpp:200] fire6/expand1x1_2 does not need backward computation.
I0523 09:47:57.852421  8195 net.cpp:200] fire6/expand1x1_1 does not need backward computation.
I0523 09:47:57.852424  8195 net.cpp:200] fire6/squeeze1x1_2_fire6/squeeze1x1_2_0_split does not need backward computation.
I0523 09:47:57.852430  8195 net.cpp:200] fire6/squeeze1x1_2 does not need backward computation.
I0523 09:47:57.852435  8195 net.cpp:200] fire6/squeeze1x1_1 does not need backward computation.
I0523 09:47:57.852439  8195 net.cpp:200] pool5 does not need backward computation.
I0523 09:47:57.852443  8195 net.cpp:200] fire5/concat does not need backward computation.
I0523 09:47:57.852448  8195 net.cpp:200] fire5/expand3x3_2 does not need backward computation.
I0523 09:47:57.852452  8195 net.cpp:200] fire5/expand3x3_1 does not need backward computation.
I0523 09:47:57.852457  8195 net.cpp:200] fire5/expand1x1_2 does not need backward computation.
I0523 09:47:57.852460  8195 net.cpp:200] fire5/expand1x1_1 does not need backward computation.
I0523 09:47:57.852465  8195 net.cpp:200] fire5/squeeze1x1_2_fire5/squeeze1x1_2_0_split does not need backward computation.
I0523 09:47:57.852469  8195 net.cpp:200] fire5/squeeze1x1_2 does not need backward computation.
I0523 09:47:57.852473  8195 net.cpp:200] fire5/squeeze1x1_1 does not need backward computation.
I0523 09:47:57.852478  8195 net.cpp:200] fire4/concat does not need backward computation.
I0523 09:47:57.852483  8195 net.cpp:200] fire4/expand3x3_2 does not need backward computation.
I0523 09:47:57.852486  8195 net.cpp:200] fire4/expand3x3_1 does not need backward computation.
I0523 09:47:57.852490  8195 net.cpp:200] fire4/expand1x1_2 does not need backward computation.
I0523 09:47:57.852494  8195 net.cpp:200] fire4/expand1x1_1 does not need backward computation.
I0523 09:47:57.852499  8195 net.cpp:200] fire4/squeeze1x1_2_fire4/squeeze1x1_2_0_split does not need backward computation.
I0523 09:47:57.852502  8195 net.cpp:200] fire4/squeeze1x1_2 does not need backward computation.
I0523 09:47:57.852506  8195 net.cpp:200] fire4/squeeze1x1_1 does not need backward computation.
I0523 09:47:57.852510  8195 net.cpp:200] pool3 does not need backward computation.
I0523 09:47:57.852514  8195 net.cpp:200] fire3/concat does not need backward computation.
I0523 09:47:57.852519  8195 net.cpp:200] fire3/expand3x3_2 does not need backward computation.
I0523 09:47:57.852524  8195 net.cpp:200] fire3/expand3x3_1 does not need backward computation.
I0523 09:47:57.852527  8195 net.cpp:200] fire3/expand1x1_2 does not need backward computation.
I0523 09:47:57.852531  8195 net.cpp:200] fire3/expand1x1_1 does not need backward computation.
I0523 09:47:57.852535  8195 net.cpp:200] fire3/squeeze1x1_2_fire3/squeeze1x1_2_0_split does not need backward computation.
I0523 09:47:57.852540  8195 net.cpp:200] fire3/squeeze1x1_2 does not need backward computation.
I0523 09:47:57.852543  8195 net.cpp:200] fire3/squeeze1x1_1 does not need backward computation.
I0523 09:47:57.852547  8195 net.cpp:200] fire2/concat does not need backward computation.
I0523 09:47:57.852552  8195 net.cpp:200] fire2/expand3x3_2 does not need backward computation.
I0523 09:47:57.852556  8195 net.cpp:200] fire2/expand3x3_1 does not need backward computation.
I0523 09:47:57.852560  8195 net.cpp:200] fire2/expand1x1_2 does not need backward computation.
I0523 09:47:57.852564  8195 net.cpp:200] fire2/expand1x1_1 does not need backward computation.
I0523 09:47:57.852571  8195 net.cpp:200] fire2/squeeze1x1_2_fire2/squeeze1x1_2_0_split does not need backward computation.
I0523 09:47:57.852576  8195 net.cpp:200] fire2/squeeze1x1_2 does not need backward computation.
I0523 09:47:57.852579  8195 net.cpp:200] fire2/squeeze1x1_1 does not need backward computation.
I0523 09:47:57.852583  8195 net.cpp:200] pool1 does not need backward computation.
I0523 09:47:57.852587  8195 net.cpp:200] conv1_2 does not need backward computation.
I0523 09:47:57.852591  8195 net.cpp:200] conv1_1 does not need backward computation.
I0523 09:47:57.852596  8195 net.cpp:200] data does not need backward computation.
I0523 09:47:57.852598  8195 net.cpp:242] This network produces output softmaxout
I0523 09:47:57.852644  8195 net.cpp:255] Network initialization done.
/home/aswin/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:526: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint8 = np.dtype([("qint8", np.int8, 1)])
/home/aswin/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:527: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint8 = np.dtype([("quint8", np.uint8, 1)])
/home/aswin/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:528: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint16 = np.dtype([("qint16", np.int16, 1)])
/home/aswin/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:529: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_quint16 = np.dtype([("quint16", np.uint16, 1)])
/home/aswin/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:530: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  _np_qint32 = np.dtype([("qint32", np.int32, 1)])
/home/aswin/.local/lib/python3.6/site-packages/tensorflow/python/framework/dtypes.py:535: FutureWarning: Passing (type, 1) or '1type' as a synonym of type is deprecated; in a future version of numpy, it will be understood as (type, (1,)) / '(1,)type'.
  np_resource = np.dtype([("resource", np.ubyte, 1)])

  warnings.warn(GET_NEXT_CALL_WARNING_MESSAGE)


Inference for accuracy check.
total iteration is 2008
steps = 0 step
steps = 10 step
steps = 20 step
steps = 30 step
steps = 40 step
steps = 50 step
steps = 60 step
steps = 70 step
steps = 80 step
steps = 90 step
steps = 100 step
steps = 110 step
steps = 120 step
steps = 130 step
steps = 140 step
steps = 150 step
steps = 160 step
steps = 170 step
steps = 180 step
steps = 190 step
steps = 200 step
steps = 210 step
steps = 220 step
steps = 230 step
steps = 240 step
steps = 250 step
steps = 260 step
steps = 270 step
steps = 280 step
steps = 290 step
steps = 300 step
steps = 310 step
steps = 320 step
steps = 330 step
steps = 340 step
steps = 350 step
steps = 360 step
steps = 370 step
steps = 380 step
steps = 390 step
steps = 400 step
steps = 410 step
steps = 420 step
steps = 430 step
steps = 440 step
steps = 450 step
steps = 460 step
steps = 470 step
steps = 480 step
steps = 490 step
steps = 500 step
steps = 510 step
steps = 520 step
steps = 530 step
steps = 540 step
steps = 550 step
steps = 560 step
steps = 570 step
steps = 580 step
steps = 590 step
steps = 600 step
steps = 610 step
steps = 620 step
steps = 630 step
steps = 640 step
steps = 650 step
steps = 660 step
steps = 670 step
steps = 680 step
steps = 690 step
steps = 700 step
steps = 710 step
steps = 720 step
steps = 730 step
steps = 740 step
steps = 750 step
steps = 760 step
steps = 770 step
steps = 780 step
steps = 790 step
steps = 800 step
steps = 810 step
steps = 820 step
steps = 830 step
steps = 840 step
steps = 850 step
steps = 860 step
steps = 870 step
steps = 880 step
steps = 890 step
steps = 900 step
steps = 910 step
steps = 920 step
steps = 930 step
steps = 940 step
steps = 950 step
steps = 960 step
steps = 970 step
steps = 980 step
steps = 990 step
steps = 1000 step
steps = 1010 step
steps = 1020 step
steps = 1030 step
steps = 1040 step
steps = 1050 step
steps = 1060 step
steps = 1070 step
steps = 1080 step
steps = 1090 step
steps = 1100 step
steps = 1110 step
steps = 1120 step
steps = 1130 step
steps = 1140 step
steps = 1150 step
steps = 1160 step
steps = 1170 step
steps = 1180 step
steps = 1190 step
steps = 1200 step
steps = 1210 step
steps = 1220 step
steps = 1230 step
steps = 1240 step
steps = 1250 step
steps = 1260 step
steps = 1270 step
steps = 1280 step
steps = 1290 step
steps = 1300 step
steps = 1310 step
steps = 1320 step
steps = 1330 step
steps = 1340 step
steps = 1350 step
steps = 1360 step
steps = 1370 step
steps = 1380 step
steps = 1390 step
steps = 1400 step
steps = 1410 step
steps = 1420 step
steps = 1430 step
steps = 1440 step
steps = 1450 step
steps = 1460 step
steps = 1470 step
steps = 1480 step
steps = 1490 step
steps = 1500 step
steps = 1510 step
steps = 1520 step
steps = 1530 step
steps = 1540 step
steps = 1550 step
steps = 1560 step
steps = 1570 step
steps = 1580 step
steps = 1590 step
steps = 1600 step
steps = 1610 step
steps = 1620 step
steps = 1630 step
steps = 1640 step
steps = 1650 step
steps = 1660 step
steps = 1670 step
steps = 1680 step
steps = 1690 step
steps = 1700 step
steps = 1710 step
steps = 1720 step
steps = 1730 step
steps = 1740 step
steps = 1750 step
steps = 1760 step
steps = 1770 step
steps = 1780 step
steps = 1790 step
steps = 1800 step
steps = 1810 step
steps = 1820 step
steps = 1830 step
steps = 1840 step
steps = 1850 step
steps = 1860 step
steps = 1870 step
steps = 1880 step
steps = 1890 step
steps = 1900 step
steps = 1910 step
steps = 1920 step
steps = 1930 step
steps = 1940 step
steps = 1950 step
steps = 1960 step
steps = 1970 step
steps = 1980 step
steps = 1990 step
steps = 2000 step
creating index...
index created!
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.47s).
Accumulating evaluation results...
DONE (t=0.08s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 1.000
{   'DetectionBoxes_Precision/mAP': 1.0,
    'DetectionBoxes_Precision/mAP (large)': 1.0,
    'DetectionBoxes_Precision/mAP (medium)': 1.0,
    'DetectionBoxes_Precision/mAP (small)': 1.0,
    'DetectionBoxes_Precision/mAP@.50IOU': 1.0,
    'DetectionBoxes_Precision/mAP@.75IOU': 1.0,
    'DetectionBoxes_Recall/AR@1': 1.0,
    'DetectionBoxes_Recall/AR@10': 1.0,
    'DetectionBoxes_Recall/AR@100': 1.0,
    'DetectionBoxes_Recall/AR@100 (large)': 1.0,
    'DetectionBoxes_Recall/AR@100 (medium)': 1.0,
    'DetectionBoxes_Recall/AR@100 (small)': 1.0}
{   'DetectionBoxes_Precision/mAP': 1.0,
    'DetectionBoxes_Precision/mAP (large)': 1.0,
    'DetectionBoxes_Precision/mAP (medium)': 1.0,
    'DetectionBoxes_Precision/mAP (small)': 1.0,
    'DetectionBoxes_Precision/mAP@.50IOU': 1.0,
    'DetectionBoxes_Precision/mAP@.75IOU': 1.0,
    'DetectionBoxes_Recall/AR@1': 1.0,
    'DetectionBoxes_Recall/AR@10': 1.0,
    'DetectionBoxes_Recall/AR@100': 1.0,
    'DetectionBoxes_Recall/AR@100 (large)': 1.0,
    'DetectionBoxes_Recall/AR@100 (medium)': 1.0,
    'DetectionBoxes_Recall/AR@100 (small)': 1.0}
{   0: {   'boxes': array([[141, 150, 284, 229]]),
           'classes': array([1]),
           'scores': array([4.99741234e+08])},
    1: {   'boxes': array([[201, 285, 331, 327]]),
           'classes': array([1]),
           'scores': array([4.99650699e+08])},
    2: {   'boxes': array([[198, 258, 329, 297]]),
           'classes': array([1]),
           'scores': array([4.99770734e+08])},
    3: {   'boxes': array([[ 62, 185, 199, 279]]),
           'classes': array([1]),
           'scores': array([5.00046972e+08])},
    4: {   'boxes': array([[ 36, 210, 482, 336]]),
           'classes': array([1]),
           'scores': array([4.99732754e+08])},
    5: {   'boxes': array([[ 82,  46, 365, 170]]),
           'classes': array([1]),
           'scores': array([4.99857927e+08])},
    6: {   'boxes': array([[181,  11, 419, 142]]),
           'classes': array([1]),
           'scores': array([4.99684517e+08])},
    7: {   'boxes': array([[  1,   2, 369, 117]]),
           'classes': array([1]),
           'scores': array([4.99716021e+08])},
    8: {   'boxes': array([[  2,   3, 462, 243]]),
           'classes': array([1]),
           'scores': array([4.99622869e+08])},
    9: {   'boxes': array([[  1, 225, 486, 334]]),
           'classes': array([1]),
           'scores': array([4.99709372e+08])},
    10: {   'boxes': array([[160,  51, 292, 150]]),
            'classes': array([1]),
            'scores': array([4.99691566e+08])},
    11: {   'boxes': array([[138, 295, 290, 450]]),
            'classes': array([1]),
            'scores': array([4.99613606e+08])},
    12: {   'boxes': array([[180, 195, 229, 213]]),
            'classes': array([1]),
            'scores': array([4.99755086e+08])},
    13: {   'boxes': array([[189,  26, 238,  44]]),
            'classes': array([1]),
            'scores': array([4.99689875e+08])},
    14: {   'boxes': array([[ 96,   1, 361, 191]]),
            'classes': array([1]),
            'scores': array([4.99680136e+08])},
    15: {   'boxes': array([[ 98, 218, 318, 465]]),
            'classes': array([1]),
            'scores': array([4.99719267e+08])},
    16: {   'boxes': array([[195, 468, 317, 500]]),
            'classes': array([1]),
            'scores': array([4.99648729e+08])},
    17: {   'boxes': array([[ 92, 216, 302, 307]]),
            'classes': array([1]),
            'scores': array([4.99724903e+08])},
    18: {   'boxes': array([[148, 164, 244, 227]]),
            'classes': array([1]),
            'scores': array([5.00123742e+08])},
    19: {   'boxes': array([[  1,   2, 500, 302]]),
            'classes': array([1]),
            'scores': array([5.0003195e+08])},
    20: {   'boxes': array([[ 96,  68, 375, 293]]),
            'classes': array([1]),
            'scores': array([4.99719677e+08])},
    21: {   'boxes': array([[ 71,   1, 332,  87]]),
            'classes': array([1]),
            'scores': array([4.99670243e+08])},
    22: {   'boxes': array([[ 68, 185, 197, 259]]),
            'classes': array([1]),
            'scores': array([5.00064514e+08])},
    23: {   'boxes': array([[ 64, 286, 238, 406]]),
            'classes': array([1]),
            'scores': array([4.99926602e+08])},
    24: {   'boxes': array([[ 61, 415, 195, 465]]),
            'classes': array([1]),
            'scores': array([4.99935015e+08])},
    25: {   'boxes': array([[187, 209, 230, 228]]),
            'classes': array([1]),
            'scores': array([4.99742795e+08])},
    26: {   'boxes': array([[182, 242, 259, 274]]),
            'classes': array([1]),
            'scores': array([5.00089268e+08])},
    27: {   'boxes': array([[188, 269, 259, 295]]),
            'classes': array([1]),
            'scores': array([4.9962998e+08])},
    28: {   'boxes': array([[143,  22, 500, 146]]),
            'classes': array([1]),
            'scores': array([4.9977882e+08])},
    29: {   'boxes': array([[156, 198, 231, 341]]),
            'classes': array([1]),
            'scores': array([5.00164056e+08])},
    30: {   'boxes': array([[145, 349, 266, 451]]),
            'classes': array([1]),
            'scores': array([4.99735758e+08])},
    31: {   'boxes': array([[162,  61, 333, 170]]),
            'classes': array([1]),
            'scores': array([4.99701971e+08])},
    32: {   'boxes': array([[166,  29, 213,  73]]),
            'classes': array([1]),
            'scores': array([4.99695506e+08])},
    33: {   'boxes': array([[246, 298, 333, 433]]),
            'classes': array([1]),
            'scores': array([4.99744239e+08])},
    34: {   'boxes': array([[215, 196, 330, 332]]),
            'classes': array([1]),
            'scores': array([4.99849494e+08])},
    35: {   'boxes': array([[298, 270, 333, 324]]),
            'classes': array([1]),
            'scores': array([4.99284662e+08])},
    36: {   'boxes': array([[  2, 380, 135, 435]]),
            'classes': array([1]),
            'scores': array([4.99685336e+08])},
    37: {   'boxes': array([[ 14, 138, 115, 193]]),
            'classes': array([1]),
            'scores': array([4.99669674e+08])},
    38: {   'boxes': array([[ 20,   1, 204,  75]]),
            'classes': array([1]),
            'scores': array([4.99779151e+08])},
    39: {   'boxes': array([[ 48,  60, 156, 114]]),
            'classes': array([1]),
            'scores': array([4.99606248e+08])},
    40: {   'boxes': array([[ 51,  32, 162,  83]]),
            'classes': array([1]),
            'scores': array([4.99668447e+08])},
    41: {   'boxes': array([[  7,  20, 355, 183]]),
            'classes': array([1]),
            'scores': array([4.99678658e+08])},
    42: {   'boxes': array([[215,  98, 374, 429]]),
            'classes': array([1]),
            'scores': array([4.99653029e+08])},
    43: {   'boxes': array([[140, 332, 366, 455]]),
            'classes': array([1]),
            'scores': array([4.99968323e+08])},
    44: {   'boxes': array([[ 64,  40, 333, 400]]),
            'classes': array([1]),
            'scores': array([4.99578688e+08])},
    45: {   'boxes': array([[ 75, 291, 333, 442]]),
            'classes': array([1]),
            'scores': array([4.9978305e+08])},
    46: {   'boxes': array([[  7,   2, 333, 283]]),
            'classes': array([1]),
            'scores': array([4.99851769e+08])},
    47: {   'boxes': array([[233,  79, 314, 110]]),
            'classes': array([1]),
            'scores': array([4.99620811e+08])},
    48: {   'boxes': array([[231,  54, 344,  86]]),
            'classes': array([1]),
            'scores': array([4.99500274e+08])},
    49: {   'boxes': array([[233,  31, 303,  50]]),
            'classes': array([1]),
            'scores': array([4.99379646e+08])},
    50: {   'boxes': array([[101, 277, 260, 354]]),
            'classes': array([1]),
            'scores': array([5.00076182e+08])},
    51: {   'boxes': array([[188, 436, 223, 467]]),
            'classes': array([1]),
            'scores': array([4.99817063e+08])},
    52: {   'boxes': array([[209, 382, 272, 415]]),
            'classes': array([1]),
            'scores': array([4.99666023e+08])},
    53: {   'boxes': array([[213, 353, 276, 383]]),
            'classes': array([1]),
            'scores': array([4.99891957e+08])},
    54: {   'boxes': array([[  1,   1, 363,  66]]),
            'classes': array([1]),
            'scores': array([4.9982158e+08])},
    55: {   'boxes': array([[  1,  74, 462, 272]]),
            'classes': array([1]),
            'scores': array([4.99672798e+08])},
    56: {   'boxes': array([[ 19, 252, 487, 334]]),
            'classes': array([1]),
            'scores': array([4.99799908e+08])},
    57: {   'boxes': array([[125, 124, 153, 172]]),
            'classes': array([1]),
            'scores': array([4.99778114e+08])},
    58: {   'boxes': array([[121, 184, 154, 226]]),
            'classes': array([1]),
            'scores': array([4.99701825e+08])},
    59: {   'boxes': array([[ 18,  73, 500, 406]]),
            'classes': array([1]),
            'scores': array([4.99799739e+08])},
    60: {   'boxes': array([[  1,  22, 101,  75]]),
            'classes': array([1]),
            'scores': array([4.99698836e+08])},
    61: {   'boxes': array([[ 56,   6, 285, 154]]),
            'classes': array([1]),
            'scores': array([4.99707477e+08])},
    62: {   'boxes': array([[ 58, 145, 258, 283]]),
            'classes': array([1]),
            'scores': array([4.99632792e+08])},
    63: {   'boxes': array([[ 54, 224, 332, 440]]),
            'classes': array([1]),
            'scores': array([4.99810484e+08])},
    64: {   'boxes': array([[ 96, 363, 332, 500]]),
            'classes': array([1]),
            'scores': array([4.99683304e+08])},
    65: {   'boxes': array([[208, 155, 408, 252]]),
            'classes': array([1]),
            'scores': array([4.99777739e+08])},
    66: {   'boxes': array([[ 54, 353, 108, 379]]),
            'classes': array([1]),
            'scores': array([4.99668828e+08])},
    67: {   'boxes': array([[ 51, 415, 109, 436]]),
            'classes': array([1]),
            'scores': array([4.99731157e+08])},
    68: {   'boxes': array([[ 54, 476,  98, 492]]),
            'classes': array([1]),
            'scores': array([4.99697172e+08])},
    69: {   'boxes': array([[ 51, 489, 100, 500]]),
            'classes': array([1]),
            'scores': array([4.99667046e+08])},
    70: {   'boxes': array([[ 48, 234, 124, 286]]),
            'classes': array([1]),
            'scores': array([4.99745291e+08])},
    71: {   'boxes': array([[143, 196, 369, 309]]),
            'classes': array([1]),
            'scores': array([4.99687068e+08])},
    72: {   'boxes': array([[ 22,  52, 328, 308]]),
            'classes': array([1]),
            'scores': array([4.99505752e+08])},
    73: {   'boxes': array([[ 49, 292, 370, 446]]),
            'classes': array([1]),
            'scores': array([4.9957128e+08])},
    74: {   'boxes': array([[ 54,  89, 386, 500]]),
            'classes': array([1]),
            'scores': array([4.99746719e+08])},
    75: {   'boxes': array([[  3, 102,  50, 161]]),
            'classes': array([1]),
            'scores': array([5.00158102e+08])},
    76: {   'boxes': array([[ 39, 419,  81, 444]]),
            'classes': array([1]),
            'scores': array([4.99797956e+08])},
    77: {   'boxes': array([[ 36, 454,  71, 487]]),
            'classes': array([1]),
            'scores': array([4.99649456e+08])},
    78: {   'boxes': array([[ 91,   3, 206,  43]]),
            'classes': array([1]),
            'scores': array([4.99447674e+08])},
    79: {   'boxes': array([[ 28,   4, 372, 461]]),
            'classes': array([1]),
            'scores': array([5.00160568e+08])},
    80: {   'boxes': array([[ 21,  94, 491, 375]]),
            'classes': array([1]),
            'scores': array([4.99619203e+08])},
    81: {   'boxes': array([[ 64, 106, 297, 270]]),
            'classes': array([1]),
            'scores': array([4.99710854e+08])},
    82: {   'boxes': array([[  5, 143, 333, 426]]),
            'classes': array([1]),
            'scores': array([4.99763742e+08])},
    83: {   'boxes': array([[150, 344, 375, 500]]),
            'classes': array([1]),
            'scores': array([4.9979736e+08])},
    84: {   'boxes': array([[ 85, 321, 362, 456]]),
            'classes': array([1]),
            'scores': array([4.9981271e+08])},
    85: {   'boxes': array([[ 92, 304, 209, 391]]),
            'classes': array([1]),
            'scores': array([4.99725815e+08])},
    86: {   'boxes': array([[ 88, 280, 172, 337]]),
            'classes': array([1]),
            'scores': array([4.99786399e+08])},
    87: {   'boxes': array([[ 67, 163, 163, 261]]),
            'classes': array([1]),
            'scores': array([4.99771827e+08])},
    88: {   'boxes': array([[ 94, 125, 166, 164]]),
            'classes': array([1]),
            'scores': array([4.99656574e+08])},
    89: {   'boxes': array([[ 87,  75, 231, 158]]),
            'classes': array([1]),
            'scores': array([4.99791021e+08])},
    90: {   'boxes': array([[112,   1, 375,  97]]),
            'classes': array([1]),
            'scores': array([4.99914614e+08])},
    91: {   'boxes': array([[241, 314, 269, 325]]),
            'classes': array([1]),
            'scores': array([4.9993935e+08])},
    92: {   'boxes': array([[242, 278, 268, 284]]),
            'classes': array([1]),
            'scores': array([4.99667925e+08])},
    93: {   'boxes': array([[245, 273, 257, 278]]),
            'classes': array([1]),
            'scores': array([4.9967063e+08])},
    94: {   'boxes': array([[242, 256, 270, 265]]),
            'classes': array([1]),
            'scores': array([5.00020988e+08])},
    95: {   'boxes': array([[244, 247, 268, 256]]),
            'classes': array([1]),
            'scores': array([4.99762435e+08])},
    96: {   'boxes': array([[245, 206, 270, 214]]),
            'classes': array([1]),
            'scores': array([4.99653821e+08])},
    97: {   'boxes': array([[243, 187, 277, 196]]),
            'classes': array([1]),
            'scores': array([4.99636801e+08])},
    98: {   'boxes': array([[245, 178, 273, 185]]),
            'classes': array([1]),
            'scores': array([4.99608943e+08])},
    99: {   'boxes': array([[ 64, 116, 375, 356]]),
            'classes': array([1]),
            'scores': array([4.99783299e+08])},
    100: {   'boxes': array([[  4,  80, 375, 500]]),
             'classes': array([1]),
             'scores': array([4.9970565e+08])},
    101: {   'boxes': array([[ 29,   1, 375, 227]]),
             'classes': array([1]),
             'scores': array([4.99689136e+08])},
    102: {   'boxes': array([[ 19, 115, 136, 203]]),
             'classes': array([1]),
             'scores': array([4.99701419e+08])},
    103: {   'boxes': array([[ 77, 279, 132, 298]]),
             'classes': array([1]),
             'scores': array([4.99920467e+08])},
    104: {   'boxes': array([[ 19,  28, 167, 103]]),
             'classes': array([1]),
             'scores': array([4.99914665e+08])},
    105: {   'boxes': array([[ 54,  16, 249, 178]]),
             'classes': array([1]),
             'scores': array([4.9973594e+08])},
    106: {   'boxes': array([[ 23, 147, 249, 245]]),
             'classes': array([1]),
             'scores': array([4.99851476e+08])},
    107: {   'boxes': array([[ 19, 227, 249, 400]]),
             'classes': array([1]),
             'scores': array([4.99987981e+08])},
    108: {   'boxes': array([[ 13, 391, 249, 500]]),
             'classes': array([1]),
             'scores': array([4.99646424e+08])},
    109: {   'boxes': array([[211, 443, 235, 468]]),
             'classes': array([1]),
             'scores': array([4.99752239e+08])},
    110: {   'boxes': array([[214, 323, 226, 335]]),
             'classes': array([1]),
             'scores': array([4.99697839e+08])},
    111: {   'boxes': array([[214,  15, 240,  37]]),
             'classes': array([1]),
             'scores': array([4.99860486e+08])},
    112: {   'boxes': array([[  1,   1, 222, 253]]),
             'classes': array([1]),
             'scores': array([4.99720729e+08])},
    113: {   'boxes': array([[147, 265, 187, 312]]),
             'classes': array([1]),
             'scores': array([4.99863738e+08])},
    114: {   'boxes': array([[237, 250, 375, 362]]),
             'classes': array([1]),
             'scores': array([4.99695799e+08])},
    115: {   'boxes': array([[236, 114, 375, 227]]),
             'classes': array([1]),
             'scores': array([4.99743532e+08])},
    116: {   'boxes': array([[149, 341, 216, 376]]),
             'classes': array([1]),
             'scores': array([4.99392259e+08])},
    117: {   'boxes': array([[191, 158, 221, 185]]),
             'classes': array([1]),
             'scores': array([4.99642893e+08])},
    118: {   'boxes': array([[177, 159, 230, 206]]),
             'classes': array([1]),
             'scores': array([4.9962254e+08])},
    119: {   'boxes': array([[178, 252, 228, 311]]),
             'classes': array([1]),
             'scores': array([4.99919131e+08])},
    120: {   'boxes': array([[193, 291, 241, 332]]),
             'classes': array([1]),
             'scores': array([4.99546688e+08])},
    121: {   'boxes': array([[209, 318, 330, 373]]),
             'classes': array([1]),
             'scores': array([4.99689983e+08])},
    122: {   'boxes': array([[206, 100, 342, 170]]),
             'classes': array([1]),
             'scores': array([4.99810074e+08])},
    123: {   'boxes': array([[283, 444, 375, 500]]),
             'classes': array([1]),
             'scores': array([4.99544528e+08])},
    124: {   'boxes': array([[219, 399, 371, 486]]),
             'classes': array([1]),
             'scores': array([5.00062007e+08])},
    125: {   'boxes': array([[161, 341, 298, 432]]),
             'classes': array([1]),
             'scores': array([4.99766329e+08])},
    126: {   'boxes': array([[135, 384, 203, 437]]),
             'classes': array([1]),
             'scores': array([4.99251925e+08])},
    127: {   'boxes': array([[131, 239, 211, 337]]),
             'classes': array([1]),
             'scores': array([5.00032668e+08])},
    128: {   'boxes': array([[144, 148, 255, 211]]),
             'classes': array([1]),
             'scores': array([4.99869605e+08])},
    129: {   'boxes': array([[160,  81, 327, 197]]),
             'classes': array([1]),
             'scores': array([4.99919229e+08])},
    130: {   'boxes': array([[188,   2, 375, 107]]),
             'classes': array([1]),
             'scores': array([4.99782878e+08])},
    131: {   'boxes': array([[ 32, 230,  93, 252]]),
             'classes': array([1]),
             'scores': array([5.00117527e+08])},
    132: {   'boxes': array([[139, 303, 171, 329]]),
             'classes': array([1]),
             'scores': array([4.99697209e+08])},
    133: {   'boxes': array([[429,  52, 469,  70]]),
             'classes': array([1]),
             'scores': array([4.99746528e+08])},
    134: {   'boxes': array([[421,  13, 450,  29]]),
             'classes': array([1]),
             'scores': array([4.99652136e+08])},
    135: {   'boxes': array([[429,  21, 461,  39]]),
             'classes': array([1]),
             'scores': array([4.99676281e+08])},
    136: {   'boxes': array([[142,  20, 375, 222]]),
             'classes': array([1]),
             'scores': array([4.99846018e+08])},
    137: {   'boxes': array([[ 68, 175, 375, 472]]),
             'classes': array([1]),
             'scores': array([4.99855043e+08])},
    138: {   'boxes': array([[  1, 171, 361, 499]]),
             'classes': array([1]),
             'scores': array([4.99812827e+08])},
    139: {   'boxes': array([[  1,   3, 264, 271]]),
             'classes': array([1]),
             'scores': array([4.99762772e+08])},
    140: {   'boxes': array([[ 56, 227, 213, 497]]),
             'classes': array([1]),
             'scores': array([4.99789147e+08])},
    141: {   'boxes': array([[ 59,  91, 261, 245]]),
             'classes': array([1]),
             'scores': array([4.99803285e+08])},
    142: {   'boxes': array([[134,  37, 332, 203]]),
             'classes': array([1]),
             'scores': array([4.99740208e+08])},
    143: {   'boxes': array([[ 75, 226, 274, 413]]),
             'classes': array([1]),
             'scores': array([4.99739331e+08])},
    144: {   'boxes': array([[114, 258, 332, 477]]),
             'classes': array([1]),
             'scores': array([4.99725714e+08])},
    145: {   'boxes': array([[ 50, 204, 301, 339]]),
             'classes': array([1]),
             'scores': array([4.99855083e+08])},
    146: {   'boxes': array([[163, 129, 319, 230]]),
             'classes': array([1]),
             'scores': array([4.99819428e+08])},
    147: {   'boxes': array([[373, 162, 500, 260]]),
             'classes': array([1]),
             'scores': array([4.99762159e+08])},
    148: {   'boxes': array([[373,  32, 500, 140]]),
             'classes': array([1]),
             'scores': array([4.99711577e+08])},
    149: {   'boxes': array([[ 89, 111, 375, 417]]),
             'classes': array([1]),
             'scores': array([4.99628693e+08])},
    150: {   'boxes': array([[104, 327, 300, 476]]),
             'classes': array([1]),
             'scores': array([4.99996877e+08])},
    151: {   'boxes': array([[ 57, 232, 374, 357]]),
             'classes': array([1]),
             'scores': array([4.99730417e+08])},
    152: {   'boxes': array([[ 32,   3, 374, 199]]),
             'classes': array([1]),
             'scores': array([5.00108749e+08])},
    153: {   'boxes': array([[139,  58, 374, 296]]),
             'classes': array([1]),
             'scores': array([4.99828657e+08])},
    154: {   'boxes': array([[ 77, 224, 375, 462]]),
             'classes': array([1]),
             'scores': array([4.99667404e+08])},
    155: {   'boxes': array([[ 97,  37, 375, 231]]),
             'classes': array([1]),
             'scores': array([4.99759044e+08])},
    156: {   'boxes': array([[174, 485, 280, 500]]),
             'classes': array([1]),
             'scores': array([4.99667669e+08])},
    157: {   'boxes': array([[163, 443, 283, 479]]),
             'classes': array([1]),
             'scores': array([4.99727684e+08])},
    158: {   'boxes': array([[153, 419, 240, 439]]),
             'classes': array([1]),
             'scores': array([4.9986581e+08])},
    159: {   'boxes': array([[156, 440, 230, 450]]),
             'classes': array([1]),
             'scores': array([4.99697102e+08])},
    160: {   'boxes': array([[163, 412, 248, 430]]),
             'classes': array([1]),
             'scores': array([4.99905324e+08])},
    161: {   'boxes': array([[194, 407, 251, 422]]),
             'classes': array([1]),
             'scores': array([4.99776762e+08])},
    162: {   'boxes': array([[157, 373, 266, 401]]),
             'classes': array([1]),
             'scores': array([5.00125439e+08])},
    163: {   'boxes': array([[169, 330, 252, 380]]),
             'classes': array([1]),
             'scores': array([4.9967415e+08])},
    164: {   'boxes': array([[153,   1, 305,  20]]),
             'classes': array([1]),
             'scores': array([4.99978247e+08])},
    165: {   'boxes': array([[110,   1, 244,  33]]),
             'classes': array([1]),
             'scores': array([4.99822088e+08])},
    166: {   'boxes': array([[ 32, 180, 359, 339]]),
             'classes': array([1]),
             'scores': array([4.99775626e+08])},
    167: {   'boxes': array([[244, 133, 367, 184]]),
             'classes': array([1]),
             'scores': array([4.99777066e+08])},
    168: {   'boxes': array([[ 79,  57, 474, 323]]),
             'classes': array([1]),
             'scores': array([5.00161003e+08])},
    169: {   'boxes': array([[100, 351, 190, 433]]),
             'classes': array([1]),
             'scores': array([4.99466072e+08])},
    170: {   'boxes': array([[124, 224, 269, 308]]),
             'classes': array([1]),
             'scores': array([4.99776364e+08])},
    171: {   'boxes': array([[153, 144, 321, 243]]),
             'classes': array([1]),
             'scores': array([4.99922449e+08])},
    172: {   'boxes': array([[114, 163, 167, 225]]),
             'classes': array([1]),
             'scores': array([4.99448721e+08])},
    173: {   'boxes': array([[107, 216, 154, 276]]),
             'classes': array([1]),
             'scores': array([4.99802235e+08])},
    174: {   'boxes': array([[ 78,   1, 500, 372]]),
             'classes': array([1]),
             'scores': array([4.99748167e+08])},
    175: {   'boxes': array([[167,  44, 458, 249]]),
             'classes': array([1]),
             'scores': array([4.99740592e+08])},
    176: {   'boxes': array([[  2,  13, 332, 469]]),
             'classes': array([1]),
             'scores': array([4.99804149e+08])},
    177: {   'boxes': array([[  3, 277, 375, 500]]),
             'classes': array([1]),
             'scores': array([4.99716303e+08])},
    178: {   'boxes': array([[  3,  12, 375, 305]]),
             'classes': array([1]),
             'scores': array([5.0008182e+08])},
    179: {   'boxes': array([[ 62, 116, 226, 174]]),
             'classes': array([1]),
             'scores': array([4.99595169e+08])},
    180: {   'boxes': array([[ 83, 170, 148, 209]]),
             'classes': array([1]),
             'scores': array([4.99794224e+08])},
    181: {   'boxes': array([[ 70, 362, 204, 414]]),
             'classes': array([1]),
             'scores': array([4.99218855e+08])},
    182: {   'boxes': array([[ 94, 281, 180, 327]]),
             'classes': array([1]),
             'scores': array([4.99721188e+08])},
    183: {   'boxes': array([[254, 149, 467, 358]]),
             'classes': array([1]),
             'scores': array([4.9965302e+08])},
    184: {   'boxes': array([[ 29,  29, 111,  68]]),
             'classes': array([1]),
             'scores': array([4.99374722e+08])},
    185: {   'boxes': array([[  4, 119,  53, 162]]),
             'classes': array([1]),
             'scores': array([4.9991659e+08])},
    186: {   'boxes': array([[ 28, 113, 131, 167]]),
             'classes': array([1]),
             'scores': array([4.99603369e+08])},
    187: {   'boxes': array([[ 17, 242, 103, 288]]),
             'classes': array([1]),
             'scores': array([4.99765915e+08])},
    188: {   'boxes': array([[ 92,  68, 255, 165]]),
             'classes': array([1]),
             'scores': array([4.99622679e+08])},
    189: {   'boxes': array([[ 30, 235, 362, 403]]),
             'classes': array([1]),
             'scores': array([4.99729867e+08])},
    190: {   'boxes': array([[ 75, 176, 132, 231]]),
             'classes': array([1]),
             'scores': array([4.99798614e+08])},
    191: {   'boxes': array([[ 88, 233, 157, 259]]),
             'classes': array([1]),
             'scores': array([4.99762247e+08])},
    192: {   'boxes': array([[ 91, 197, 116, 216]]),
             'classes': array([1]),
             'scores': array([4.99761448e+08])},
    193: {   'boxes': array([[ 68,  33, 247, 149]]),
             'classes': array([1]),
             'scores': array([5.00179817e+08])},
    194: {   'boxes': array([[ 68, 131, 249, 210]]),
             'classes': array([1]),
             'scores': array([4.99670443e+08])},
    195: {   'boxes': array([[ 59, 198, 284, 309]]),
             'classes': array([1]),
             'scores': array([5.00017292e+08])},
    196: {   'boxes': array([[ 83, 294, 375, 500]]),
             'classes': array([1]),
             'scores': array([4.99753392e+08])},
    197: {   'boxes': array([[ 13,  25, 481, 288]]),
             'classes': array([1]),
             'scores': array([4.99807332e+08])},
    198: {   'boxes': array([[ 66, 174, 265, 295]]),
             'classes': array([1]),
             'scores': array([5.00165932e+08])},
    199: {   'boxes': array([[ 19, 269, 296, 360]]),
             'classes': array([1]),
             'scores': array([4.99729097e+08])},
    200: {   'boxes': array([[  2, 125, 344, 316]]),
             'classes': array([1]),
             'scores': array([4.99844025e+08])},
    201: {   'boxes': array([[  4,  44, 167, 128]]),
             'classes': array([1]),
             'scores': array([4.99709877e+08])},
    202: {   'boxes': array([[ 31, 315, 112, 352]]),
             'classes': array([1]),
             'scores': array([4.99778247e+08])},
    203: {   'boxes': array([[ 35, 295, 115, 320]]),
             'classes': array([1]),
             'scores': array([4.99666992e+08])},
    204: {   'boxes': array([[ 40, 217,  84, 248]]),
             'classes': array([1]),
             'scores': array([4.99768547e+08])},
    205: {   'boxes': array([[ 43,  21, 127,  51]]),
             'classes': array([1]),
             'scores': array([4.99657804e+08])},
    206: {   'boxes': array([[ 36, 471, 126, 497]]),
             'classes': array([1]),
             'scores': array([4.99775174e+08])},
    207: {   'boxes': array([[ 36, 447, 117, 472]]),
             'classes': array([1]),
             'scores': array([4.99575612e+08])},
    208: {   'boxes': array([[ 41, 361, 103, 374]]),
             'classes': array([1]),
             'scores': array([4.99667116e+08])},
    209: {   'boxes': array([[265, 100, 298, 119]]),
             'classes': array([1]),
             'scores': array([4.99518394e+08])},
    210: {   'boxes': array([[ 85,  29, 375, 253]]),
             'classes': array([1]),
             'scores': array([4.99690412e+08])},
    211: {   'boxes': array([[ 23,   1, 307, 344]]),
             'classes': array([1]),
             'scores': array([4.9983173e+08])},
    212: {   'boxes': array([[ 25, 439,  71, 483]]),
             'classes': array([1]),
             'scores': array([4.99640682e+08])},
    213: {   'boxes': array([[ 32, 434,  70, 454]]),
             'classes': array([1]),
             'scores': array([4.99664343e+08])},
    214: {   'boxes': array([[279, 109, 319, 125]]),
             'classes': array([1]),
             'scores': array([5.00102457e+08])},
    215: {   'boxes': array([[291,  92, 323, 110]]),
             'classes': array([1]),
             'scores': array([4.99708371e+08])},
    216: {   'boxes': array([[281, 237, 309, 249]]),
             'classes': array([1]),
             'scores': array([4.99748914e+08])},
    217: {   'boxes': array([[290, 280, 308, 291]]),
             'classes': array([1]),
             'scores': array([4.99706605e+08])},
    218: {   'boxes': array([[ 33, 210, 191, 328]]),
             'classes': array([1]),
             'scores': array([4.99819503e+08])},
    219: {   'boxes': array([[190, 127, 277, 158]]),
             'classes': array([1]),
             'scores': array([4.99574725e+08])},
    220: {   'boxes': array([[194, 199, 275, 233]]),
             'classes': array([1]),
             'scores': array([4.99680146e+08])},
    221: {   'boxes': array([[196, 167, 236, 185]]),
             'classes': array([1]),
             'scores': array([4.99690278e+08])},
    222: {   'boxes': array([[193,  88, 294, 119]]),
             'classes': array([1]),
             'scores': array([4.99650707e+08])},
    223: {   'boxes': array([[195,  20, 297,  55]]),
             'classes': array([1]),
             'scores': array([4.99681445e+08])},
    224: {   'boxes': array([[198,   1, 295,  19]]),
             'classes': array([1]),
             'scores': array([4.99655766e+08])},
    225: {   'boxes': array([[185,  65, 293,  90]]),
             'classes': array([1]),
             'scores': array([4.99648013e+08])},
    226: {   'boxes': array([[195,  53, 230,  65]]),
             'classes': array([1]),
             'scores': array([4.99668664e+08])},
    227: {   'boxes': array([[196,  43, 214,  56]]),
             'classes': array([1]),
             'scores': array([4.99656596e+08])},
    228: {   'boxes': array([[195, 157, 223, 168]]),
             'classes': array([1]),
             'scores': array([4.99759706e+08])},
    229: {   'boxes': array([[ 16,   2, 221,  41]]),
             'classes': array([1]),
             'scores': array([4.99662703e+08])},
    230: {   'boxes': array([[  1,  61,  62, 109]]),
             'classes': array([1]),
             'scores': array([4.9980776e+08])},
    231: {   'boxes': array([[  2, 459, 174, 500]]),
             'classes': array([1]),
             'scores': array([4.99586734e+08])},
    232: {   'boxes': array([[106,  76, 404, 376]]),
             'classes': array([1]),
             'scores': array([4.99840907e+08])},
    233: {   'boxes': array([[ 56, 196, 357, 454]]),
             'classes': array([1]),
             'scores': array([4.99709703e+08])},
    234: {   'boxes': array([[ 75, 235, 275, 300]]),
             'classes': array([1]),
             'scores': array([4.99913078e+08])},
    235: {   'boxes': array([[ 56, 240, 332, 432]]),
             'classes': array([1]),
             'scores': array([4.99613562e+08])},
    236: {   'boxes': array([[ 18,  12, 318, 107]]),
             'classes': array([1]),
             'scores': array([4.99641758e+08])},
    237: {   'boxes': array([[167, 130, 476, 230]]),
             'classes': array([1]),
             'scores': array([4.99710711e+08])},
    238: {   'boxes': array([[ 86,  81, 472, 244]]),
             'classes': array([1]),
             'scores': array([4.9976776e+08])},
    239: {   'boxes': array([[ 57, 229, 167, 314]]),
             'classes': array([1]),
             'scores': array([5.00099738e+08])},
    240: {   'boxes': array([[ 85, 171, 152, 221]]),
             'classes': array([1]),
             'scores': array([4.99986733e+08])},
    241: {   'boxes': array([[ 71,  88, 112, 142]]),
             'classes': array([1]),
             'scores': array([4.99651102e+08])},
    242: {   'boxes': array([[124,  11, 375, 190]]),
             'classes': array([1]),
             'scores': array([5.00195857e+08])},
    243: {   'boxes': array([[ 92,  47, 286, 225]]),
             'classes': array([1]),
             'scores': array([4.99752614e+08])},
    244: {   'boxes': array([[ 72,   5, 182,  51]]),
             'classes': array([1]),
             'scores': array([4.99631992e+08])},
    245: {   'boxes': array([[ 65, 251, 188, 357]]),
             'classes': array([1]),
             'scores': array([4.99776086e+08])},
    246: {   'boxes': array([[ 81, 335, 217, 422]]),
             'classes': array([1]),
             'scores': array([4.99933272e+08])},
    247: {   'boxes': array([[107, 345, 338, 500]]),
             'classes': array([1]),
             'scores': array([5.00111863e+08])},
    248: {   'boxes': array([[226, 279, 375, 433]]),
             'classes': array([1]),
             'scores': array([5.00155889e+08])},
    249: {   'boxes': array([[ 47, 221, 161, 283]]),
             'classes': array([1]),
             'scores': array([4.99675698e+08])},
    250: {   'boxes': array([[143,   1, 500, 217]]),
             'classes': array([1]),
             'scores': array([4.99735723e+08])},
    251: {   'boxes': array([[191,   6, 500, 330]]),
             'classes': array([1]),
             'scores': array([4.99761023e+08])},
    252: {   'boxes': array([[132, 145, 230, 173]]),
             'classes': array([1]),
             'scores': array([4.9997321e+08])},
    253: {   'boxes': array([[110, 126, 373, 358]]),
             'classes': array([1]),
             'scores': array([4.99724425e+08])},
    254: {   'boxes': array([[130, 100, 237, 137]]),
             'classes': array([1]),
             'scores': array([4.99593048e+08])},
    255: {   'boxes': array([[131, 151, 246, 181]]),
             'classes': array([1]),
             'scores': array([4.99786966e+08])},
    256: {   'boxes': array([[133, 179, 207, 193]]),
             'classes': array([1]),
             'scores': array([5.00163997e+08])},
    257: {   'boxes': array([[135, 193, 287, 231]]),
             'classes': array([1]),
             'scores': array([5.0001365e+08])},
    258: {   'boxes': array([[138, 229, 258, 254]]),
             'classes': array([1]),
             'scores': array([5.00143827e+08])},
    259: {   'boxes': array([[121, 255, 278, 298]]),
             'classes': array([1]),
             'scores': array([4.99756699e+08])},
    260: {   'boxes': array([[132, 279, 289, 323]]),
             'classes': array([1]),
             'scores': array([4.9973584e+08])},
    261: {   'boxes': array([[130, 301, 316, 339]]),
             'classes': array([1]),
             'scores': array([4.99765156e+08])},
    262: {   'boxes': array([[117, 318, 333, 362]]),
             'classes': array([1]),
             'scores': array([4.99867615e+08])},
    263: {   'boxes': array([[131, 332, 332, 400]]),
             'classes': array([1]),
             'scores': array([4.99713068e+08])},
    264: {   'boxes': array([[125, 380, 333, 449]]),
             'classes': array([1]),
             'scores': array([4.99823432e+08])},
    265: {   'boxes': array([[125, 427, 333, 500]]),
             'classes': array([1]),
             'scores': array([4.99627629e+08])},
    266: {   'boxes': array([[162, 293, 375, 419]]),
             'classes': array([1]),
             'scores': array([5.00068298e+08])},
    267: {   'boxes': array([[165, 114, 373, 228]]),
             'classes': array([1]),
             'scores': array([5.00098402e+08])},
    268: {   'boxes': array([[172,   5, 373, 116]]),
             'classes': array([1]),
             'scores': array([4.99893328e+08])},
    269: {   'boxes': array([[103,  33, 137,  58]]),
             'classes': array([1]),
             'scores': array([4.99992593e+08])},
    270: {   'boxes': array([[ 44, 167, 289, 255]]),
             'classes': array([1]),
             'scores': array([4.99921179e+08])},
    271: {   'boxes': array([[256, 114, 496, 201]]),
             'classes': array([1]),
             'scores': array([4.99657581e+08])},
    272: {   'boxes': array([[ 68, 230, 291, 374]]),
             'classes': array([1]),
             'scores': array([5.00329094e+08])},
    273: {   'boxes': array([[ 51,  88, 375, 218]]),
             'classes': array([1]),
             'scores': array([4.99938656e+08])},
    274: {   'boxes': array([[ 72, 288, 135, 321]]),
             'classes': array([1]),
             'scores': array([4.99769322e+08])},
    275: {   'boxes': array([[ 71, 231, 135, 253]]),
             'classes': array([1]),
             'scores': array([5.00044728e+08])},
    276: {   'boxes': array([[ 85, 200, 119, 221]]),
             'classes': array([1]),
             'scores': array([4.99746565e+08])},
    277: {   'boxes': array([[ 76, 187, 101, 200]]),
             'classes': array([1]),
             'scores': array([4.99650391e+08])},
    278: {   'boxes': array([[ 75,  97, 114, 123]]),
             'classes': array([1]),
             'scores': array([4.99645661e+08])},
    279: {   'boxes': array([[ 74,  76, 107,  98]]),
             'classes': array([1]),
             'scores': array([4.9981432e+08])},
    280: {   'boxes': array([[ 85,   1, 279,  29]]),
             'classes': array([1]),
             'scores': array([4.99710467e+08])},
    281: {   'boxes': array([[  9,   1, 375, 500]]),
             'classes': array([1]),
             'scores': array([4.99875709e+08])},
    282: {   'boxes': array([[ 33, 383,  93, 418]]),
             'classes': array([1]),
             'scores': array([5.00126847e+08])},
    283: {   'boxes': array([[191,   1, 252,  21]]),
             'classes': array([1]),
             'scores': array([4.99568315e+08])},
    284: {   'boxes': array([[ 79,  74, 451, 334]]),
             'classes': array([1]),
             'scores': array([4.99687377e+08])},
    285: {   'boxes': array([[103, 145, 256, 209]]),
             'classes': array([1]),
             'scores': array([4.99895944e+08])},
    286: {   'boxes': array([[187, 212, 417, 304]]),
             'classes': array([1]),
             'scores': array([4.99673196e+08])},
    287: {   'boxes': array([[227, 365, 274, 375]]),
             'classes': array([1]),
             'scores': array([4.99695652e+08])},
    288: {   'boxes': array([[228, 343, 255, 361]]),
             'classes': array([1]),
             'scores': array([4.99662965e+08])},
    289: {   'boxes': array([[231, 325, 246, 339]]),
             'classes': array([1]),
             'scores': array([4.99653761e+08])},
    290: {   'boxes': array([[224, 308, 243, 326]]),
             'classes': array([1]),
             'scores': array([4.99648007e+08])},
    291: {   'boxes': array([[ 78, 252, 238, 312]]),
             'classes': array([1]),
             'scores': array([4.99754418e+08])},
    292: {   'boxes': array([[116, 443, 217, 500]]),
             'classes': array([1]),
             'scores': array([4.99796536e+08])},
    293: {   'boxes': array([[113, 104, 219, 169]]),
             'classes': array([1]),
             'scores': array([5.0008178e+08])},
    294: {   'boxes': array([[122, 167, 275, 222]]),
             'classes': array([1]),
             'scores': array([4.99611066e+08])},
    295: {   'boxes': array([[ 88, 189, 316, 276]]),
             'classes': array([1]),
             'scores': array([4.99720969e+08])},
    296: {   'boxes': array([[ 73, 259, 320, 440]]),
             'classes': array([1]),
             'scores': array([4.99737962e+08])},
    297: {   'boxes': array([[ 85, 376, 304, 443]]),
             'classes': array([1]),
             'scores': array([4.99648051e+08])},
    298: {   'boxes': array([[113,   1, 408, 183]]),
             'classes': array([1]),
             'scores': array([4.99713177e+08])},
    299: {   'boxes': array([[111,  61, 476, 324]]),
             'classes': array([1]),
             'scores': array([4.99715611e+08])},
    300: {   'boxes': array([[ 61, 159, 168, 217]]),
             'classes': array([1]),
             'scores': array([4.99677351e+08])},
    301: {   'boxes': array([[ 97, 315, 155, 332]]),
             'classes': array([1]),
             'scores': array([4.99784373e+08])},
    302: {   'boxes': array([[ 42,  27, 375, 205]]),
             'classes': array([1]),
             'scores': array([4.99965832e+08])},
    303: {   'boxes': array([[ 31, 150, 373, 362]]),
             'classes': array([1]),
             'scores': array([5.00143209e+08])},
    304: {   'boxes': array([[ 57, 366, 311, 500]]),
             'classes': array([1]),
             'scores': array([4.99857971e+08])},
    305: {   'boxes': array([[ 55, 206, 239, 316]]),
             'classes': array([1]),
             'scores': array([4.99824542e+08])},
    306: {   'boxes': array([[ 67,   1, 247,  71]]),
             'classes': array([1]),
             'scores': array([4.99792721e+08])},
    307: {   'boxes': array([[ 92, 130, 207, 171]]),
             'classes': array([1]),
             'scores': array([4.99966024e+08])},
    308: {   'boxes': array([[205, 155, 222, 169]]),
             'classes': array([1]),
             'scores': array([4.99711663e+08])},
    309: {   'boxes': array([[146, 148, 500, 269]]),
             'classes': array([1]),
             'scores': array([4.99604986e+08])},
    310: {   'boxes': array([[173, 287, 250, 330]]),
             'classes': array([1]),
             'scores': array([4.99762784e+08])},
    311: {   'boxes': array([[172, 315, 249, 336]]),
             'classes': array([1]),
             'scores': array([4.99787876e+08])},
    312: {   'boxes': array([[174, 144, 244, 171]]),
             'classes': array([1]),
             'scores': array([4.99879905e+08])},
    313: {   'boxes': array([[174, 123, 200, 136]]),
             'classes': array([1]),
             'scores': array([4.99805102e+08])},
    314: {   'boxes': array([[174,  50, 206,  63]]),
             'classes': array([1]),
             'scores': array([4.99717617e+08])},
    315: {   'boxes': array([[ 77, 118, 304, 260]]),
             'classes': array([1]),
             'scores': array([4.99782736e+08])},
    316: {   'boxes': array([[ 63, 266, 181, 332]]),
             'classes': array([1]),
             'scores': array([5.00251927e+08])},
    317: {   'boxes': array([[179, 160, 374, 422]]),
             'classes': array([1]),
             'scores': array([4.99848184e+08])},
    318: {   'boxes': array([[174,   1, 374, 186]]),
             'classes': array([1]),
             'scores': array([4.99990212e+08])},
    319: {   'boxes': array([[ 90,  61, 255, 213]]),
             'classes': array([1]),
             'scores': array([4.99319953e+08])},
    320: {   'boxes': array([[ 79, 157, 217, 261]]),
             'classes': array([1]),
             'scores': array([4.99737451e+08])},
    321: {   'boxes': array([[ 92, 380, 217, 499]]),
             'classes': array([1]),
             'scores': array([4.99766315e+08])},
    322: {   'boxes': array([[123, 315, 259, 389]]),
             'classes': array([1]),
             'scores': array([4.99727047e+08])},
    323: {   'boxes': array([[126, 333, 271, 415]]),
             'classes': array([1]),
             'scores': array([4.99694311e+08])},
    324: {   'boxes': array([[ 29, 233,  81, 319]]),
             'classes': array([1]),
             'scores': array([4.99734075e+08])},
    325: {   'boxes': array([[ 56, 111, 112, 129]]),
             'classes': array([1]),
             'scores': array([4.99537636e+08])},
    326: {   'boxes': array([[ 54, 132,  94, 152]]),
             'classes': array([1]),
             'scores': array([4.99478056e+08])},
    327: {   'boxes': array([[ 59, 156,  75, 171]]),
             'classes': array([1]),
             'scores': array([4.99555201e+08])},
    328: {   'boxes': array([[  1,  73, 375, 315]]),
             'classes': array([1]),
             'scores': array([4.99646374e+08])},
    329: {   'boxes': array([[127,  54, 268,  97]]),
             'classes': array([1]),
             'scores': array([4.99809831e+08])},
    330: {   'boxes': array([[118,   1, 300,  33]]),
             'classes': array([1]),
             'scores': array([4.99713928e+08])},
    331: {   'boxes': array([[192,   2, 265,  32]]),
             'classes': array([1]),
             'scores': array([4.99671408e+08])},
    332: {   'boxes': array([[185,  20, 250,  68]]),
             'classes': array([1]),
             'scores': array([4.9986764e+08])},
    333: {   'boxes': array([[190,  48, 252,  76]]),
             'classes': array([1]),
             'scores': array([4.99826921e+08])},
    334: {   'boxes': array([[191,  75, 280, 100]]),
             'classes': array([1]),
             'scores': array([5.00373307e+08])},
    335: {   'boxes': array([[179, 108, 250, 143]]),
             'classes': array([1]),
             'scores': array([4.99816938e+08])},
    336: {   'boxes': array([[199, 473, 234, 489]]),
             'classes': array([1]),
             'scores': array([4.99793357e+08])},
    337: {   'boxes': array([[163, 310, 220, 356]]),
             'classes': array([1]),
             'scores': array([4.99703352e+08])},
    338: {   'boxes': array([[168, 459, 208, 497]]),
             'classes': array([1]),
             'scores': array([4.99777606e+08])},
    339: {   'boxes': array([[177, 386, 314, 465]]),
             'classes': array([1]),
             'scores': array([4.99773938e+08])},
    340: {   'boxes': array([[144, 158, 500, 375]]),
             'classes': array([1]),
             'scores': array([4.99648591e+08])},
    341: {   'boxes': array([[110,   1, 500, 195]]),
             'classes': array([1]),
             'scores': array([4.99678754e+08])},
    342: {   'boxes': array([[110,  31, 333, 184]]),
             'classes': array([1]),
             'scores': array([4.99693173e+08])},
    343: {   'boxes': array([[110,  87, 257, 181]]),
             'classes': array([1]),
             'scores': array([4.99912429e+08])},
    344: {   'boxes': array([[103, 137, 224, 194]]),
             'classes': array([1]),
             'scores': array([4.99987931e+08])},
    345: {   'boxes': array([[ 91, 174, 190, 238]]),
             'classes': array([1]),
             'scores': array([4.99704661e+08])},
    346: {   'boxes': array([[102, 206, 154, 245]]),
             'classes': array([1]),
             'scores': array([4.99593462e+08])},
    347: {   'boxes': array([[ 39, 131, 127, 173]]),
             'classes': array([1]),
             'scores': array([4.99659121e+08])},
    348: {   'boxes': array([[ 84, 256, 129, 311]]),
             'classes': array([1]),
             'scores': array([4.99715024e+08])},
    349: {   'boxes': array([[ 87, 335, 104, 369]]),
             'classes': array([1]),
             'scores': array([4.99696559e+08])},
    350: {   'boxes': array([[ 85, 377, 108, 408]]),
             'classes': array([1]),
             'scores': array([4.99671685e+08])},
    351: {   'boxes': array([[ 97, 455, 137, 489]]),
             'classes': array([1]),
             'scores': array([4.99570176e+08])},
    352: {   'boxes': array([[127, 301, 332, 465]]),
             'classes': array([1]),
             'scores': array([4.99782279e+08])},
    353: {   'boxes': array([[103, 335, 197, 445]]),
             'classes': array([1]),
             'scores': array([4.99946038e+08])},
    354: {   'boxes': array([[ 95, 328, 179, 382]]),
             'classes': array([1]),
             'scores': array([4.99751947e+08])},
    355: {   'boxes': array([[100, 329, 153, 357]]),
             'classes': array([1]),
             'scores': array([4.99720305e+08])},
    356: {   'boxes': array([[209, 395, 294, 486]]),
             'classes': array([1]),
             'scores': array([4.99668443e+08])},
    357: {   'boxes': array([[214, 271, 291, 373]]),
             'classes': array([1]),
             'scores': array([4.99953624e+08])},
    358: {   'boxes': array([[274, 270, 334, 372]]),
             'classes': array([1]),
             'scores': array([4.99446252e+08])},
    359: {   'boxes': array([[269, 164, 328, 253]]),
             'classes': array([1]),
             'scores': array([4.9970586e+08])},
    360: {   'boxes': array([[204, 171, 278, 263]]),
             'classes': array([1]),
             'scores': array([4.9938897e+08])},
    361: {   'boxes': array([[194,  76, 273, 164]]),
             'classes': array([1]),
             'scores': array([5.00258913e+08])},
    362: {   'boxes': array([[273,  76, 326, 162]]),
             'classes': array([1]),
             'scores': array([4.99818429e+08])},
    363: {   'boxes': array([[ 82,  67, 160, 169]]),
             'classes': array([1]),
             'scores': array([4.99771496e+08])},
    364: {   'boxes': array([[ 16,  64,  96, 162]]),
             'classes': array([1]),
             'scores': array([4.99851911e+08])},
    365: {   'boxes': array([[ 17, 174,  92, 274]]),
             'classes': array([1]),
             'scores': array([4.99952121e+08])},
    366: {   'boxes': array([[ 85, 172, 184, 283]]),
             'classes': array([1]),
             'scores': array([4.99884931e+08])},
    367: {   'boxes': array([[ 21, 279, 110, 376]]),
             'classes': array([1]),
             'scores': array([4.99791762e+08])},
    368: {   'boxes': array([[ 94, 283, 183, 418]]),
             'classes': array([1]),
             'scores': array([4.99923209e+08])},
    369: {   'boxes': array([[ 18, 388,  96, 483]]),
             'classes': array([1]),
             'scores': array([4.99861114e+08])},
    370: {   'boxes': array([[ 88, 389, 183, 500]]),
             'classes': array([1]),
             'scores': array([4.9980894e+08])},
    371: {   'boxes': array([[20,  1, 96, 57]]),
             'classes': array([1]),
             'scores': array([4.99727973e+08])},
    372: {   'boxes': array([[ 84,   3, 178, 115]]),
             'classes': array([1]),
             'scores': array([4.99688149e+08])},
    373: {   'boxes': array([[ 23, 141, 298, 313]]),
             'classes': array([1]),
             'scores': array([4.99697976e+08])},
    374: {   'boxes': array([[247, 320, 301, 348]]),
             'classes': array([1]),
             'scores': array([4.99892385e+08])},
    375: {   'boxes': array([[ 22,   2, 375, 339]]),
             'classes': array([1]),
             'scores': array([4.99697314e+08])},
    376: {   'boxes': array([[199, 394, 223, 404]]),
             'classes': array([1]),
             'scores': array([4.99789453e+08])},
    377: {   'boxes': array([[199, 424, 220, 436]]),
             'classes': array([1]),
             'scores': array([4.99710346e+08])},
    378: {   'boxes': array([[196, 434, 220, 444]]),
             'classes': array([1]),
             'scores': array([4.99651168e+08])},
    379: {   'boxes': array([[195, 443, 220, 452]]),
             'classes': array([1]),
             'scores': array([4.99634213e+08])},
    380: {   'boxes': array([[ 51, 411, 146, 463]]),
             'classes': array([1]),
             'scores': array([4.99629499e+08])},
    381: {   'boxes': array([[ 49, 368, 146, 424]]),
             'classes': array([1]),
             'scores': array([5.00053658e+08])},
    382: {   'boxes': array([[ 81, 323, 133, 369]]),
             'classes': array([1]),
             'scores': array([4.99667857e+08])},
    383: {   'boxes': array([[ 61, 114, 139, 171]]),
             'classes': array([1]),
             'scores': array([4.99743703e+08])},
    384: {   'boxes': array([[ 60,  86, 139, 129]]),
             'classes': array([1]),
             'scores': array([4.99410376e+08])},
    385: {   'boxes': array([[ 54,  66, 436, 333]]),
             'classes': array([1]),
             'scores': array([4.99658301e+08])},
    386: {   'boxes': array([[112,  52, 183,  91]]),
             'classes': array([1]),
             'scores': array([4.99578681e+08])},
    387: {   'boxes': array([[196,   1, 322, 118]]),
             'classes': array([1]),
             'scores': array([4.99521439e+08])},
    388: {   'boxes': array([[101,  77, 314, 241]]),
             'classes': array([1]),
             'scores': array([4.99659224e+08])},
    389: {   'boxes': array([[ 49, 217, 304, 294]]),
             'classes': array([1]),
             'scores': array([4.99916555e+08])},
    390: {   'boxes': array([[ 22, 285, 366, 361]]),
             'classes': array([1]),
             'scores': array([4.99771467e+08])},
    391: {   'boxes': array([[  3, 297, 369, 375]]),
             'classes': array([1]),
             'scores': array([4.99666951e+08])},
    392: {   'boxes': array([[14, 73, 82, 97]]),
             'classes': array([1]),
             'scores': array([4.99813758e+08])},
    393: {   'boxes': array([[ 10, 103,  75, 121]]),
             'classes': array([1]),
             'scores': array([4.99997014e+08])},
    394: {   'boxes': array([[  1, 117,  78, 134]]),
             'classes': array([1]),
             'scores': array([4.99929114e+08])},
    395: {   'boxes': array([[  3,  83,  80, 114]]),
             'classes': array([1]),
             'scores': array([4.99639187e+08])},
    396: {   'boxes': array([[ 1, 38, 83, 76]]),
             'classes': array([1]),
             'scores': array([4.995279e+08])},
    397: {   'boxes': array([[ 1,  1, 96, 46]]),
             'classes': array([1]),
             'scores': array([4.99660331e+08])},
    398: {   'boxes': array([[ 87, 226, 371, 336]]),
             'classes': array([1]),
             'scores': array([4.99662672e+08])},
    399: {   'boxes': array([[151, 282, 375, 379]]),
             'classes': array([1]),
             'scores': array([4.99660327e+08])},
    400: {   'boxes': array([[132, 200, 374, 284]]),
             'classes': array([1]),
             'scores': array([4.99669352e+08])},
    401: {   'boxes': array([[160, 376, 375, 460]]),
             'classes': array([1]),
             'scores': array([4.9981973e+08])},
    402: {   'boxes': array([[130, 369, 218, 412]]),
             'classes': array([1]),
             'scores': array([4.99697288e+08])},
    403: {   'boxes': array([[167,  83, 297, 191]]),
             'classes': array([1]),
             'scores': array([4.9965738e+08])},
    404: {   'boxes': array([[ 40, 220, 264, 286]]),
             'classes': array([1]),
             'scores': array([4.99699583e+08])},
    405: {   'boxes': array([[ 94, 471, 189, 500]]),
             'classes': array([1]),
             'scores': array([4.99669943e+08])},
    406: {   'boxes': array([[150, 420, 230, 466]]),
             'classes': array([1]),
             'scores': array([4.99623938e+08])},
    407: {   'boxes': array([[218,  73, 238,  83]]),
             'classes': array([1]),
             'scores': array([4.99788166e+08])},
    408: {   'boxes': array([[213,  59, 238,  71]]),
             'classes': array([1]),
             'scores': array([4.99864653e+08])},
    409: {   'boxes': array([[216,  49, 239,  59]]),
             'classes': array([1]),
             'scores': array([4.99839288e+08])},
    410: {   'boxes': array([[ 63, 120, 355, 334]]),
             'classes': array([1]),
             'scores': array([4.99595838e+08])},
    411: {   'boxes': array([[ 57, 119, 352, 394]]),
             'classes': array([1]),
             'scores': array([4.996271e+08])},
    412: {   'boxes': array([[ 39, 230, 341, 308]]),
             'classes': array([1]),
             'scores': array([4.99780004e+08])},
    413: {   'boxes': array([[131, 182, 344, 240]]),
             'classes': array([1]),
             'scores': array([4.99693928e+08])},
    414: {   'boxes': array([[199,  15, 372, 146]]),
             'classes': array([1]),
             'scores': array([4.99580139e+08])},
    415: {   'boxes': array([[128, 302, 373, 391]]),
             'classes': array([1]),
             'scores': array([4.99785533e+08])},
    416: {   'boxes': array([[128, 440, 375, 500]]),
             'classes': array([1]),
             'scores': array([4.99695828e+08])},
    417: {   'boxes': array([[128, 393, 369, 461]]),
             'classes': array([1]),
             'scores': array([4.99735289e+08])},
    418: {   'boxes': array([[146,  47, 373, 149]]),
             'classes': array([1]),
             'scores': array([4.99640218e+08])},
    419: {   'boxes': array([[151,   3, 258,  36]]),
             'classes': array([1]),
             'scores': array([4.99737586e+08])},
    420: {   'boxes': array([[ 45, 153, 333, 274]]),
             'classes': array([1]),
             'scores': array([4.99880678e+08])},
    421: {   'boxes': array([[ 75,  70, 333, 499]]),
             'classes': array([1]),
             'scores': array([4.99841478e+08])},
    422: {   'boxes': array([[ 75,  13, 373, 300]]),
             'classes': array([1]),
             'scores': array([4.99715588e+08])},
    423: {   'boxes': array([[  1,   1, 273, 165]]),
             'classes': array([1]),
             'scores': array([4.99742523e+08])},
    424: {   'boxes': array([[ 25,  48, 383, 273]]),
             'classes': array([1]),
             'scores': array([5.00011091e+08])},
    425: {   'boxes': array([[ 30, 404, 374, 500]]),
             'classes': array([1]),
             'scores': array([4.99868681e+08])},
    426: {   'boxes': array([[ 83, 386, 374, 479]]),
             'classes': array([1]),
             'scores': array([5.0010071e+08])},
    427: {   'boxes': array([[ 65, 272, 374, 415]]),
             'classes': array([1]),
             'scores': array([4.99679391e+08])},
    428: {   'boxes': array([[ 55, 239, 374, 335]]),
             'classes': array([1]),
             'scores': array([4.99771282e+08])},
    429: {   'boxes': array([[ 55, 165, 374, 252]]),
             'classes': array([1]),
             'scores': array([4.99672333e+08])},
    430: {   'boxes': array([[ 66,  97, 374, 193]]),
             'classes': array([1]),
             'scores': array([4.99768715e+08])},
    431: {   'boxes': array([[ 26,   3, 374, 108]]),
             'classes': array([1]),
             'scores': array([5.00112029e+08])},
    432: {   'boxes': array([[ 99, 212, 253, 296]]),
             'classes': array([1]),
             'scores': array([4.99623947e+08])},
    433: {   'boxes': array([[  2,  22, 500, 375]]),
             'classes': array([1]),
             'scores': array([4.99640387e+08])},
    434: {   'boxes': array([[118, 225, 319, 500]]),
             'classes': array([1]),
             'scores': array([4.99772453e+08])},
    435: {   'boxes': array([[119,   1, 332, 125]]),
             'classes': array([1]),
             'scores': array([4.99740924e+08])},
    436: {   'boxes': array([[ 99, 263, 227, 352]]),
             'classes': array([1]),
             'scores': array([4.99862574e+08])},
    437: {   'boxes': array([[105, 334, 254, 430]]),
             'classes': array([1]),
             'scores': array([4.99975119e+08])},
    438: {   'boxes': array([[ 99, 351, 373, 500]]),
             'classes': array([1]),
             'scores': array([4.99846296e+08])},
    439: {   'boxes': array([[101, 128, 214, 192]]),
             'classes': array([1]),
             'scores': array([4.99881171e+08])},
    440: {   'boxes': array([[108,  55, 373, 189]]),
             'classes': array([1]),
             'scores': array([4.996749e+08])},
    441: {   'boxes': array([[199, 367, 224, 386]]),
             'classes': array([1]),
             'scores': array([4.9969841e+08])},
    442: {   'boxes': array([[190, 340, 213, 358]]),
             'classes': array([1]),
             'scores': array([4.99789605e+08])},
    443: {   'boxes': array([[199, 327, 223, 343]]),
             'classes': array([1]),
             'scores': array([4.99746804e+08])},
    444: {   'boxes': array([[190, 291, 212, 305]]),
             'classes': array([1]),
             'scores': array([4.99730955e+08])},
    445: {   'boxes': array([[199, 274, 220, 291]]),
             'classes': array([1]),
             'scores': array([4.99748635e+08])},
    446: {   'boxes': array([[199, 250, 213, 261]]),
             'classes': array([1]),
             'scores': array([4.99670677e+08])},
    447: {   'boxes': array([[201, 238, 224, 254]]),
             'classes': array([1]),
             'scores': array([4.99732195e+08])},
    448: {   'boxes': array([[200, 210, 221, 225]]),
             'classes': array([1]),
             'scores': array([4.99734287e+08])},
    449: {   'boxes': array([[203, 183, 220, 197]]),
             'classes': array([1]),
             'scores': array([4.99740278e+08])},
    450: {   'boxes': array([[206, 159, 220, 170]]),
             'classes': array([1]),
             'scores': array([4.99641404e+08])},
    451: {   'boxes': array([[202, 135, 218, 148]]),
             'classes': array([1]),
             'scores': array([4.99665493e+08])},
    452: {   'boxes': array([[206, 119, 219, 129]]),
             'classes': array([1]),
             'scores': array([4.99639186e+08])},
    453: {   'boxes': array([[205, 103, 219, 112]]),
             'classes': array([1]),
             'scores': array([4.99640032e+08])},
    454: {   'boxes': array([[202,  87, 216,  97]]),
             'classes': array([1]),
             'scores': array([4.9970403e+08])},
    455: {   'boxes': array([[133, 121, 228, 159]]),
             'classes': array([1]),
             'scores': array([4.99596521e+08])},
    456: {   'boxes': array([[144, 215, 227, 243]]),
             'classes': array([1]),
             'scores': array([4.99961944e+08])},
    457: {   'boxes': array([[126, 277, 168, 315]]),
             'classes': array([1]),
             'scores': array([4.99771226e+08])},
    458: {   'boxes': array([[121, 320, 223, 360]]),
             'classes': array([1]),
             'scores': array([4.99844896e+08])},
    459: {   'boxes': array([[131, 373, 192, 410]]),
             'classes': array([1]),
             'scores': array([4.99542798e+08])},
    460: {   'boxes': array([[ 65, 134, 366, 227]]),
             'classes': array([1]),
             'scores': array([4.99908154e+08])},
    461: {   'boxes': array([[  1, 224, 209, 361]]),
             'classes': array([1]),
             'scores': array([4.9966138e+08])},
    462: {   'boxes': array([[361,  95, 422, 139]]),
             'classes': array([1]),
             'scores': array([4.99734029e+08])},
    463: {   'boxes': array([[182, 191, 242, 256]]),
             'classes': array([1]),
             'scores': array([4.99521757e+08])},
    464: {   'boxes': array([[100, 289, 183, 316]]),
             'classes': array([1]),
             'scores': array([5.00141825e+08])},
    465: {   'boxes': array([[111, 241, 180, 270]]),
             'classes': array([1]),
             'scores': array([4.99752937e+08])},
    466: {   'boxes': array([[107, 218, 178, 236]]),
             'classes': array([1]),
             'scores': array([4.99981208e+08])},
    467: {   'boxes': array([[ 67, 402, 259, 467]]),
             'classes': array([1]),
             'scores': array([5.00139052e+08])},
    468: {   'boxes': array([[110,  65, 161,  84]]),
             'classes': array([1]),
             'scores': array([4.99673953e+08])},
    469: {   'boxes': array([[107,  96, 159, 114]]),
             'classes': array([1]),
             'scores': array([4.9972341e+08])},
    470: {   'boxes': array([[ 78, 100, 282, 190]]),
             'classes': array([1]),
             'scores': array([4.99923514e+08])},
    471: {   'boxes': array([[103, 273, 182, 295]]),
             'classes': array([1]),
             'scores': array([5.00142944e+08])},
    472: {   'boxes': array([[ 94,  68, 370, 424]]),
             'classes': array([1]),
             'scores': array([4.99688793e+08])},
    473: {   'boxes': array([[225, 230, 441, 500]]),
             'classes': array([1]),
             'scores': array([4.99725032e+08])},
    474: {   'boxes': array([[145,   1, 266,  58]]),
             'classes': array([1]),
             'scores': array([4.99532995e+08])},
    475: {   'boxes': array([[ 85, 308, 214, 375]]),
             'classes': array([1]),
             'scores': array([5.00087498e+08])},
    476: {   'boxes': array([[275, 119, 500, 333]]),
             'classes': array([1]),
             'scores': array([4.99696998e+08])},
    477: {   'boxes': array([[298,   1, 500, 106]]),
             'classes': array([1]),
             'scores': array([4.99701688e+08])},
    478: {   'boxes': array([[ 13, 151, 375, 498]]),
             'classes': array([1]),
             'scores': array([4.99699656e+08])},
    479: {   'boxes': array([[ 33, 149, 333, 360]]),
             'classes': array([1]),
             'scores': array([4.99735036e+08])},
    480: {   'boxes': array([[ 18, 369,  62, 431]]),
             'classes': array([1]),
             'scores': array([4.99672261e+08])},
    481: {   'boxes': array([[  1,  97,  25, 141]]),
             'classes': array([1]),
             'scores': array([4.99658397e+08])},
    482: {   'boxes': array([[ 63,  59, 500, 268]]),
             'classes': array([1]),
             'scores': array([4.9968437e+08])},
    483: {   'boxes': array([[305,   3, 500,  83]]),
             'classes': array([1]),
             'scores': array([4.99742153e+08])},
    484: {   'boxes': array([[122, 153, 474, 309]]),
             'classes': array([1]),
             'scores': array([4.99707763e+08])},
    485: {   'boxes': array([[178,   2, 288,  32]]),
             'classes': array([1]),
             'scores': array([4.9957055e+08])},
    486: {   'boxes': array([[174,  26, 291,  65]]),
             'classes': array([1]),
             'scores': array([4.9942751e+08])},
    487: {   'boxes': array([[185,  66, 272,  96]]),
             'classes': array([1]),
             'scores': array([4.99345553e+08])},
    488: {   'boxes': array([[ 93, 267, 289, 361]]),
             'classes': array([1]),
             'scores': array([4.995636e+08])},
    489: {   'boxes': array([[146,  19, 500, 191]]),
             'classes': array([1]),
             'scores': array([4.99683809e+08])},
    490: {   'boxes': array([[ 89, 157, 500, 375]]),
             'classes': array([1]),
             'scores': array([4.99641253e+08])},
    491: {   'boxes': array([[315, 468, 351, 492]]),
             'classes': array([1]),
             'scores': array([4.99806782e+08])},
    492: {   'boxes': array([[331, 459, 372, 500]]),
             'classes': array([1]),
             'scores': array([4.99541744e+08])},
    493: {   'boxes': array([[245,  98, 375, 390]]),
             'classes': array([1]),
             'scores': array([4.99321008e+08])},
    494: {   'boxes': array([[  2, 231, 276, 500]]),
             'classes': array([1]),
             'scores': array([4.99348309e+08])},
    495: {   'boxes': array([[ 54, 351, 304, 426]]),
             'classes': array([1]),
             'scores': array([5.001507e+08])},
    496: {   'boxes': array([[111,  61, 500, 296]]),
             'classes': array([1]),
             'scores': array([4.99665834e+08])},
    497: {   'boxes': array([[109,  33, 334, 260]]),
             'classes': array([1]),
             'scores': array([4.99758086e+08])},
    498: {   'boxes': array([[170, 309, 334, 461]]),
             'classes': array([1]),
             'scores': array([5.00031255e+08])},
    499: {   'boxes': array([[ 69, 174, 335, 302]]),
             'classes': array([1]),
             'scores': array([5.00182084e+08])},
    500: {   'boxes': array([[158, 309, 311, 387]]),
             'classes': array([1]),
             'scores': array([4.99791234e+08])},
    501: {   'boxes': array([[153, 262, 308, 315]]),
             'classes': array([1]),
             'scores': array([5.00089197e+08])},
    502: {   'boxes': array([[150, 215, 301, 269]]),
             'classes': array([1]),
             'scores': array([4.99857255e+08])},
    503: {   'boxes': array([[176,  33, 331, 154]]),
             'classes': array([1]),
             'scores': array([4.99948653e+08])},
    504: {   'boxes': array([[419, 252, 450, 262]]),
             'classes': array([1]),
             'scores': array([4.99660116e+08])},
    505: {   'boxes': array([[ 23, 269, 313, 500]]),
             'classes': array([1]),
             'scores': array([4.99605936e+08])},
    506: {   'boxes': array([[ 56,  13, 331, 327]]),
             'classes': array([1]),
             'scores': array([4.9967145e+08])},
    507: {   'boxes': array([[ 35, 224,  86, 281]]),
             'classes': array([1]),
             'scores': array([4.99829879e+08])},
    508: {   'boxes': array([[ 45, 298, 115, 358]]),
             'classes': array([1]),
             'scores': array([4.9973427e+08])},
    509: {   'boxes': array([[131, 305, 151, 318]]),
             'classes': array([1]),
             'scores': array([4.99718479e+08])},
    510: {   'boxes': array([[102, 190, 375, 337]]),
             'classes': array([1]),
             'scores': array([5.00168272e+08])},
    511: {   'boxes': array([[129,   3, 294, 124]]),
             'classes': array([1]),
             'scores': array([4.99733086e+08])},
    512: {   'boxes': array([[  2,   2, 332, 387]]),
             'classes': array([1]),
             'scores': array([4.99762787e+08])},
    513: {   'boxes': array([[ 52, 149, 307, 259]]),
             'classes': array([1]),
             'scores': array([5.00054086e+08])},
    514: {   'boxes': array([[132, 252, 375, 320]]),
             'classes': array([1]),
             'scores': array([4.9975109e+08])},
    515: {   'boxes': array([[ 12,  11, 456, 485]]),
             'classes': array([1]),
             'scores': array([4.9980635e+08])},
    516: {   'boxes': array([[191, 350, 375, 427]]),
             'classes': array([1]),
             'scores': array([4.996503e+08])},
    517: {   'boxes': array([[ 71, 126, 359, 274]]),
             'classes': array([1]),
             'scores': array([4.99678207e+08])},
    518: {   'boxes': array([[101,  25, 343, 157]]),
             'classes': array([1]),
             'scores': array([4.99685662e+08])},
    519: {   'boxes': array([[195, 297, 233, 351]]),
             'classes': array([1]),
             'scores': array([4.99313774e+08])},
    520: {   'boxes': array([[  2,   4, 281, 158]]),
             'classes': array([1]),
             'scores': array([4.99730073e+08])},
    521: {   'boxes': array([[  2, 318,  80, 385]]),
             'classes': array([1]),
             'scores': array([4.99969187e+08])},
    522: {   'boxes': array([[ 25,  41, 366, 407]]),
             'classes': array([1]),
             'scores': array([4.99866121e+08])},
    523: {   'boxes': array([[113,  17, 238, 165]]),
             'classes': array([1]),
             'scores': array([4.99754148e+08])},
    524: {   'boxes': array([[159, 160, 221, 191]]),
             'classes': array([1]),
             'scores': array([4.99781526e+08])},
    525: {   'boxes': array([[173, 387, 200, 411]]),
             'classes': array([1]),
             'scores': array([4.99692636e+08])},
    526: {   'boxes': array([[166, 397, 205, 425]]),
             'classes': array([1]),
             'scores': array([4.9966328e+08])},
    527: {   'boxes': array([[142, 411, 277, 456]]),
             'classes': array([1]),
             'scores': array([4.99680006e+08])},
    528: {   'boxes': array([[ 71, 206, 375, 416]]),
             'classes': array([1]),
             'scores': array([4.99633629e+08])},
    529: {   'boxes': array([[103,  94, 264, 284]]),
             'classes': array([1]),
             'scores': array([4.99638679e+08])},
    530: {   'boxes': array([[ 15,  18, 500, 333]]),
             'classes': array([1]),
             'scores': array([4.99632201e+08])},
    531: {   'boxes': array([[ 67, 281, 199, 330]]),
             'classes': array([1]),
             'scores': array([4.99378906e+08])},
    532: {   'boxes': array([[ 16,   1, 500, 340]]),
             'classes': array([1]),
             'scores': array([4.99862642e+08])},
    533: {   'boxes': array([[334, 211, 390, 241]]),
             'classes': array([1]),
             'scores': array([4.99855206e+08])},
    534: {   'boxes': array([[309, 199, 351, 231]]),
             'classes': array([1]),
             'scores': array([4.999855e+08])},
    535: {   'boxes': array([[315, 174, 360, 210]]),
             'classes': array([1]),
             'scores': array([4.9993634e+08])},
    536: {   'boxes': array([[354, 192, 396, 226]]),
             'classes': array([1]),
             'scores': array([4.9972718e+08])},
    537: {   'boxes': array([[354, 153, 398, 197]]),
             'classes': array([1]),
             'scores': array([5.00082484e+08])},
    538: {   'boxes': array([[355, 126, 401, 160]]),
             'classes': array([1]),
             'scores': array([5.00075829e+08])},
    539: {   'boxes': array([[319, 124, 373, 163]]),
             'classes': array([1]),
             'scores': array([4.99910802e+08])},
    540: {   'boxes': array([[320, 147, 355, 180]]),
             'classes': array([1]),
             'scores': array([4.99412025e+08])},
    541: {   'boxes': array([[ 59, 417,  75, 432]]),
             'classes': array([1]),
             'scores': array([4.99686622e+08])},
    542: {   'boxes': array([[135, 234, 316, 500]]),
             'classes': array([1]),
             'scores': array([4.99816089e+08])},
    543: {   'boxes': array([[146, 305, 267, 426]]),
             'classes': array([1]),
             'scores': array([4.99673253e+08])},
    544: {   'boxes': array([[147, 238, 245, 356]]),
             'classes': array([1]),
             'scores': array([4.99688599e+08])},
    545: {   'boxes': array([[149, 223, 223, 298]]),
             'classes': array([1]),
             'scores': array([4.99700071e+08])},
    546: {   'boxes': array([[140,  76, 266, 146]]),
             'classes': array([1]),
             'scores': array([4.99804369e+08])},
    547: {   'boxes': array([[ 69,   1, 372, 178]]),
             'classes': array([1]),
             'scores': array([4.99931066e+08])},
    548: {   'boxes': array([[107, 330, 159, 375]]),
             'classes': array([1]),
             'scores': array([4.99789148e+08])},
    549: {   'boxes': array([[114,  50, 185, 101]]),
             'classes': array([1]),
             'scores': array([4.99804569e+08])},
    550: {   'boxes': array([[110, 288, 157, 321]]),
             'classes': array([1]),
             'scores': array([4.99564173e+08])},
    551: {   'boxes': array([[ 89, 198, 353, 315]]),
             'classes': array([1]),
             'scores': array([4.99729925e+08])},
    552: {   'boxes': array([[ 36, 145, 412, 328]]),
             'classes': array([1]),
             'scores': array([4.99632793e+08])},
    553: {   'boxes': array([[ 48,   1, 393, 159]]),
             'classes': array([1]),
             'scores': array([4.99671799e+08])},
    554: {   'boxes': array([[149, 308, 189, 337]]),
             'classes': array([1]),
             'scores': array([4.99609944e+08])},
    555: {   'boxes': array([[233, 419, 372, 468]]),
             'classes': array([1]),
             'scores': array([4.99710682e+08])},
    556: {   'boxes': array([[222, 148, 375, 205]]),
             'classes': array([1]),
             'scores': array([4.9973202e+08])},
    557: {   'boxes': array([[241, 161, 375, 231]]),
             'classes': array([1]),
             'scores': array([4.99427034e+08])},
    558: {   'boxes': array([[223,  12, 330,  39]]),
             'classes': array([1]),
             'scores': array([4.99640832e+08])},
    559: {   'boxes': array([[185, 324, 241, 344]]),
             'classes': array([1]),
             'scores': array([5.00236799e+08])},
    560: {   'boxes': array([[190, 344, 249, 371]]),
             'classes': array([1]),
             'scores': array([4.99922533e+08])},
    561: {   'boxes': array([[188, 364, 246, 385]]),
             'classes': array([1]),
             'scores': array([4.99765018e+08])},
    562: {   'boxes': array([[193, 392, 230, 410]]),
             'classes': array([1]),
             'scores': array([4.99956246e+08])},
    563: {   'boxes': array([[185, 445, 208, 468]]),
             'classes': array([1]),
             'scores': array([4.99680057e+08])},
    564: {   'boxes': array([[156, 387, 375, 500]]),
             'classes': array([1]),
             'scores': array([4.99655498e+08])},
    565: {   'boxes': array([[194,  72, 240,  89]]),
             'classes': array([1]),
             'scores': array([4.99731177e+08])},
    566: {   'boxes': array([[202,   1, 327,  34]]),
             'classes': array([1]),
             'scores': array([4.99816603e+08])},
    567: {   'boxes': array([[211,  37, 251,  61]]),
             'classes': array([1]),
             'scores': array([4.99714776e+08])},
    568: {   'boxes': array([[196,  19, 272,  43]]),
             'classes': array([1]),
             'scores': array([4.99804089e+08])},
    569: {   'boxes': array([[313, 410, 377, 500]]),
             'classes': array([1]),
             'scores': array([4.99662653e+08])},
    570: {   'boxes': array([[121, 267, 140, 291]]),
             'classes': array([1]),
             'scores': array([4.99718505e+08])},
    571: {   'boxes': array([[120, 328, 207, 367]]),
             'classes': array([1]),
             'scores': array([4.99653381e+08])},
    572: {   'boxes': array([[ 19, 199, 401, 390]]),
             'classes': array([1]),
             'scores': array([5.0012553e+08])},
    573: {   'boxes': array([[ 15, 217, 157, 270]]),
             'classes': array([1]),
             'scores': array([4.99699215e+08])},
    574: {   'boxes': array([[ 18, 393, 162, 432]]),
             'classes': array([1]),
             'scores': array([4.99706306e+08])},
    575: {   'boxes': array([[ 87,  88, 322, 235]]),
             'classes': array([1]),
             'scores': array([4.99682199e+08])},
    576: {   'boxes': array([[  6,  19, 175,  93]]),
             'classes': array([1]),
             'scores': array([4.99663241e+08])},
    577: {   'boxes': array([[ 16,   2, 500, 280]]),
             'classes': array([1]),
             'scores': array([4.99615931e+08])},
    578: {   'boxes': array([[102, 315, 356, 427]]),
             'classes': array([1]),
             'scores': array([4.99688057e+08])},
    579: {   'boxes': array([[  1, 169, 375, 500]]),
             'classes': array([1]),
             'scores': array([4.9969292e+08])},
    580: {   'boxes': array([[  3, 239, 375, 500]]),
             'classes': array([1]),
             'scores': array([5.00254676e+08])},
    581: {   'boxes': array([[ 91,  86, 375, 359]]),
             'classes': array([1]),
             'scores': array([4.99609139e+08])},
    582: {   'boxes': array([[205,  85, 408, 199]]),
             'classes': array([1]),
             'scores': array([4.99816251e+08])},
    583: {   'boxes': array([[ 30,   1, 375, 386]]),
             'classes': array([1]),
             'scores': array([4.9986228e+08])},
    584: {   'boxes': array([[231,  52, 500, 301]]),
             'classes': array([1]),
             'scores': array([4.99680308e+08])},
    585: {   'boxes': array([[ 29,   7, 493, 406]]),
             'classes': array([1]),
             'scores': array([4.99704943e+08])},
    586: {   'boxes': array([[164, 359, 229, 406]]),
             'classes': array([1]),
             'scores': array([4.99704805e+08])},
    587: {   'boxes': array([[174, 306, 232, 357]]),
             'classes': array([1]),
             'scores': array([4.99892478e+08])},
    588: {   'boxes': array([[173, 285, 221, 321]]),
             'classes': array([1]),
             'scores': array([4.99759774e+08])},
    589: {   'boxes': array([[186,  63, 374, 124]]),
             'classes': array([1]),
             'scores': array([4.99628886e+08])},
    590: {   'boxes': array([[178,   1, 373,  35]]),
             'classes': array([1]),
             'scores': array([4.9964734e+08])},
    591: {   'boxes': array([[ 87, 143, 191, 228]]),
             'classes': array([1]),
             'scores': array([4.99398736e+08])},
    592: {   'boxes': array([[  5, 227, 375, 500]]),
             'classes': array([1]),
             'scores': array([4.99593269e+08])},
    593: {   'boxes': array([[188, 110, 375, 361]]),
             'classes': array([1]),
             'scores': array([4.99814525e+08])},
    594: {   'boxes': array([[174, 149, 322, 220]]),
             'classes': array([1]),
             'scores': array([4.9981622e+08])},
    595: {   'boxes': array([[163, 277, 347, 334]]),
             'classes': array([1]),
             'scores': array([4.99742803e+08])},
    596: {   'boxes': array([[163, 362, 336, 409]]),
             'classes': array([1]),
             'scores': array([4.99792796e+08])},
    597: {   'boxes': array([[126, 235, 241, 297]]),
             'classes': array([1]),
             'scores': array([4.99416017e+08])},
    598: {   'boxes': array([[ 94, 107, 146, 154]]),
             'classes': array([1]),
             'scores': array([4.99678544e+08])},
    599: {   'boxes': array([[ 91, 174, 141, 257]]),
             'classes': array([1]),
             'scores': array([4.99585151e+08])},
    600: {   'boxes': array([[172, 157, 232, 187]]),
             'classes': array([1]),
             'scores': array([4.99865335e+08])},
    601: {   'boxes': array([[116, 320, 375, 489]]),
             'classes': array([1]),
             'scores': array([4.99583117e+08])},
    602: {   'boxes': array([[  1,   1, 375, 321]]),
             'classes': array([1]),
             'scores': array([4.99704627e+08])},
    603: {   'boxes': array([[ 14, 140, 128, 215]]),
             'classes': array([1]),
             'scores': array([4.99711171e+08])},
    604: {   'boxes': array([[ 83, 390, 153, 460]]),
             'classes': array([1]),
             'scores': array([4.99699972e+08])},
    605: {   'boxes': array([[ 20, 101, 150, 165]]),
             'classes': array([1]),
             'scores': array([4.99768874e+08])},
    606: {   'boxes': array([[100,  72, 323, 172]]),
             'classes': array([1]),
             'scores': array([4.99899056e+08])},
    607: {   'boxes': array([[113, 198, 331, 287]]),
             'classes': array([1]),
             'scores': array([5.00128221e+08])},
    608: {   'boxes': array([[170, 160, 309, 316]]),
             'classes': array([1]),
             'scores': array([4.99756414e+08])},
    609: {   'boxes': array([[131, 188, 261, 285]]),
             'classes': array([1]),
             'scores': array([5.00099433e+08])},
    610: {   'boxes': array([[147, 256, 269, 375]]),
             'classes': array([1]),
             'scores': array([4.99993804e+08])},
    611: {   'boxes': array([[146, 372, 319, 475]]),
             'classes': array([1]),
             'scores': array([4.9976678e+08])},
    612: {   'boxes': array([[138, 121, 283, 217]]),
             'classes': array([1]),
             'scores': array([5.00133112e+08])},
    613: {   'boxes': array([[140,   2, 354, 144]]),
             'classes': array([1]),
             'scores': array([5.00121846e+08])},
    614: {   'boxes': array([[101,  99, 337, 225]]),
             'classes': array([1]),
             'scores': array([4.99723688e+08])},
    615: {   'boxes': array([[174, 390, 357, 500]]),
             'classes': array([1]),
             'scores': array([4.99705708e+08])},
    616: {   'boxes': array([[148, 254, 327, 426]]),
             'classes': array([1]),
             'scores': array([5.00005266e+08])},
    617: {   'boxes': array([[167,  84, 361, 247]]),
             'classes': array([1]),
             'scores': array([4.99808968e+08])},
    618: {   'boxes': array([[307,   1, 375,  84]]),
             'classes': array([1]),
             'scores': array([4.99735836e+08])},
    619: {   'boxes': array([[  1, 449, 334, 500]]),
             'classes': array([1]),
             'scores': array([4.99668931e+08])},
    620: {   'boxes': array([[ 68,   1, 334,  93]]),
             'classes': array([1]),
             'scores': array([4.99725587e+08])},
    621: {   'boxes': array([[ 88,  33, 334, 189]]),
             'classes': array([1]),
             'scores': array([4.99725975e+08])},
    622: {   'boxes': array([[191, 174, 334, 268]]),
             'classes': array([1]),
             'scores': array([4.99822056e+08])},
    623: {   'boxes': array([[119, 260, 333, 430]]),
             'classes': array([1]),
             'scores': array([4.9983846e+08])},
    624: {   'boxes': array([[ 39, 343, 334, 500]]),
             'classes': array([1]),
             'scores': array([4.99956746e+08])},
    625: {   'boxes': array([[ 61, 143, 211, 225]]),
             'classes': array([1]),
             'scores': array([4.99863065e+08])},
    626: {   'boxes': array([[ 28,   2, 332, 217]]),
             'classes': array([1]),
             'scores': array([4.99756188e+08])},
    627: {   'boxes': array([[ 79, 306, 311, 401]]),
             'classes': array([1]),
             'scores': array([4.99712576e+08])},
    628: {   'boxes': array([[ 82, 370, 330, 496]]),
             'classes': array([1]),
             'scores': array([4.99745719e+08])},
    629: {   'boxes': array([[ 14, 274, 333, 500]]),
             'classes': array([1]),
             'scores': array([4.99776399e+08])},
    630: {   'boxes': array([[ 72,   1, 333, 302]]),
             'classes': array([1]),
             'scores': array([5.00329371e+08])},
    631: {   'boxes': array([[ 47, 387, 160, 487]]),
             'classes': array([1]),
             'scores': array([4.99804957e+08])},
    632: {   'boxes': array([[ 80, 395, 339, 478]]),
             'classes': array([1]),
             'scores': array([4.99704026e+08])},
    633: {   'boxes': array([[ 78, 251, 326, 406]]),
             'classes': array([1]),
             'scores': array([4.99739508e+08])},
    634: {   'boxes': array([[149, 164, 220, 199]]),
             'classes': array([1]),
             'scores': array([4.99713942e+08])},
    635: {   'boxes': array([[152, 302, 312, 342]]),
             'classes': array([1]),
             'scores': array([4.99586018e+08])},
    636: {   'boxes': array([[ 38, 217, 179, 500]]),
             'classes': array([1]),
             'scores': array([4.99650766e+08])},
    637: {   'boxes': array([[ 51,   1, 168, 186]]),
             'classes': array([1]),
             'scores': array([5.00021485e+08])},
    638: {   'boxes': array([[117, 174, 333, 272]]),
             'classes': array([1]),
             'scores': array([4.99785753e+08])},
    639: {   'boxes': array([[ 74, 242, 333, 347]]),
             'classes': array([1]),
             'scores': array([4.99648708e+08])},
    640: {   'boxes': array([[ 64, 169, 402, 260]]),
             'classes': array([1]),
             'scores': array([4.99905467e+08])},
    641: {   'boxes': array([[117, 114, 416, 304]]),
             'classes': array([1]),
             'scores': array([4.99650159e+08])},
    642: {   'boxes': array([[  1, 126, 322, 241]]),
             'classes': array([1]),
             'scores': array([4.99720812e+08])},
    643: {   'boxes': array([[433, 230, 500, 375]]),
             'classes': array([1]),
             'scores': array([4.99685634e+08])},
    644: {   'boxes': array([[  2,  19, 333, 497]]),
             'classes': array([1]),
             'scores': array([4.9961753e+08])},
    645: {   'boxes': array([[  3, 263, 329, 428]]),
             'classes': array([1]),
             'scores': array([4.99596244e+08])},
    646: {   'boxes': array([[  1, 221, 196, 329]]),
             'classes': array([1]),
             'scores': array([4.99617368e+08])},
    647: {   'boxes': array([[113, 200, 336, 272]]),
             'classes': array([1]),
             'scores': array([4.9971293e+08])},
    648: {   'boxes': array([[146,  75, 336, 146]]),
             'classes': array([1]),
             'scores': array([4.99711505e+08])},
    649: {   'boxes': array([[165, 308, 220, 359]]),
             'classes': array([1]),
             'scores': array([4.99502113e+08])},
    650: {   'boxes': array([[121, 330, 318, 490]]),
             'classes': array([1]),
             'scores': array([4.99810828e+08])},
    651: {   'boxes': array([[125,   6, 326, 131]]),
             'classes': array([1]),
             'scores': array([4.99681628e+08])},
    652: {   'boxes': array([[117, 104, 223, 180]]),
             'classes': array([1]),
             'scores': array([4.99672296e+08])},
    653: {   'boxes': array([[104, 231, 195, 322]]),
             'classes': array([1]),
             'scores': array([4.99906911e+08])},
    654: {   'boxes': array([[  1,   1, 375, 403]]),
             'classes': array([1]),
             'scores': array([4.99662946e+08])},
    655: {   'boxes': array([[ 73, 279, 375, 500]]),
             'classes': array([1]),
             'scores': array([5.00011415e+08])},
    656: {   'boxes': array([[114, 102, 375, 341]]),
             'classes': array([1]),
             'scores': array([4.99780967e+08])},
    657: {   'boxes': array([[140, 318, 375, 500]]),
             'classes': array([1]),
             'scores': array([4.99740937e+08])},
    658: {   'boxes': array([[ 59,   4, 243,  66]]),
             'classes': array([1]),
             'scores': array([4.99794909e+08])},
    659: {   'boxes': array([[ 85, 144, 251, 229]]),
             'classes': array([1]),
             'scores': array([4.9965768e+08])},
    660: {   'boxes': array([[ 66, 290, 331, 479]]),
             'classes': array([1]),
             'scores': array([4.99709174e+08])},
    661: {   'boxes': array([[153, 119, 329, 367]]),
             'classes': array([1]),
             'scores': array([4.99631828e+08])},
    662: {   'boxes': array([[169, 157, 230, 256]]),
             'classes': array([1]),
             'scores': array([4.99741929e+08])},
    663: {   'boxes': array([[ 78,  78, 375, 289]]),
             'classes': array([1]),
             'scores': array([4.99621356e+08])},
    664: {   'boxes': array([[196, 154, 270, 192]]),
             'classes': array([1]),
             'scores': array([4.99622963e+08])},
    665: {   'boxes': array([[ 53, 234, 217, 315]]),
             'classes': array([1]),
             'scores': array([4.99627882e+08])},
    666: {   'boxes': array([[  1, 274, 288, 429]]),
             'classes': array([1]),
             'scores': array([4.99619923e+08])},
    667: {   'boxes': array([[166, 107, 329, 299]]),
             'classes': array([1]),
             'scores': array([4.99731892e+08])},
    668: {   'boxes': array([[ 25, 207, 159, 355]]),
             'classes': array([1]),
             'scores': array([4.99626407e+08])},
    669: {   'boxes': array([[ 67, 277, 226, 357]]),
             'classes': array([1]),
             'scores': array([4.99738732e+08])},
    670: {   'boxes': array([[ 78, 375, 370, 500]]),
             'classes': array([1]),
             'scores': array([4.99518684e+08])},
    671: {   'boxes': array([[249, 167, 284, 197]]),
             'classes': array([1]),
             'scores': array([4.99702145e+08])},
    672: {   'boxes': array([[270, 129, 283, 143]]),
             'classes': array([1]),
             'scores': array([4.99696607e+08])},
    673: {   'boxes': array([[ 78,  95, 223, 189]]),
             'classes': array([1]),
             'scores': array([5.00012287e+08])},
    674: {   'boxes': array([[ 82, 206, 248, 298]]),
             'classes': array([1]),
             'scores': array([5.00139519e+08])},
    675: {   'boxes': array([[ 91, 326, 247, 433]]),
             'classes': array([1]),
             'scores': array([4.9964744e+08])},
    676: {   'boxes': array([[ 87, 274, 238, 349]]),
             'classes': array([1]),
             'scores': array([4.99750124e+08])},
    677: {   'boxes': array([[  2, 104,  79, 135]]),
             'classes': array([1]),
             'scores': array([4.99873719e+08])},
    678: {   'boxes': array([[  1, 370,  64, 397]]),
             'classes': array([1]),
             'scores': array([4.99695856e+08])},
    679: {   'boxes': array([[ 75,  15, 375, 429]]),
             'classes': array([1]),
             'scores': array([4.99718077e+08])},
    680: {   'boxes': array([[157, 348, 375, 500]]),
             'classes': array([1]),
             'scores': array([4.9973375e+08])},
    681: {   'boxes': array([[ 14,  70, 443, 182]]),
             'classes': array([1]),
             'scores': array([4.99710012e+08])},
    682: {   'boxes': array([[167,  33, 500, 190]]),
             'classes': array([1]),
             'scores': array([4.99713142e+08])},
    683: {   'boxes': array([[ 28, 223, 253, 309]]),
             'classes': array([1]),
             'scores': array([4.99783669e+08])},
    684: {   'boxes': array([[  1,   1, 374, 135]]),
             'classes': array([1]),
             'scores': array([4.9981026e+08])},
    685: {   'boxes': array([[159,  22, 375, 427]]),
             'classes': array([1]),
             'scores': array([4.99957702e+08])},
    686: {   'boxes': array([[252, 344, 375, 500]]),
             'classes': array([1]),
             'scores': array([4.9948841e+08])},
    687: {   'boxes': array([[ 19, 142, 500, 375]]),
             'classes': array([1]),
             'scores': array([4.99783086e+08])},
    688: {   'boxes': array([[218, 349, 298, 375]]),
             'classes': array([1]),
             'scores': array([4.99817901e+08])},
    689: {   'boxes': array([[298,   1, 500,  57]]),
             'classes': array([1]),
             'scores': array([4.99994854e+08])},
    690: {   'boxes': array([[270,   1, 351,  17]]),
             'classes': array([1]),
             'scores': array([4.99829479e+08])},
    691: {   'boxes': array([[ 42, 116,  77, 138]]),
             'classes': array([1]),
             'scores': array([4.99809825e+08])},
    692: {   'boxes': array([[ 45, 103,  71, 117]]),
             'classes': array([1]),
             'scores': array([4.99702077e+08])},
    693: {   'boxes': array([[ 35,  89,  61, 101]]),
             'classes': array([1]),
             'scores': array([4.99724239e+08])},
    694: {   'boxes': array([[ 59,   1, 292,  68]]),
             'classes': array([1]),
             'scores': array([4.99808996e+08])},
    695: {   'boxes': array([[ 29, 236, 269, 277]]),
             'classes': array([1]),
             'scores': array([4.99671015e+08])},
    696: {   'boxes': array([[ 43, 273,  78, 293]]),
             'classes': array([1]),
             'scores': array([4.99611687e+08])},
    697: {   'boxes': array([[ 32, 256, 253, 334]]),
             'classes': array([1]),
             'scores': array([4.99610421e+08])},
    698: {   'boxes': array([[ 61, 281, 305, 346]]),
             'classes': array([1]),
             'scores': array([4.9955975e+08])},
    699: {   'boxes': array([[ 45, 331, 304, 372]]),
             'classes': array([1]),
             'scores': array([4.99748553e+08])},
    700: {   'boxes': array([[ 21, 353, 333, 429]]),
             'classes': array([1]),
             'scores': array([4.9959054e+08])},
    701: {   'boxes': array([[ 20, 397, 345, 497]]),
             'classes': array([1]),
             'scores': array([5.0012574e+08])},
    702: {   'boxes': array([[230, 135, 278, 178]]),
             'classes': array([1]),
             'scores': array([4.99675129e+08])},
    703: {   'boxes': array([[238,  75, 281, 121]]),
             'classes': array([1]),
             'scores': array([4.99730749e+08])},
    704: {   'boxes': array([[ 41, 263, 229, 331]]),
             'classes': array([1]),
             'scores': array([4.99683335e+08])},
    705: {   'boxes': array([[152, 220, 310, 282]]),
             'classes': array([1]),
             'scores': array([4.99815502e+08])},
    706: {   'boxes': array([[155, 190, 307, 242]]),
             'classes': array([1]),
             'scores': array([4.99426884e+08])},
    707: {   'boxes': array([[ 44,   8, 469, 313]]),
             'classes': array([1]),
             'scores': array([5.00072387e+08])},
    708: {   'boxes': array([[ 22, 128, 332, 315]]),
             'classes': array([1]),
             'scores': array([4.99786775e+08])},
    709: {   'boxes': array([[  2, 281, 332, 500]]),
             'classes': array([1]),
             'scores': array([4.99975106e+08])},
    710: {   'boxes': array([[ 92, 108, 333, 389]]),
             'classes': array([1]),
             'scores': array([4.99789783e+08])},
    711: {   'boxes': array([[ 22, 103, 414, 469]]),
             'classes': array([1]),
             'scores': array([4.99771203e+08])},
    712: {   'boxes': array([[ 32,  39, 450, 216]]),
             'classes': array([1]),
             'scores': array([4.99771057e+08])},
    713: {   'boxes': array([[224, 121, 441, 264]]),
             'classes': array([1]),
             'scores': array([4.99745484e+08])},
    714: {   'boxes': array([[165,  24, 500, 373]]),
             'classes': array([1]),
             'scores': array([4.99680841e+08])},
    715: {   'boxes': array([[144, 247, 167, 271]]),
             'classes': array([1]),
             'scores': array([4.99762428e+08])},
    716: {   'boxes': array([[123, 135, 375, 235]]),
             'classes': array([1]),
             'scores': array([4.99813056e+08])},
    717: {   'boxes': array([[115, 238, 304, 376]]),
             'classes': array([1]),
             'scores': array([4.99670008e+08])},
    718: {   'boxes': array([[117, 276, 360, 426]]),
             'classes': array([1]),
             'scores': array([4.99745217e+08])},
    719: {   'boxes': array([[114, 301, 375, 500]]),
             'classes': array([1]),
             'scores': array([4.9977348e+08])},
    720: {   'boxes': array([[114,   1, 375, 126]]),
             'classes': array([1]),
             'scores': array([4.99807356e+08])},
    721: {   'boxes': array([[ 90, 419, 139, 448]]),
             'classes': array([1]),
             'scores': array([4.99771148e+08])},
    722: {   'boxes': array([[103, 178, 401, 500]]),
             'classes': array([1]),
             'scores': array([4.99757786e+08])},
    723: {   'boxes': array([[ 63,  61, 338, 308]]),
             'classes': array([1]),
             'scores': array([4.99685845e+08])},
    724: {   'boxes': array([[ 45, 242, 103, 266]]),
             'classes': array([1]),
             'scores': array([4.99631508e+08])},
    725: {   'boxes': array([[ 69, 186,  89, 215]]),
             'classes': array([1]),
             'scores': array([4.99650838e+08])},
    726: {   'boxes': array([[ 60, 313, 193, 432]]),
             'classes': array([1]),
             'scores': array([5.00040121e+08])},
    727: {   'boxes': array([[167, 193, 280, 324]]),
             'classes': array([1]),
             'scores': array([5.00038895e+08])},
    728: {   'boxes': array([[107, 107, 262, 208]]),
             'classes': array([1]),
             'scores': array([4.99751391e+08])},
    729: {   'boxes': array([[184, 143, 264, 175]]),
             'classes': array([1]),
             'scores': array([4.99232953e+08])},
    730: {   'boxes': array([[ 26,  92, 486, 253]]),
             'classes': array([1]),
             'scores': array([4.99775059e+08])},
    731: {   'boxes': array([[ 90, 263, 326, 381]]),
             'classes': array([1]),
             'scores': array([4.99887044e+08])},
    732: {   'boxes': array([[110,  67, 199, 104]]),
             'classes': array([1]),
             'scores': array([4.9971622e+08])},
    733: {   'boxes': array([[132,  19, 166,  43]]),
             'classes': array([1]),
             'scores': array([5.00042293e+08])},
    734: {   'boxes': array([[ 67, 463, 154, 495]]),
             'classes': array([1]),
             'scores': array([4.99868385e+08])},
    735: {   'boxes': array([[ 75, 431, 171, 468]]),
             'classes': array([1]),
             'scores': array([5.00023912e+08])},
    736: {   'boxes': array([[ 79, 333, 201, 387]]),
             'classes': array([1]),
             'scores': array([4.99775514e+08])},
    737: {   'boxes': array([[106, 187, 293, 291]]),
             'classes': array([1]),
             'scores': array([4.99736507e+08])},
    738: {   'boxes': array([[ 61,  27, 280, 163]]),
             'classes': array([1]),
             'scores': array([4.99775967e+08])},
    739: {   'boxes': array([[ 51,  42, 334, 237]]),
             'classes': array([1]),
             'scores': array([4.99747985e+08])},
    740: {   'boxes': array([[119, 222, 176, 270]]),
             'classes': array([1]),
             'scores': array([4.99963216e+08])},
    741: {   'boxes': array([[ 54,   1, 500, 226]]),
             'classes': array([1]),
             'scores': array([4.99916589e+08])},
    742: {   'boxes': array([[159, 140, 287, 172]]),
             'classes': array([1]),
             'scores': array([4.99561775e+08])},
    743: {   'boxes': array([[127, 189, 197, 215]]),
             'classes': array([1]),
             'scores': array([4.99658017e+08])},
    744: {   'boxes': array([[139, 155, 196, 202]]),
             'classes': array([1]),
             'scores': array([4.99847172e+08])},
    745: {   'boxes': array([[ 93,  77, 313, 237]]),
             'classes': array([1]),
             'scores': array([5.00091315e+08])},
    746: {   'boxes': array([[  4, 220, 229, 447]]),
             'classes': array([1]),
             'scores': array([4.99661061e+08])},
    747: {   'boxes': array([[  1,   7, 228, 262]]),
             'classes': array([1]),
             'scores': array([4.9956366e+08])},
    748: {   'boxes': array([[114, 111, 375, 264]]),
             'classes': array([1]),
             'scores': array([4.99745531e+08])},
    749: {   'boxes': array([[152, 245, 375, 356]]),
             'classes': array([1]),
             'scores': array([4.99738157e+08])},
    750: {   'boxes': array([[114, 307, 372, 462]]),
             'classes': array([1]),
             'scores': array([4.99682242e+08])},
    751: {   'boxes': array([[143, 168, 375, 335]]),
             'classes': array([1]),
             'scores': array([4.99739924e+08])},
    752: {   'boxes': array([[130, 196, 156, 217]]),
             'classes': array([1]),
             'scores': array([4.99829216e+08])},
    753: {   'boxes': array([[114, 159, 161, 183]]),
             'classes': array([1]),
             'scores': array([4.99846843e+08])},
    754: {   'boxes': array([[150,  13, 175,  30]]),
             'classes': array([1]),
             'scores': array([4.99648644e+08])},
    755: {   'boxes': array([[161,   3, 190,  26]]),
             'classes': array([1]),
             'scores': array([4.99572975e+08])},
    756: {   'boxes': array([[167,   2, 203,  26]]),
             'classes': array([1]),
             'scores': array([4.99911157e+08])},
    757: {   'boxes': array([[194,   1, 238,  29]]),
             'classes': array([1]),
             'scores': array([4.99625834e+08])},
    758: {   'boxes': array([[155,  36, 190,  51]]),
             'classes': array([1]),
             'scores': array([4.99803174e+08])},
    759: {   'boxes': array([[159,  44, 214,  71]]),
             'classes': array([1]),
             'scores': array([4.99882803e+08])},
    760: {   'boxes': array([[165,  65, 216, 107]]),
             'classes': array([1]),
             'scores': array([4.9981503e+08])},
    761: {   'boxes': array([[154, 135, 232, 168]]),
             'classes': array([1]),
             'scores': array([4.99740663e+08])},
    762: {   'boxes': array([[184, 204, 275, 232]]),
             'classes': array([1]),
             'scores': array([4.99952487e+08])},
    763: {   'boxes': array([[176, 233, 275, 281]]),
             'classes': array([1]),
             'scores': array([4.99831238e+08])},
    764: {   'boxes': array([[147, 211, 214, 238]]),
             'classes': array([1]),
             'scores': array([4.99488166e+08])},
    765: {   'boxes': array([[139, 248, 195, 264]]),
             'classes': array([1]),
             'scores': array([4.99879681e+08])},
    766: {   'boxes': array([[142, 270, 199, 283]]),
             'classes': array([1]),
             'scores': array([4.99883153e+08])},
    767: {   'boxes': array([[137, 284, 198, 303]]),
             'classes': array([1]),
             'scores': array([4.99735797e+08])},
    768: {   'boxes': array([[147, 279, 204, 302]]),
             'classes': array([1]),
             'scores': array([4.99682854e+08])},
    769: {   'boxes': array([[147, 305, 199, 327]]),
             'classes': array([1]),
             'scores': array([4.99903553e+08])},
    770: {   'boxes': array([[137, 330, 197, 344]]),
             'classes': array([1]),
             'scores': array([4.99921843e+08])},
    771: {   'boxes': array([[147, 341, 200, 358]]),
             'classes': array([1]),
             'scores': array([4.9998613e+08])},
    772: {   'boxes': array([[157, 350, 246, 370]]),
             'classes': array([1]),
             'scores': array([4.99748298e+08])},
    773: {   'boxes': array([[151, 389, 225, 415]]),
             'classes': array([1]),
             'scores': array([4.99588785e+08])},
    774: {   'boxes': array([[141, 377, 207, 390]]),
             'classes': array([1]),
             'scores': array([4.99628825e+08])},
    775: {   'boxes': array([[158, 415, 260, 432]]),
             'classes': array([1]),
             'scores': array([5.00001714e+08])},
    776: {   'boxes': array([[141, 421, 183, 438]]),
             'classes': array([1]),
             'scores': array([4.99806328e+08])},
    777: {   'boxes': array([[139, 444, 166, 461]]),
             'classes': array([1]),
             'scores': array([4.99727595e+08])},
    778: {   'boxes': array([[156, 472, 185, 496]]),
             'classes': array([1]),
             'scores': array([4.99839201e+08])},
    779: {   'boxes': array([[163, 446, 182, 465]]),
             'classes': array([1]),
             'scores': array([4.99711478e+08])},
    780: {   'boxes': array([[185, 482, 275, 500]]),
             'classes': array([1]),
             'scores': array([4.9982577e+08])},
    781: {   'boxes': array([[187, 429, 275, 486]]),
             'classes': array([1]),
             'scores': array([4.99989335e+08])},
    782: {   'boxes': array([[244, 332, 276, 377]]),
             'classes': array([1]),
             'scores': array([4.99650771e+08])},
    783: {   'boxes': array([[ 32,  33, 500, 333]]),
             'classes': array([1]),
             'scores': array([4.99749996e+08])},
    784: {   'boxes': array([[176,   1, 481, 189]]),
             'classes': array([1]),
             'scores': array([4.99537259e+08])},
    785: {   'boxes': array([[  9,  61, 400, 432]]),
             'classes': array([1]),
             'scores': array([4.99756244e+08])},
    786: {   'boxes': array([[ 55, 196, 184, 228]]),
             'classes': array([1]),
             'scores': array([4.99928263e+08])},
    787: {   'boxes': array([[ 55,  70, 230, 114]]),
             'classes': array([1]),
             'scores': array([4.99816777e+08])},
    788: {   'boxes': array([[ 62, 235, 187, 281]]),
             'classes': array([1]),
             'scores': array([4.99745075e+08])},
    789: {   'boxes': array([[ 57, 269, 202, 319]]),
             'classes': array([1]),
             'scores': array([4.99908735e+08])},
    790: {   'boxes': array([[140, 289, 375, 480]]),
             'classes': array([1]),
             'scores': array([4.99659364e+08])},
    791: {   'boxes': array([[130, 291, 355, 400]]),
             'classes': array([1]),
             'scores': array([4.99720093e+08])},
    792: {   'boxes': array([[124, 197, 199, 285]]),
             'classes': array([1]),
             'scores': array([4.99858725e+08])},
    793: {   'boxes': array([[131,  15, 375, 133]]),
             'classes': array([1]),
             'scores': array([4.99743553e+08])},
    794: {   'boxes': array([[180, 134, 375, 250]]),
             'classes': array([1]),
             'scores': array([4.99755192e+08])},
    795: {   'boxes': array([[ 89, 135, 453, 271]]),
             'classes': array([1]),
             'scores': array([4.99845749e+08])},
    796: {   'boxes': array([[ 36,   1, 303, 417]]),
             'classes': array([1]),
             'scores': array([4.99716817e+08])},
    797: {   'boxes': array([[107,  56, 387, 230]]),
             'classes': array([1]),
             'scores': array([4.99973989e+08])},
    798: {   'boxes': array([[  1, 210, 108, 333]]),
             'classes': array([1]),
             'scores': array([4.99753938e+08])},
    799: {   'boxes': array([[ 87,  84, 366, 333]]),
             'classes': array([1]),
             'scores': array([4.99674064e+08])},
    800: {   'boxes': array([[ 93, 298, 122, 325]]),
             'classes': array([1]),
             'scores': array([4.99988256e+08])},
    801: {   'boxes': array([[ 92, 348, 123, 395]]),
             'classes': array([1]),
             'scores': array([4.99911842e+08])},
    802: {   'boxes': array([[ 87,  15, 196,  43]]),
             'classes': array([1]),
             'scores': array([4.99633814e+08])},
    803: {   'boxes': array([[ 88,  40, 128,  65]]),
             'classes': array([1]),
             'scores': array([4.99878087e+08])},
    804: {   'boxes': array([[ 84, 324, 130, 362]]),
             'classes': array([1]),
             'scores': array([4.99919143e+08])},
    805: {   'boxes': array([[ 86, 428, 153, 465]]),
             'classes': array([1]),
             'scores': array([4.99407359e+08])},
    806: {   'boxes': array([[ 93, 475, 234, 500]]),
             'classes': array([1]),
             'scores': array([4.99628033e+08])},
    807: {   'boxes': array([[ 80, 281, 268, 364]]),
             'classes': array([1]),
             'scores': array([4.99826668e+08])},
    808: {   'boxes': array([[  1, 189, 438, 372]]),
             'classes': array([1]),
             'scores': array([4.99661708e+08])},
    809: {   'boxes': array([[181, 116, 476, 374]]),
             'classes': array([1]),
             'scores': array([4.99572301e+08])},
    810: {   'boxes': array([[  3,   3, 498, 222]]),
             'classes': array([1]),
             'scores': array([4.99733889e+08])},
    811: {   'boxes': array([[  3, 188,  88, 240]]),
             'classes': array([1]),
             'scores': array([4.99524703e+08])},
    812: {   'boxes': array([[  1, 231,  70, 257]]),
             'classes': array([1]),
             'scores': array([4.99701612e+08])},
    813: {   'boxes': array([[ 3, 11, 76, 53]]),
             'classes': array([1]),
             'scores': array([4.99570594e+08])},
    814: {   'boxes': array([[ 58, 229, 266, 319]]),
             'classes': array([1]),
             'scores': array([4.99945436e+08])},
    815: {   'boxes': array([[114, 159, 297, 267]]),
             'classes': array([1]),
             'scores': array([4.99821718e+08])},
    816: {   'boxes': array([[154, 100, 286, 235]]),
             'classes': array([1]),
             'scores': array([4.99818703e+08])},
    817: {   'boxes': array([[146, 211, 352, 303]]),
             'classes': array([1]),
             'scores': array([4.99842093e+08])},
    818: {   'boxes': array([[154, 245, 374, 398]]),
             'classes': array([1]),
             'scores': array([4.99868445e+08])},
    819: {   'boxes': array([[271, 445, 327, 466]]),
             'classes': array([1]),
             'scores': array([4.99893239e+08])},
    820: {   'boxes': array([[268, 427, 324, 449]]),
             'classes': array([1]),
             'scores': array([4.99719001e+08])},
    821: {   'boxes': array([[160, 488, 217, 500]]),
             'classes': array([1]),
             'scores': array([4.99769074e+08])},
    822: {   'boxes': array([[164, 465, 217, 484]]),
             'classes': array([1]),
             'scores': array([4.99780596e+08])},
    823: {   'boxes': array([[ 70, 183, 253, 330]]),
             'classes': array([1]),
             'scores': array([4.99796881e+08])},
    824: {   'boxes': array([[119, 242, 244, 386]]),
             'classes': array([1]),
             'scores': array([4.99801169e+08])},
    825: {   'boxes': array([[101,  97, 251, 221]]),
             'classes': array([1]),
             'scores': array([5.0015708e+08])},
    826: {   'boxes': array([[124, 147, 345, 300]]),
             'classes': array([1]),
             'scores': array([4.99659704e+08])},
    827: {   'boxes': array([[125,  51, 187,  92]]),
             'classes': array([1]),
             'scores': array([4.99757707e+08])},
    828: {   'boxes': array([[122, 100, 182, 153]]),
             'classes': array([1]),
             'scores': array([4.99426124e+08])},
    829: {   'boxes': array([[ 68, 138, 242, 225]]),
             'classes': array([1]),
             'scores': array([4.99750086e+08])},
    830: {   'boxes': array([[284, 289, 314, 305]]),
             'classes': array([1]),
             'scores': array([4.9970875e+08])},
    831: {   'boxes': array([[ 58,  43, 128,  75]]),
             'classes': array([1]),
             'scores': array([4.99672611e+08])},
    832: {   'boxes': array([[ 65, 458,  99, 474]]),
             'classes': array([1]),
             'scores': array([4.99778006e+08])},
    833: {   'boxes': array([[ 61, 439,  99, 458]]),
             'classes': array([1]),
             'scores': array([4.99722938e+08])},
    834: {   'boxes': array([[ 64, 356,  87, 373]]),
             'classes': array([1]),
             'scores': array([4.99801275e+08])},
    835: {   'boxes': array([[ 56, 284, 102, 304]]),
             'classes': array([1]),
             'scores': array([4.99657318e+08])},
    836: {   'boxes': array([[ 61, 106, 110, 125]]),
             'classes': array([1]),
             'scores': array([4.99806766e+08])},
    837: {   'boxes': array([[ 60, 172, 101, 185]]),
             'classes': array([1]),
             'scores': array([4.99776784e+08])},
    838: {   'boxes': array([[ 66, 196,  85, 210]]),
             'classes': array([1]),
             'scores': array([4.99772908e+08])},
    839: {   'boxes': array([[ 64, 153,  77, 162]]),
             'classes': array([1]),
             'scores': array([4.99711233e+08])},
    840: {   'boxes': array([[ 64, 163,  88, 173]]),
             'classes': array([1]),
             'scores': array([4.99698883e+08])},
    841: {   'boxes': array([[ 79, 235, 354, 438]]),
             'classes': array([1]),
             'scores': array([4.99647408e+08])},
    842: {   'boxes': array([[ 88, 144, 375, 500]]),
             'classes': array([1]),
             'scores': array([4.99640684e+08])},
    843: {   'boxes': array([[  3, 125, 185, 279]]),
             'classes': array([1]),
             'scores': array([4.99724173e+08])},
    844: {   'boxes': array([[ 31, 138, 251, 329]]),
             'classes': array([1]),
             'scores': array([4.99732538e+08])},
    845: {   'boxes': array([[ 63, 173, 341, 432]]),
             'classes': array([1]),
             'scores': array([4.9975939e+08])},
    846: {   'boxes': array([[332, 146, 376, 255]]),
             'classes': array([1]),
             'scores': array([4.99326171e+08])},
    847: {   'boxes': array([[141, 251, 172, 278]]),
             'classes': array([1]),
             'scores': array([4.99666199e+08])},
    848: {   'boxes': array([[ 33, 109, 299, 251]]),
             'classes': array([1]),
             'scores': array([4.99858705e+08])},
    849: {   'boxes': array([[183,  45, 333, 169]]),
             'classes': array([1]),
             'scores': array([4.99780016e+08])},
    850: {   'boxes': array([[ 77, 389, 287, 475]]),
             'classes': array([1]),
             'scores': array([4.99659811e+08])},
    851: {   'boxes': array([[ 90, 424, 375, 500]]),
             'classes': array([1]),
             'scores': array([4.99864776e+08])},
    852: {   'boxes': array([[ 82, 183, 375, 312]]),
             'classes': array([1]),
             'scores': array([4.9974661e+08])},
    853: {   'boxes': array([[ 33,  35, 375, 222]]),
             'classes': array([1]),
             'scores': array([4.99733119e+08])},
    854: {   'boxes': array([[ 14,  51, 500, 359]]),
             'classes': array([1]),
             'scores': array([4.99795203e+08])},
    855: {   'boxes': array([[224,   2, 487,  93]]),
             'classes': array([1]),
             'scores': array([4.99695043e+08])},
    856: {   'boxes': array([[223, 212, 300, 238]]),
             'classes': array([1]),
             'scores': array([4.99881869e+08])},
    857: {   'boxes': array([[255, 292, 273, 304]]),
             'classes': array([1]),
             'scores': array([4.99700362e+08])},
    858: {   'boxes': array([[233, 238, 247, 244]]),
             'classes': array([1]),
             'scores': array([4.99694513e+08])},
    859: {   'boxes': array([[148, 330, 244, 376]]),
             'classes': array([1]),
             'scores': array([4.99624209e+08])},
    860: {   'boxes': array([[422, 146, 484, 161]]),
             'classes': array([1]),
             'scores': array([4.99718584e+08])},
    861: {   'boxes': array([[420, 190, 489, 212]]),
             'classes': array([1]),
             'scores': array([4.99832773e+08])},
    862: {   'boxes': array([[422, 259, 493, 281]]),
             'classes': array([1]),
             'scores': array([4.99901019e+08])},
    863: {   'boxes': array([[279, 139, 366, 179]]),
             'classes': array([1]),
             'scores': array([4.99743689e+08])},
    864: {   'boxes': array([[ 55, 149, 241, 220]]),
             'classes': array([1]),
             'scores': array([4.99305012e+08])},
    865: {   'boxes': array([[ 91, 366, 117, 397]]),
             'classes': array([1]),
             'scores': array([4.99677077e+08])},
    866: {   'boxes': array([[ 52,  12, 333, 312]]),
             'classes': array([1]),
             'scores': array([4.99640312e+08])},
    867: {   'boxes': array([[131, 219, 332, 467]]),
             'classes': array([1]),
             'scores': array([4.99689724e+08])},
    868: {   'boxes': array([[396,  45, 500, 188]]),
             'classes': array([1]),
             'scores': array([4.99449563e+08])},
    869: {   'boxes': array([[131, 134, 462, 230]]),
             'classes': array([1]),
             'scores': array([4.99687903e+08])},
    870: {   'boxes': array([[ 99, 285, 372, 441]]),
             'classes': array([1]),
             'scores': array([4.99852248e+08])},
    871: {   'boxes': array([[ 84, 139, 212, 205]]),
             'classes': array([1]),
             'scores': array([4.9964373e+08])},
    872: {   'boxes': array([[ 70, 281, 210, 339]]),
             'classes': array([1]),
             'scores': array([4.99397133e+08])},
    873: {   'boxes': array([[ 18,   3, 298, 114]]),
             'classes': array([1]),
             'scores': array([4.99918363e+08])},
    874: {   'boxes': array([[ 51, 109, 297, 214]]),
             'classes': array([1]),
             'scores': array([4.9980711e+08])},
    875: {   'boxes': array([[ 35, 212, 297, 316]]),
             'classes': array([1]),
             'scores': array([4.99836614e+08])},
    876: {   'boxes': array([[ 33, 289, 297, 387]]),
             'classes': array([1]),
             'scores': array([4.99635348e+08])},
    877: {   'boxes': array([[ 15, 381, 297, 500]]),
             'classes': array([1]),
             'scores': array([4.99729487e+08])},
    878: {   'boxes': array([[ 63, 279, 375, 463]]),
             'classes': array([1]),
             'scores': array([4.99800932e+08])},
    879: {   'boxes': array([[ 60, 103, 375, 225]]),
             'classes': array([1]),
             'scores': array([4.9975635e+08])},
    880: {   'boxes': array([[130, 226, 196, 262]]),
             'classes': array([1]),
             'scores': array([4.99759406e+08])},
    881: {   'boxes': array([[159, 296, 205, 317]]),
             'classes': array([1]),
             'scores': array([4.99729384e+08])},
    882: {   'boxes': array([[169, 351, 212, 371]]),
             'classes': array([1]),
             'scores': array([5.00014918e+08])},
    883: {   'boxes': array([[172, 112, 281, 309]]),
             'classes': array([1]),
             'scores': array([4.99776155e+08])},
    884: {   'boxes': array([[ 87, 414, 243, 500]]),
             'classes': array([1]),
             'scores': array([4.99791807e+08])},
    885: {   'boxes': array([[ 16, 104, 332, 367]]),
             'classes': array([1]),
             'scores': array([4.99921831e+08])},
    886: {   'boxes': array([[  1,   1, 375, 440]]),
             'classes': array([1]),
             'scores': array([4.99718005e+08])},
    887: {   'boxes': array([[145, 133, 328, 287]]),
             'classes': array([1]),
             'scores': array([4.99470273e+08])},
    888: {   'boxes': array([[ 20,  26, 330, 231]]),
             'classes': array([1]),
             'scores': array([4.996381e+08])},
    889: {   'boxes': array([[ 79, 399, 125, 421]]),
             'classes': array([1]),
             'scores': array([5.00374016e+08])},
    890: {   'boxes': array([[ 97, 205, 375, 355]]),
             'classes': array([1]),
             'scores': array([5.00090293e+08])},
    891: {   'boxes': array([[163, 188, 236, 217]]),
             'classes': array([1]),
             'scores': array([4.99905116e+08])},
    892: {   'boxes': array([[110, 344, 273, 446]]),
             'classes': array([1]),
             'scores': array([4.99770714e+08])},
    893: {   'boxes': array([[ 86, 262, 199, 360]]),
             'classes': array([1]),
             'scores': array([4.99610947e+08])},
    894: {   'boxes': array([[125, 130, 202, 200]]),
             'classes': array([1]),
             'scores': array([4.99714963e+08])},
    895: {   'boxes': array([[131, 212, 202, 288]]),
             'classes': array([1]),
             'scores': array([4.99698154e+08])},
    896: {   'boxes': array([[110,   5, 333, 132]]),
             'classes': array([1]),
             'scores': array([4.99700296e+08])},
    897: {   'boxes': array([[  2, 150, 330, 481]]),
             'classes': array([1]),
             'scores': array([4.99712506e+08])},
    898: {   'boxes': array([[  7, 113, 331, 256]]),
             'classes': array([1]),
             'scores': array([4.99571247e+08])},
    899: {   'boxes': array([[  1, 103, 242, 253]]),
             'classes': array([1]),
             'scores': array([4.99583268e+08])},
    900: {   'boxes': array([[  1, 212,  65, 270]]),
             'classes': array([1]),
             'scores': array([4.99519659e+08])},
    901: {   'boxes': array([[  3,   1, 344, 220]]),
             'classes': array([1]),
             'scores': array([4.99703839e+08])},
    902: {   'boxes': array([[  1,  79, 177, 225]]),
             'classes': array([1]),
             'scores': array([4.99532125e+08])},
    903: {   'boxes': array([[ 33, 268,  74, 316]]),
             'classes': array([1]),
             'scores': array([4.99500107e+08])},
    904: {   'boxes': array([[ 12, 313, 108, 363]]),
             'classes': array([1]),
             'scores': array([4.9985003e+08])},
    905: {   'boxes': array([[  3, 362, 124, 427]]),
             'classes': array([1]),
             'scores': array([4.99697233e+08])},
    906: {   'boxes': array([[ 36, 424,  76, 456]]),
             'classes': array([1]),
             'scores': array([4.99571778e+08])},
    907: {   'boxes': array([[135, 114, 406, 269]]),
             'classes': array([1]),
             'scores': array([4.99717638e+08])},
    908: {   'boxes': array([[157,   1, 479, 128]]),
             'classes': array([1]),
             'scores': array([4.99693875e+08])},
    909: {   'boxes': array([[ 67, 209, 341, 433]]),
             'classes': array([1]),
             'scores': array([4.99738699e+08])},
    910: {   'boxes': array([[ 96,   3, 337, 189]]),
             'classes': array([1]),
             'scores': array([4.99762165e+08])},
    911: {   'boxes': array([[ 40, 211, 237, 331]]),
             'classes': array([1]),
             'scores': array([4.99618504e+08])},
    912: {   'boxes': array([[128, 125, 433, 276]]),
             'classes': array([1]),
             'scores': array([4.99907224e+08])},
    913: {   'boxes': array([[336, 195, 500, 382]]),
             'classes': array([1]),
             'scores': array([4.99646464e+08])},
    914: {   'boxes': array([[259, 264, 355, 339]]),
             'classes': array([1]),
             'scores': array([4.99835996e+08])},
    915: {   'boxes': array([[242, 315, 424, 412]]),
             'classes': array([1]),
             'scores': array([4.99749788e+08])},
    916: {   'boxes': array([[238,  17, 410, 128]]),
             'classes': array([1]),
             'scores': array([4.99856388e+08])},
    917: {   'boxes': array([[ 93, 335, 218, 400]]),
             'classes': array([1]),
             'scores': array([4.99732435e+08])},
    918: {   'boxes': array([[125, 320, 199, 383]]),
             'classes': array([1]),
             'scores': array([4.99504926e+08])},
    919: {   'boxes': array([[ 45, 371, 108, 422]]),
             'classes': array([1]),
             'scores': array([4.9978262e+08])},
    920: {   'boxes': array([[154,   1, 218,  15]]),
             'classes': array([1]),
             'scores': array([4.99646816e+08])},
    921: {   'boxes': array([[  1,   3, 375, 500]]),
             'classes': array([1]),
             'scores': array([4.99732176e+08])},
    922: {   'boxes': array([[213, 159, 500, 292]]),
             'classes': array([1]),
             'scores': array([5.00022523e+08])},
    923: {   'boxes': array([[171, 158, 500, 425]]),
             'classes': array([1]),
             'scores': array([4.99771247e+08])},
    924: {   'boxes': array([[100, 192, 265, 348]]),
             'classes': array([1]),
             'scores': array([4.99742785e+08])},
    925: {   'boxes': array([[ 67, 320, 279, 493]]),
             'classes': array([1]),
             'scores': array([4.99913354e+08])},
    926: {   'boxes': array([[  4, 343, 100, 416]]),
             'classes': array([1]),
             'scores': array([4.99881639e+08])},
    927: {   'boxes': array([[ 56, 170, 159, 271]]),
             'classes': array([1]),
             'scores': array([4.99769323e+08])},
    928: {   'boxes': array([[ 45, 209, 100, 292]]),
             'classes': array([1]),
             'scores': array([4.99729947e+08])},
    929: {   'boxes': array([[238, 200, 391, 247]]),
             'classes': array([1]),
             'scores': array([4.99680473e+08])},
    930: {   'boxes': array([[253, 156, 291, 167]]),
             'classes': array([1]),
             'scores': array([4.99740021e+08])},
    931: {   'boxes': array([[252, 139, 291, 153]]),
             'classes': array([1]),
             'scores': array([4.99712529e+08])},
    932: {   'boxes': array([[  7, 104, 475, 280]]),
             'classes': array([1]),
             'scores': array([4.99649124e+08])},
    933: {   'boxes': array([[ 55, 200, 217, 320]]),
             'classes': array([1]),
             'scores': array([5.00016758e+08])},
    934: {   'boxes': array([[ 14,  26, 114,  54]]),
             'classes': array([1]),
             'scores': array([4.99811667e+08])},
    935: {   'boxes': array([[ 27, 190, 214, 268]]),
             'classes': array([1]),
             'scores': array([4.99509797e+08])},
    936: {   'boxes': array([[ 95,  74, 358, 200]]),
             'classes': array([1]),
             'scores': array([4.99634692e+08])},
    937: {   'boxes': array([[196, 155, 500, 343]]),
             'classes': array([1]),
             'scores': array([4.99681762e+08])},
    938: {   'boxes': array([[ 32, 333, 255, 373]]),
             'classes': array([1]),
             'scores': array([5.00003213e+08])},
    939: {   'boxes': array([[ 39, 413, 131, 441]]),
             'classes': array([1]),
             'scores': array([4.9983478e+08])},
    940: {   'boxes': array([[56, 50, 90, 64]]),
             'classes': array([1]),
             'scores': array([4.99588414e+08])},
    941: {   'boxes': array([[ 53,  12, 499, 438]]),
             'classes': array([1]),
             'scores': array([4.99762767e+08])},
    942: {   'boxes': array([[121,  63, 500, 342]]),
             'classes': array([1]),
             'scores': array([4.99718907e+08])},
    943: {   'boxes': array([[100, 274, 275, 402]]),
             'classes': array([1]),
             'scores': array([4.99652643e+08])},
    944: {   'boxes': array([[ 66, 145, 266, 284]]),
             'classes': array([1]),
             'scores': array([4.99735344e+08])},
    945: {   'boxes': array([[121,  28, 214,  85]]),
             'classes': array([1]),
             'scores': array([4.99686604e+08])},
    946: {   'boxes': array([[ 82,   2, 161,  29]]),
             'classes': array([1]),
             'scores': array([4.99772895e+08])},
    947: {   'boxes': array([[131, 117, 259, 167]]),
             'classes': array([1]),
             'scores': array([5.00054869e+08])},
    948: {   'boxes': array([[192, 107, 333, 256]]),
             'classes': array([1]),
             'scores': array([4.99669547e+08])},
    949: {   'boxes': array([[ 62, 119, 110, 157]]),
             'classes': array([1]),
             'scores': array([4.99662212e+08])},
    950: {   'boxes': array([[ 86,   3, 333, 372]]),
             'classes': array([1]),
             'scores': array([4.99839017e+08])},
    951: {   'boxes': array([[ 11,  34, 330, 500]]),
             'classes': array([1]),
             'scores': array([4.99941987e+08])},
    952: {   'boxes': array([[168, 112, 500, 206]]),
             'classes': array([1]),
             'scores': array([4.99739421e+08])},
    953: {   'boxes': array([[ 86, 122, 376, 227]]),
             'classes': array([1]),
             'scores': array([4.99654739e+08])},
    954: {   'boxes': array([[183,   1, 500, 203]]),
             'classes': array([1]),
             'scores': array([4.99595332e+08])},
    955: {   'boxes': array([[ 55, 333, 286, 405]]),
             'classes': array([1]),
             'scores': array([4.99663938e+08])},
    956: {   'boxes': array([[ 23, 455,  99, 480]]),
             'classes': array([1]),
             'scores': array([5.0017101e+08])},
    957: {   'boxes': array([[ 29, 323,  98, 349]]),
             'classes': array([1]),
             'scores': array([4.99801191e+08])},
    958: {   'boxes': array([[ 32, 233,  85, 258]]),
             'classes': array([1]),
             'scores': array([4.99696954e+08])},
    959: {   'boxes': array([[ 29, 207,  82, 230]]),
             'classes': array([1]),
             'scores': array([4.99754479e+08])},
    960: {   'boxes': array([[ 46, 190,  82, 208]]),
             'classes': array([1]),
             'scores': array([5.00048235e+08])},
    961: {   'boxes': array([[ 32, 130,  66, 153]]),
             'classes': array([1]),
             'scores': array([4.99832926e+08])},
    962: {   'boxes': array([[27, 59, 65, 79]]),
             'classes': array([1]),
             'scores': array([4.99797112e+08])},
    963: {   'boxes': array([[18, 77, 64, 97]]),
             'classes': array([1]),
             'scores': array([4.99738441e+08])},
    964: {   'boxes': array([[26, 48, 67, 61]]),
             'classes': array([1]),
             'scores': array([4.99907758e+08])},
    965: {   'boxes': array([[29, 35, 68, 54]]),
             'classes': array([1]),
             'scores': array([4.99707213e+08])},
    966: {   'boxes': array([[18,  1, 69, 19]]),
             'classes': array([1]),
             'scores': array([4.9973284e+08])},
    967: {   'boxes': array([[20, 16, 70, 37]]),
             'classes': array([1]),
             'scores': array([5.00257906e+08])},
    968: {   'boxes': array([[  1, 312, 153, 430]]),
             'classes': array([1]),
             'scores': array([5.00011491e+08])},
    969: {   'boxes': array([[  1,   3, 209,  41]]),
             'classes': array([1]),
             'scores': array([4.9984741e+08])},
    970: {   'boxes': array([[ 1, 35, 89, 72]]),
             'classes': array([1]),
             'scores': array([4.99438851e+08])},
    971: {   'boxes': array([[170, 418, 218, 447]]),
             'classes': array([1]),
             'scores': array([4.99808385e+08])},
    972: {   'boxes': array([[174, 411, 194, 428]]),
             'classes': array([1]),
             'scores': array([4.99943439e+08])},
    973: {   'boxes': array([[181, 389, 228, 424]]),
             'classes': array([1]),
             'scores': array([4.99953746e+08])},
    974: {   'boxes': array([[117, 216, 165, 250]]),
             'classes': array([1]),
             'scores': array([4.99667445e+08])},
    975: {   'boxes': array([[160, 371, 404, 476]]),
             'classes': array([1]),
             'scores': array([4.99716142e+08])},
    976: {   'boxes': array([[ 99, 227, 404, 365]]),
             'classes': array([1]),
             'scores': array([4.99692653e+08])},
    977: {   'boxes': array([[169, 229, 341, 383]]),
             'classes': array([1]),
             'scores': array([4.99898463e+08])},
    978: {   'boxes': array([[193, 229, 324, 309]]),
             'classes': array([1]),
             'scores': array([4.99658478e+08])},
    979: {   'boxes': array([[224, 409, 309, 499]]),
             'classes': array([1]),
             'scores': array([4.99650157e+08])},
    980: {   'boxes': array([[223, 371, 273, 401]]),
             'classes': array([1]),
             'scores': array([4.99638147e+08])},
    981: {   'boxes': array([[231, 474, 243, 497]]),
             'classes': array([1]),
             'scores': array([4.99665206e+08])},
    982: {   'boxes': array([[ 11,  78, 372, 309]]),
             'classes': array([1]),
             'scores': array([4.99672076e+08])},
    983: {   'boxes': array([[113,   1, 390, 285]]),
             'classes': array([1]),
             'scores': array([4.99923266e+08])},
    984: {   'boxes': array([[194, 387, 255, 429]]),
             'classes': array([1]),
             'scores': array([4.9976783e+08])},
    985: {   'boxes': array([[201,   1, 317,  42]]),
             'classes': array([1]),
             'scores': array([4.99760764e+08])},
    986: {   'boxes': array([[  2,  41, 104, 138]]),
             'classes': array([1]),
             'scores': array([5.00328951e+08])},
    987: {   'boxes': array([[107, 284, 280, 411]]),
             'classes': array([1]),
             'scores': array([4.99728677e+08])},
    988: {   'boxes': array([[ 83, 126, 285, 257]]),
             'classes': array([1]),
             'scores': array([4.99831742e+08])},
    989: {   'boxes': array([[167,   9, 333, 160]]),
             'classes': array([1]),
             'scores': array([4.99803429e+08])},
    990: {   'boxes': array([[ 36, 269, 203, 341]]),
             'classes': array([1]),
             'scores': array([4.99650803e+08])},
    991: {   'boxes': array([[ 84, 162, 368, 239]]),
             'classes': array([1]),
             'scores': array([4.99688238e+08])},
    992: {   'boxes': array([[ 26,  24, 498, 333]]),
             'classes': array([1]),
             'scores': array([4.99644041e+08])},
    993: {   'boxes': array([[ 54,  15, 494, 275]]),
             'classes': array([1]),
             'scores': array([4.99641559e+08])},
    994: {   'boxes': array([[146, 281, 394, 374]]),
             'classes': array([1]),
             'scores': array([4.99723781e+08])},
    995: {   'boxes': array([[104, 256, 415, 363]]),
             'classes': array([1]),
             'scores': array([4.99854188e+08])},
    996: {   'boxes': array([[ 33,   5, 453, 262]]),
             'classes': array([1]),
             'scores': array([4.99688765e+08])},
    997: {   'boxes': array([[108,   4, 500, 232]]),
             'classes': array([1]),
             'scores': array([4.9981278e+08])},
    998: {   'boxes': array([[114, 160, 375, 366]]),
             'classes': array([1]),
             'scores': array([4.99852224e+08])},
    999: {   'boxes': array([[  2, 298, 375, 500]]),
             'classes': array([1]),
             'scores': array([5.00020542e+08])},
    1000: {   'boxes': array([[  3,   1, 181, 300]]),
              'classes': array([1]),
              'scores': array([4.99357825e+08])},
    1001: {   'boxes': array([[ 28, 394,  98, 422]]),
              'classes': array([1]),
              'scores': array([5.00083148e+08])},
    1002: {   'boxes': array([[  8, 361,  91, 381]]),
              'classes': array([1]),
              'scores': array([4.99685465e+08])},
    1003: {   'boxes': array([[  8, 342,  85, 362]]),
              'classes': array([1]),
              'scores': array([4.99707448e+08])},
    1004: {   'boxes': array([[  2, 278,  70, 308]]),
              'classes': array([1]),
              'scores': array([4.99718012e+08])},
    1005: {   'boxes': array([[  1, 250,  58, 271]]),
              'classes': array([1]),
              'scores': array([4.99640554e+08])},
    1006: {   'boxes': array([[ 72, 117, 177, 188]]),
              'classes': array([1]),
              'scores': array([4.99777953e+08])},
    1007: {   'boxes': array([[ 84, 222, 173, 270]]),
              'classes': array([1]),
              'scores': array([5.00138822e+08])},
    1008: {   'boxes': array([[240,  27, 329,  99]]),
              'classes': array([1]),
              'scores': array([4.99709605e+08])},
    1009: {   'boxes': array([[309,  56, 375, 126]]),
              'classes': array([1]),
              'scores': array([4.9990397e+08])},
    1010: {   'boxes': array([[195, 246, 294, 306]]),
              'classes': array([1]),
              'scores': array([4.99366292e+08])},
    1011: {   'boxes': array([[229,  97, 294, 113]]),
              'classes': array([1]),
              'scores': array([5.00209332e+08])},
    1012: {   'boxes': array([[  2, 192, 335, 500]]),
              'classes': array([1]),
              'scores': array([4.99857907e+08])},
    1013: {   'boxes': array([[  1,  62, 332, 500]]),
              'classes': array([1]),
              'scores': array([4.99652951e+08])},
    1014: {   'boxes': array([[206, 359, 278, 386]]),
              'classes': array([1]),
              'scores': array([4.99688423e+08])},
    1015: {   'boxes': array([[234, 357, 375, 477]]),
              'classes': array([1]),
              'scores': array([4.99770471e+08])},
    1016: {   'boxes': array([[130,  62, 347, 275]]),
              'classes': array([1]),
              'scores': array([4.99718592e+08])},
    1017: {   'boxes': array([[ 70, 152, 500, 375]]),
              'classes': array([1]),
              'scores': array([4.99871745e+08])},
    1018: {   'boxes': array([[141, 178, 281, 301]]),
              'classes': array([1]),
              'scores': array([4.99684602e+08])},
    1019: {   'boxes': array([[  4,   5, 375, 265]]),
              'classes': array([1]),
              'scores': array([4.99744605e+08])},
    1020: {   'boxes': array([[ 35, 176, 255, 346]]),
              'classes': array([1]),
              'scores': array([4.99741068e+08])},
    1021: {   'boxes': array([[ 69,  40, 354, 261]]),
              'classes': array([1]),
              'scores': array([4.9929716e+08])},
    1022: {   'boxes': array([[ 13,  15, 266,  90]]),
              'classes': array([1]),
              'scores': array([4.99730967e+08])},
    1023: {   'boxes': array([[ 89, 412, 355, 500]]),
              'classes': array([1]),
              'scores': array([4.99582843e+08])},
    1024: {   'boxes': array([[287,  13, 391, 143]]),
              'classes': array([1]),
              'scores': array([4.99330938e+08])},
    1025: {   'boxes': array([[ 72,  84, 371, 242]]),
              'classes': array([1]),
              'scores': array([4.99789726e+08])},
    1026: {   'boxes': array([[156, 323, 235, 389]]),
              'classes': array([1]),
              'scores': array([5.00066052e+08])},
    1027: {   'boxes': array([[143, 194, 280, 257]]),
              'classes': array([1]),
              'scores': array([4.99865984e+08])},
    1028: {   'boxes': array([[151, 125, 214, 157]]),
              'classes': array([1]),
              'scores': array([4.99802405e+08])},
    1029: {   'boxes': array([[151,  96, 194, 118]]),
              'classes': array([1]),
              'scores': array([4.99968145e+08])},
    1030: {   'boxes': array([[213,  60, 382, 162]]),
              'classes': array([1]),
              'scores': array([4.99546677e+08])},
    1031: {   'boxes': array([[157, 117, 375, 209]]),
              'classes': array([1]),
              'scores': array([4.99623351e+08])},
    1032: {   'boxes': array([[100, 330, 194, 363]]),
              'classes': array([1]),
              'scores': array([4.99830164e+08])},
    1033: {   'boxes': array([[103, 238, 185, 267]]),
              'classes': array([1]),
              'scores': array([4.99787786e+08])},
    1034: {   'boxes': array([[ 85, 113, 169, 150]]),
              'classes': array([1]),
              'scores': array([4.99964627e+08])},
    1035: {   'boxes': array([[273,  19, 341,  78]]),
              'classes': array([1]),
              'scores': array([4.99618604e+08])},
    1036: {   'boxes': array([[ 51,   1, 375, 236]]),
              'classes': array([1]),
              'scores': array([4.99825688e+08])},
    1037: {   'boxes': array([[ 33,   2, 445, 289]]),
              'classes': array([1]),
              'scores': array([4.99842469e+08])},
    1038: {   'boxes': array([[123,  38, 440, 298]]),
              'classes': array([1]),
              'scores': array([4.99713544e+08])},
    1039: {   'boxes': array([[ 47,  38, 491, 265]]),
              'classes': array([1]),
              'scores': array([4.99784555e+08])},
    1040: {   'boxes': array([[  1,   1, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99306357e+08])},
    1041: {   'boxes': array([[ 83, 171, 300, 309]]),
              'classes': array([1]),
              'scores': array([4.99797225e+08])},
    1042: {   'boxes': array([[ 78,  57, 365, 320]]),
              'classes': array([1]),
              'scores': array([4.99530362e+08])},
    1043: {   'boxes': array([[187,  32, 229,  54]]),
              'classes': array([1]),
              'scores': array([4.99724596e+08])},
    1044: {   'boxes': array([[199, 332, 330, 375]]),
              'classes': array([1]),
              'scores': array([4.99560119e+08])},
    1045: {   'boxes': array([[193,  64, 487, 248]]),
              'classes': array([1]),
              'scores': array([5.00005807e+08])},
    1046: {   'boxes': array([[220,   2, 500, 278]]),
              'classes': array([1]),
              'scores': array([4.99754467e+08])},
    1047: {   'boxes': array([[ 35, 252, 375, 445]]),
              'classes': array([1]),
              'scores': array([4.997866e+08])},
    1048: {   'boxes': array([[ 17,  62, 375, 273]]),
              'classes': array([1]),
              'scores': array([4.9978655e+08])},
    1049: {   'boxes': array([[126, 100, 375, 284]]),
              'classes': array([1]),
              'scores': array([4.99730491e+08])},
    1050: {   'boxes': array([[162, 260, 375, 408]]),
              'classes': array([1]),
              'scores': array([5.00029682e+08])},
    1051: {   'boxes': array([[218, 360, 308, 422]]),
              'classes': array([1]),
              'scores': array([4.99710859e+08])},
    1052: {   'boxes': array([[246, 394, 375, 474]]),
              'classes': array([1]),
              'scores': array([4.99760362e+08])},
    1053: {   'boxes': array([[225, 419, 328, 472]]),
              'classes': array([1]),
              'scores': array([4.99602777e+08])},
    1054: {   'boxes': array([[ 38,  52, 499, 277]]),
              'classes': array([1]),
              'scores': array([4.99868375e+08])},
    1055: {   'boxes': array([[  9,  66, 363, 484]]),
              'classes': array([1]),
              'scores': array([4.99951101e+08])},
    1056: {   'boxes': array([[ 60, 218, 375, 381]]),
              'classes': array([1]),
              'scores': array([4.99769884e+08])},
    1057: {   'boxes': array([[ 83,  91, 375, 216]]),
              'classes': array([1]),
              'scores': array([4.99684524e+08])},
    1058: {   'boxes': array([[ 55,  61, 375, 249]]),
              'classes': array([1]),
              'scores': array([4.99858486e+08])},
    1059: {   'boxes': array([[ 36, 209,  75, 249]]),
              'classes': array([1]),
              'scores': array([4.99633307e+08])},
    1060: {   'boxes': array([[ 37, 266,  75, 299]]),
              'classes': array([1]),
              'scores': array([4.9971922e+08])},
    1061: {   'boxes': array([[241, 255, 334, 285]]),
              'classes': array([1]),
              'scores': array([4.99630462e+08])},
    1062: {   'boxes': array([[302,  95, 375, 167]]),
              'classes': array([1]),
              'scores': array([4.99672746e+08])},
    1063: {   'boxes': array([[298, 205, 372, 235]]),
              'classes': array([1]),
              'scores': array([4.99722907e+08])},
    1064: {   'boxes': array([[273, 185, 319, 217]]),
              'classes': array([1]),
              'scores': array([4.99765557e+08])},
    1065: {   'boxes': array([[273, 287, 345, 339]]),
              'classes': array([1]),
              'scores': array([4.99854693e+08])},
    1066: {   'boxes': array([[242, 384, 296, 415]]),
              'classes': array([1]),
              'scores': array([4.99807452e+08])},
    1067: {   'boxes': array([[195, 477, 273, 500]]),
              'classes': array([1]),
              'scores': array([4.99946729e+08])},
    1068: {   'boxes': array([[227, 400, 297, 436]]),
              'classes': array([1]),
              'scores': array([4.99871964e+08])},
    1069: {   'boxes': array([[  2,   2, 375, 238]]),
              'classes': array([1]),
              'scores': array([4.99806116e+08])},
    1070: {   'boxes': array([[177, 174, 267, 225]]),
              'classes': array([1]),
              'scores': array([5.00159084e+08])},
    1071: {   'boxes': array([[223, 139, 266, 150]]),
              'classes': array([1]),
              'scores': array([4.99795708e+08])},
    1072: {   'boxes': array([[223, 120, 265, 135]]),
              'classes': array([1]),
              'scores': array([4.99697534e+08])},
    1073: {   'boxes': array([[226, 104, 266, 119]]),
              'classes': array([1]),
              'scores': array([4.99740549e+08])},
    1074: {   'boxes': array([[219,  81, 263, 100]]),
              'classes': array([1]),
              'scores': array([4.99865931e+08])},
    1075: {   'boxes': array([[ 74, 104, 310, 258]]),
              'classes': array([1]),
              'scores': array([5.00197558e+08])},
    1076: {   'boxes': array([[107, 371, 284, 470]]),
              'classes': array([1]),
              'scores': array([4.99656804e+08])},
    1077: {   'boxes': array([[114, 201, 335, 422]]),
              'classes': array([1]),
              'scores': array([4.99792951e+08])},
    1078: {   'boxes': array([[117, 130, 246, 210]]),
              'classes': array([1]),
              'scores': array([4.99606574e+08])},
    1079: {   'boxes': array([[126,  40, 236, 132]]),
              'classes': array([1]),
              'scores': array([4.99658219e+08])},
    1080: {   'boxes': array([[ 55,   1, 285,  55]]),
              'classes': array([1]),
              'scores': array([4.99713725e+08])},
    1081: {   'boxes': array([[ 69, 199, 335, 391]]),
              'classes': array([1]),
              'scores': array([4.99805309e+08])},
    1082: {   'boxes': array([[184, 291, 375, 368]]),
              'classes': array([1]),
              'scores': array([5.0027957e+08])},
    1083: {   'boxes': array([[186, 362, 375, 449]]),
              'classes': array([1]),
              'scores': array([4.99888101e+08])},
    1084: {   'boxes': array([[ 28,  33, 321, 337]]),
              'classes': array([1]),
              'scores': array([4.99678699e+08])},
    1085: {   'boxes': array([[ 55, 284, 274, 349]]),
              'classes': array([1]),
              'scores': array([4.99901014e+08])},
    1086: {   'boxes': array([[126, 220, 327, 314]]),
              'classes': array([1]),
              'scores': array([4.99717974e+08])},
    1087: {   'boxes': array([[  2, 168, 325, 453]]),
              'classes': array([1]),
              'scores': array([4.99589188e+08])},
    1088: {   'boxes': array([[120,   2, 371,  92]]),
              'classes': array([1]),
              'scores': array([4.99651044e+08])},
    1089: {   'boxes': array([[  3,   3, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99901328e+08])},
    1090: {   'boxes': array([[  2, 291, 272, 472]]),
              'classes': array([1]),
              'scores': array([5.00271031e+08])},
    1091: {   'boxes': array([[ 57, 257, 141, 334]]),
              'classes': array([1]),
              'scores': array([4.99777524e+08])},
    1092: {   'boxes': array([[ 57, 244, 114, 279]]),
              'classes': array([1]),
              'scores': array([4.99772092e+08])},
    1093: {   'boxes': array([[ 70,   1, 243,  35]]),
              'classes': array([1]),
              'scores': array([4.99695954e+08])},
    1094: {   'boxes': array([[108,  69, 331, 238]]),
              'classes': array([1]),
              'scores': array([5.00006157e+08])},
    1095: {   'boxes': array([[240, 153, 291, 170]]),
              'classes': array([1]),
              'scores': array([4.99644933e+08])},
    1096: {   'boxes': array([[344, 335, 375, 354]]),
              'classes': array([1]),
              'scores': array([4.999306e+08])},
    1097: {   'boxes': array([[340, 187, 375, 209]]),
              'classes': array([1]),
              'scores': array([5.00012007e+08])},
    1098: {   'boxes': array([[339, 157, 375, 176]]),
              'classes': array([1]),
              'scores': array([4.99876615e+08])},
    1099: {   'boxes': array([[334, 146, 375, 171]]),
              'classes': array([1]),
              'scores': array([4.9981371e+08])},
    1100: {   'boxes': array([[ 71,  46, 202, 128]]),
              'classes': array([1]),
              'scores': array([4.99913908e+08])},
    1101: {   'boxes': array([[ 92,  86, 214, 176]]),
              'classes': array([1]),
              'scores': array([4.99844032e+08])},
    1102: {   'boxes': array([[ 85, 153, 202, 196]]),
              'classes': array([1]),
              'scores': array([4.99677368e+08])},
    1103: {   'boxes': array([[160, 156, 356, 249]]),
              'classes': array([1]),
              'scores': array([4.99758653e+08])},
    1104: {   'boxes': array([[156,  99, 363, 185]]),
              'classes': array([1]),
              'scores': array([4.99684356e+08])},
    1105: {   'boxes': array([[208, 231, 375, 389]]),
              'classes': array([1]),
              'scores': array([4.9975425e+08])},
    1106: {   'boxes': array([[174, 244, 265, 297]]),
              'classes': array([1]),
              'scores': array([5.00001966e+08])},
    1107: {   'boxes': array([[ 92, 350, 365, 430]]),
              'classes': array([1]),
              'scores': array([4.99814664e+08])},
    1108: {   'boxes': array([[ 87, 297, 322, 360]]),
              'classes': array([1]),
              'scores': array([4.99936366e+08])},
    1109: {   'boxes': array([[103, 240, 221, 308]]),
              'classes': array([1]),
              'scores': array([4.99801809e+08])},
    1110: {   'boxes': array([[ 85, 191, 204, 266]]),
              'classes': array([1]),
              'scores': array([4.99726729e+08])},
    1111: {   'boxes': array([[ 78, 245, 118, 305]]),
              'classes': array([1]),
              'scores': array([4.99807443e+08])},
    1112: {   'boxes': array([[ 83, 329, 120, 372]]),
              'classes': array([1]),
              'scores': array([4.99513102e+08])},
    1113: {   'boxes': array([[ 83, 157, 310, 337]]),
              'classes': array([1]),
              'scores': array([5.00166337e+08])},
    1114: {   'boxes': array([[103,   1, 337,  98]]),
              'classes': array([1]),
              'scores': array([5.00054478e+08])},
    1115: {   'boxes': array([[ 48, 180, 234, 481]]),
              'classes': array([1]),
              'scores': array([4.99523157e+08])},
    1116: {   'boxes': array([[151, 139, 321, 210]]),
              'classes': array([1]),
              'scores': array([4.99754059e+08])},
    1117: {   'boxes': array([[130, 355, 315, 411]]),
              'classes': array([1]),
              'scores': array([4.99952755e+08])},
    1118: {   'boxes': array([[ 52, 218, 375, 500]]),
              'classes': array([1]),
              'scores': array([5.00458702e+08])},
    1119: {   'boxes': array([[ 18, 164, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99696439e+08])},
    1120: {   'boxes': array([[203, 419, 246, 433]]),
              'classes': array([1]),
              'scores': array([4.99684737e+08])},
    1121: {   'boxes': array([[205, 430, 249, 454]]),
              'classes': array([1]),
              'scores': array([4.99650127e+08])},
    1122: {   'boxes': array([[216,  24, 263,  44]]),
              'classes': array([1]),
              'scores': array([4.99685217e+08])},
    1123: {   'boxes': array([[215,   1, 319,  19]]),
              'classes': array([1]),
              'scores': array([4.99685366e+08])},
    1124: {   'boxes': array([[ 94,  66, 375, 445]]),
              'classes': array([1]),
              'scores': array([4.99727092e+08])},
    1125: {   'boxes': array([[ 54,  84, 375, 262]]),
              'classes': array([1]),
              'scores': array([4.99604517e+08])},
    1126: {   'boxes': array([[195,  33, 375, 131]]),
              'classes': array([1]),
              'scores': array([4.99732305e+08])},
    1127: {   'boxes': array([[128, 135, 255, 167]]),
              'classes': array([1]),
              'scores': array([5.00110784e+08])},
    1128: {   'boxes': array([[118, 156, 245, 195]]),
              'classes': array([1]),
              'scores': array([4.99810929e+08])},
    1129: {   'boxes': array([[  2, 170, 107, 223]]),
              'classes': array([1]),
              'scores': array([4.99769909e+08])},
    1130: {   'boxes': array([[  1,  68, 157, 141]]),
              'classes': array([1]),
              'scores': array([4.9981144e+08])},
    1131: {   'boxes': array([[  1, 259, 120, 321]]),
              'classes': array([1]),
              'scores': array([4.99713722e+08])},
    1132: {   'boxes': array([[160, 174, 199, 207]]),
              'classes': array([1]),
              'scores': array([4.99782776e+08])},
    1133: {   'boxes': array([[151, 101, 188, 151]]),
              'classes': array([1]),
              'scores': array([4.99281772e+08])},
    1134: {   'boxes': array([[150, 183, 202, 226]]),
              'classes': array([1]),
              'scores': array([4.99896203e+08])},
    1135: {   'boxes': array([[  1, 437, 374, 500]]),
              'classes': array([1]),
              'scores': array([4.99834e+08])},
    1136: {   'boxes': array([[ 68,   1, 375, 207]]),
              'classes': array([1]),
              'scores': array([4.99942616e+08])},
    1137: {   'boxes': array([[101, 199, 304, 383]]),
              'classes': array([1]),
              'scores': array([4.99988861e+08])},
    1138: {   'boxes': array([[110, 316, 246, 468]]),
              'classes': array([1]),
              'scores': array([4.99732769e+08])},
    1139: {   'boxes': array([[  2,   1, 500, 333]]),
              'classes': array([1]),
              'scores': array([4.99805281e+08])},
    1140: {   'boxes': array([[146,  64, 375, 229]]),
              'classes': array([1]),
              'scores': array([4.99865687e+08])},
    1141: {   'boxes': array([[ 46,  78, 221, 194]]),
              'classes': array([1]),
              'scores': array([5.00348936e+08])},
    1142: {   'boxes': array([[ 96, 210, 298, 467]]),
              'classes': array([1]),
              'scores': array([4.99627881e+08])},
    1143: {   'boxes': array([[ 44, 132, 337, 245]]),
              'classes': array([1]),
              'scores': array([4.99865126e+08])},
    1144: {   'boxes': array([[ 90,   8, 315, 113]]),
              'classes': array([1]),
              'scores': array([4.99644695e+08])},
    1145: {   'boxes': array([[ 73,  69, 213, 104]]),
              'classes': array([1]),
              'scores': array([5.00036272e+08])},
    1146: {   'boxes': array([[ 68, 100, 258, 142]]),
              'classes': array([1]),
              'scores': array([4.99838422e+08])},
    1147: {   'boxes': array([[ 78, 266, 305, 344]]),
              'classes': array([1]),
              'scores': array([4.99836033e+08])},
    1148: {   'boxes': array([[ 92, 306, 332, 402]]),
              'classes': array([1]),
              'scores': array([4.997751e+08])},
    1149: {   'boxes': array([[ 84, 381, 361, 486]]),
              'classes': array([1]),
              'scores': array([4.99682845e+08])},
    1150: {   'boxes': array([[ 36,  98, 242, 203]]),
              'classes': array([1]),
              'scores': array([4.99685924e+08])},
    1151: {   'boxes': array([[ 44, 110, 300, 312]]),
              'classes': array([1]),
              'scores': array([4.99675355e+08])},
    1152: {   'boxes': array([[117, 316, 209, 366]]),
              'classes': array([1]),
              'scores': array([4.99900964e+08])},
    1153: {   'boxes': array([[127, 186, 206, 220]]),
              'classes': array([1]),
              'scores': array([4.99687984e+08])},
    1154: {   'boxes': array([[146,  48, 216,  93]]),
              'classes': array([1]),
              'scores': array([4.99742302e+08])},
    1155: {   'boxes': array([[184,   1, 257,  26]]),
              'classes': array([1]),
              'scores': array([4.99940092e+08])},
    1156: {   'boxes': array([[ 45,  46, 500, 270]]),
              'classes': array([1]),
              'scores': array([4.99700532e+08])},
    1157: {   'boxes': array([[ 16, 224, 330, 380]]),
              'classes': array([1]),
              'scores': array([4.99697231e+08])},
    1158: {   'boxes': array([[115,  29, 260, 155]]),
              'classes': array([1]),
              'scores': array([4.99658681e+08])},
    1159: {   'boxes': array([[114, 141, 243, 245]]),
              'classes': array([1]),
              'scores': array([4.99685821e+08])},
    1160: {   'boxes': array([[129, 279, 222, 351]]),
              'classes': array([1]),
              'scores': array([4.99590841e+08])},
    1161: {   'boxes': array([[131, 337, 216, 409]]),
              'classes': array([1]),
              'scores': array([4.99717246e+08])},
    1162: {   'boxes': array([[159, 324, 375, 397]]),
              'classes': array([1]),
              'scores': array([4.99808795e+08])},
    1163: {   'boxes': array([[100, 311, 169, 358]]),
              'classes': array([1]),
              'scores': array([4.99768453e+08])},
    1164: {   'boxes': array([[ 32, 458,  59, 468]]),
              'classes': array([1]),
              'scores': array([4.99651533e+08])},
    1165: {   'boxes': array([[ 83,  69, 321, 224]]),
              'classes': array([1]),
              'scores': array([4.99796016e+08])},
    1166: {   'boxes': array([[ 75, 280, 408, 375]]),
              'classes': array([1]),
              'scores': array([4.99670845e+08])},
    1167: {   'boxes': array([[228,   1, 439, 114]]),
              'classes': array([1]),
              'scores': array([4.99800411e+08])},
    1168: {   'boxes': array([[ 24,  64, 334, 496]]),
              'classes': array([1]),
              'scores': array([4.99757569e+08])},
    1169: {   'boxes': array([[ 86, 178, 333, 316]]),
              'classes': array([1]),
              'scores': array([5.00107728e+08])},
    1170: {   'boxes': array([[ 71, 292, 333, 409]]),
              'classes': array([1]),
              'scores': array([4.99793666e+08])},
    1171: {   'boxes': array([[130,   5, 218,  62]]),
              'classes': array([1]),
              'scores': array([4.99689205e+08])},
    1172: {   'boxes': array([[ 60,  58, 500, 285]]),
              'classes': array([1]),
              'scores': array([4.99602913e+08])},
    1173: {   'boxes': array([[100, 334, 285, 399]]),
              'classes': array([1]),
              'scores': array([4.99828231e+08])},
    1174: {   'boxes': array([[ 91,  76, 352, 460]]),
              'classes': array([1]),
              'scores': array([5.00278014e+08])},
    1175: {   'boxes': array([[102,   3, 500, 318]]),
              'classes': array([1]),
              'scores': array([4.99504356e+08])},
    1176: {   'boxes': array([[240, 263, 262, 281]]),
              'classes': array([1]),
              'scores': array([4.99739029e+08])},
    1177: {   'boxes': array([[236,  54, 322,  82]]),
              'classes': array([1]),
              'scores': array([4.99771786e+08])},
    1178: {   'boxes': array([[174,  41, 329, 191]]),
              'classes': array([1]),
              'scores': array([4.99655793e+08])},
    1179: {   'boxes': array([[244, 270, 320, 313]]),
              'classes': array([1]),
              'scores': array([4.99752595e+08])},
    1180: {   'boxes': array([[262, 126, 354, 176]]),
              'classes': array([1]),
              'scores': array([4.99873124e+08])},
    1181: {   'boxes': array([[ 82,  27, 116,  58]]),
              'classes': array([1]),
              'scores': array([4.99695503e+08])},
    1182: {   'boxes': array([[ 89, 321, 123, 345]]),
              'classes': array([1]),
              'scores': array([4.99708678e+08])},
    1183: {   'boxes': array([[ 90, 377, 121, 390]]),
              'classes': array([1]),
              'scores': array([4.99683873e+08])},
    1184: {   'boxes': array([[ 93, 397, 141, 419]]),
              'classes': array([1]),
              'scores': array([4.99794146e+08])},
    1185: {   'boxes': array([[ 95, 423, 114, 438]]),
              'classes': array([1]),
              'scores': array([4.99613107e+08])},
    1186: {   'boxes': array([[115, 431, 152, 453]]),
              'classes': array([1]),
              'scores': array([4.99609029e+08])},
    1187: {   'boxes': array([[ 29,  96, 389, 465]]),
              'classes': array([1]),
              'scores': array([4.9963995e+08])},
    1188: {   'boxes': array([[177, 495, 194, 500]]),
              'classes': array([1]),
              'scores': array([4.99659392e+08])},
    1189: {   'boxes': array([[  1,  33, 327, 183]]),
              'classes': array([1]),
              'scores': array([5.00003073e+08])},
    1190: {   'boxes': array([[158, 233, 255, 297]]),
              'classes': array([1]),
              'scores': array([4.99846542e+08])},
    1191: {   'boxes': array([[  1, 152, 141, 194]]),
              'classes': array([1]),
              'scores': array([4.99688818e+08])},
    1192: {   'boxes': array([[ 80, 240, 233, 323]]),
              'classes': array([1]),
              'scores': array([4.99368447e+08])},
    1193: {   'boxes': array([[173, 422, 231, 474]]),
              'classes': array([1]),
              'scores': array([4.998021e+08])},
    1194: {   'boxes': array([[  3, 181, 126, 250]]),
              'classes': array([1]),
              'scores': array([5.00126695e+08])},
    1195: {   'boxes': array([[ 97, 240, 306, 326]]),
              'classes': array([1]),
              'scores': array([4.9971424e+08])},
    1196: {   'boxes': array([[103,   1, 291,  40]]),
              'classes': array([1]),
              'scores': array([4.99652643e+08])},
    1197: {   'boxes': array([[111, 279, 323, 353]]),
              'classes': array([1]),
              'scores': array([4.99793181e+08])},
    1198: {   'boxes': array([[ 99, 245, 308, 284]]),
              'classes': array([1]),
              'scores': array([4.99716685e+08])},
    1199: {   'boxes': array([[103, 205, 306, 245]]),
              'classes': array([1]),
              'scores': array([4.99934323e+08])},
    1200: {   'boxes': array([[ 99, 159, 303, 213]]),
              'classes': array([1]),
              'scores': array([4.99769362e+08])},
    1201: {   'boxes': array([[100, 120, 313, 169]]),
              'classes': array([1]),
              'scores': array([4.99716033e+08])},
    1202: {   'boxes': array([[107,  81, 278, 155]]),
              'classes': array([1]),
              'scores': array([5.00045667e+08])},
    1203: {   'boxes': array([[117, 127, 283, 177]]),
              'classes': array([1]),
              'scores': array([4.99832092e+08])},
    1204: {   'boxes': array([[126, 295, 305, 464]]),
              'classes': array([1]),
              'scores': array([4.99658205e+08])},
    1205: {   'boxes': array([[144, 377, 268, 464]]),
              'classes': array([1]),
              'scores': array([4.99683213e+08])},
    1206: {   'boxes': array([[ 11, 161, 250, 264]]),
              'classes': array([1]),
              'scores': array([4.99649004e+08])},
    1207: {   'boxes': array([[149, 188, 227, 203]]),
              'classes': array([1]),
              'scores': array([4.99742784e+08])},
    1208: {   'boxes': array([[181, 384, 206, 402]]),
              'classes': array([1]),
              'scores': array([4.99722699e+08])},
    1209: {   'boxes': array([[ 65, 220, 177, 314]]),
              'classes': array([1]),
              'scores': array([4.99734045e+08])},
    1210: {   'boxes': array([[ 41,   1, 371, 180]]),
              'classes': array([1]),
              'scores': array([5.00221145e+08])},
    1211: {   'boxes': array([[ 39, 121, 375, 365]]),
              'classes': array([1]),
              'scores': array([4.9972836e+08])},
    1212: {   'boxes': array([[ 76, 305, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99779522e+08])},
    1213: {   'boxes': array([[ 58, 148, 453, 375]]),
              'classes': array([1]),
              'scores': array([4.99227604e+08])},
    1214: {   'boxes': array([[ 74,   1, 316,  49]]),
              'classes': array([1]),
              'scores': array([4.99807397e+08])},
    1215: {   'boxes': array([[189, 282, 264, 376]]),
              'classes': array([1]),
              'scores': array([4.99742155e+08])},
    1216: {   'boxes': array([[128, 103, 323, 338]]),
              'classes': array([1]),
              'scores': array([4.99660742e+08])},
    1217: {   'boxes': array([[ 39,   1, 374, 268]]),
              'classes': array([1]),
              'scores': array([4.99708384e+08])},
    1218: {   'boxes': array([[ 46,  98, 333, 500]]),
              'classes': array([1]),
              'scores': array([4.99732272e+08])},
    1219: {   'boxes': array([[120,  17, 257, 152]]),
              'classes': array([1]),
              'scores': array([4.9982907e+08])},
    1220: {   'boxes': array([[231, 116, 493, 202]]),
              'classes': array([1]),
              'scores': array([4.99653359e+08])},
    1221: {   'boxes': array([[122, 160, 458, 257]]),
              'classes': array([1]),
              'scores': array([4.99730294e+08])},
    1222: {   'boxes': array([[  7, 135, 243, 269]]),
              'classes': array([1]),
              'scores': array([4.99730556e+08])},
    1223: {   'boxes': array([[ 60,   6, 453, 148]]),
              'classes': array([1]),
              'scores': array([4.99953802e+08])},
    1224: {   'boxes': array([[ 91, 235, 231, 300]]),
              'classes': array([1]),
              'scores': array([5.00147315e+08])},
    1225: {   'boxes': array([[152, 167, 286, 216]]),
              'classes': array([1]),
              'scores': array([4.99960027e+08])},
    1226: {   'boxes': array([[164, 242, 288, 297]]),
              'classes': array([1]),
              'scores': array([4.99830564e+08])},
    1227: {   'boxes': array([[164, 233, 282, 254]]),
              'classes': array([1]),
              'scores': array([4.9999923e+08])},
    1228: {   'boxes': array([[180, 295, 283, 335]]),
              'classes': array([1]),
              'scores': array([4.99779212e+08])},
    1229: {   'boxes': array([[162, 332, 287, 373]]),
              'classes': array([1]),
              'scores': array([4.99776355e+08])},
    1230: {   'boxes': array([[176, 376, 286, 404]]),
              'classes': array([1]),
              'scores': array([4.99799123e+08])},
    1231: {   'boxes': array([[220, 214, 269, 236]]),
              'classes': array([1]),
              'scores': array([4.99751368e+08])},
    1232: {   'boxes': array([[217, 197, 272, 217]]),
              'classes': array([1]),
              'scores': array([4.99788965e+08])},
    1233: {   'boxes': array([[214, 479, 319, 500]]),
              'classes': array([1]),
              'scores': array([4.995534e+08])},
    1234: {   'boxes': array([[224, 468, 269, 484]]),
              'classes': array([1]),
              'scores': array([5.00462044e+08])},
    1235: {   'boxes': array([[223, 457, 268, 473]]),
              'classes': array([1]),
              'scores': array([4.9978325e+08])},
    1236: {   'boxes': array([[301, 197, 351, 217]]),
              'classes': array([1]),
              'scores': array([4.99828265e+08])},
    1237: {   'boxes': array([[ 54, 259, 308, 337]]),
              'classes': array([1]),
              'scores': array([4.99758155e+08])},
    1238: {   'boxes': array([[ 39,   1, 500, 206]]),
              'classes': array([1]),
              'scores': array([4.99893772e+08])},
    1239: {   'boxes': array([[  3,   1, 375, 341]]),
              'classes': array([1]),
              'scores': array([4.996546e+08])},
    1240: {   'boxes': array([[ 14, 121, 118, 186]]),
              'classes': array([1]),
              'scores': array([4.99736733e+08])},
    1241: {   'boxes': array([[  1,  86, 500, 381]]),
              'classes': array([1]),
              'scores': array([4.99265514e+08])},
    1242: {   'boxes': array([[ 54, 189,  96, 214]]),
              'classes': array([1]),
              'scores': array([4.99667971e+08])},
    1243: {   'boxes': array([[ 84, 129, 316, 352]]),
              'classes': array([1]),
              'scores': array([4.99685032e+08])},
    1244: {   'boxes': array([[167,   1, 256, 116]]),
              'classes': array([1]),
              'scores': array([4.99221736e+08])},
    1245: {   'boxes': array([[157, 343, 185, 357]]),
              'classes': array([1]),
              'scores': array([4.99763336e+08])},
    1246: {   'boxes': array([[170, 287, 182, 300]]),
              'classes': array([1]),
              'scores': array([4.99719536e+08])},
    1247: {   'boxes': array([[167, 264, 183, 277]]),
              'classes': array([1]),
              'scores': array([4.99698831e+08])},
    1248: {   'boxes': array([[ 13, 157, 343, 421]]),
              'classes': array([1]),
              'scores': array([4.99685089e+08])},
    1249: {   'boxes': array([[ 70, 146, 122, 173]]),
              'classes': array([1]),
              'scores': array([4.99690309e+08])},
    1250: {   'boxes': array([[ 64, 115, 124, 135]]),
              'classes': array([1]),
              'scores': array([4.99679117e+08])},
    1251: {   'boxes': array([[ 78,  98, 124, 125]]),
              'classes': array([1]),
              'scores': array([4.99655582e+08])},
    1252: {   'boxes': array([[ 66,  67, 127,  89]]),
              'classes': array([1]),
              'scores': array([4.99725368e+08])},
    1253: {   'boxes': array([[ 76,  38, 128,  72]]),
              'classes': array([1]),
              'scores': array([4.99706467e+08])},
    1254: {   'boxes': array([[ 76,  11, 126,  37]]),
              'classes': array([1]),
              'scores': array([4.99753657e+08])},
    1255: {   'boxes': array([[ 70, 186, 123, 202]]),
              'classes': array([1]),
              'scores': array([4.99650373e+08])},
    1256: {   'boxes': array([[ 68, 200, 115, 219]]),
              'classes': array([1]),
              'scores': array([4.99674367e+08])},
    1257: {   'boxes': array([[ 92, 380, 112, 397]]),
              'classes': array([1]),
              'scores': array([4.9966076e+08])},
    1258: {   'boxes': array([[ 88, 371, 114, 386]]),
              'classes': array([1]),
              'scores': array([4.99783688e+08])},
    1259: {   'boxes': array([[ 85, 424, 112, 442]]),
              'classes': array([1]),
              'scores': array([4.99710552e+08])},
    1260: {   'boxes': array([[ 91, 406, 113, 422]]),
              'classes': array([1]),
              'scores': array([4.99634217e+08])},
    1261: {   'boxes': array([[ 47,   2, 500, 264]]),
              'classes': array([1]),
              'scores': array([4.99794026e+08])},
    1262: {   'boxes': array([[131,  87, 500, 287]]),
              'classes': array([1]),
              'scores': array([4.99823076e+08])},
    1263: {   'boxes': array([[ 36,  66, 372, 204]]),
              'classes': array([1]),
              'scores': array([4.99834062e+08])},
    1264: {   'boxes': array([[ 56,  20, 346, 141]]),
              'classes': array([1]),
              'scores': array([4.99640789e+08])},
    1265: {   'boxes': array([[103, 148, 333, 211]]),
              'classes': array([1]),
              'scores': array([4.99728397e+08])},
    1266: {   'boxes': array([[142, 351, 175, 397]]),
              'classes': array([1]),
              'scores': array([4.99813145e+08])},
    1267: {   'boxes': array([[  8, 338, 131, 373]]),
              'classes': array([1]),
              'scores': array([4.99904734e+08])},
    1268: {   'boxes': array([[  7, 404, 132, 445]]),
              'classes': array([1]),
              'scores': array([4.99493605e+08])},
    1269: {   'boxes': array([[ 99, 373, 155, 415]]),
              'classes': array([1]),
              'scores': array([4.9951046e+08])},
    1270: {   'boxes': array([[108, 419, 167, 474]]),
              'classes': array([1]),
              'scores': array([5.00048123e+08])},
    1271: {   'boxes': array([[ 22, 377, 105, 411]]),
              'classes': array([1]),
              'scores': array([4.99832557e+08])},
    1272: {   'boxes': array([[ 29, 444, 126, 488]]),
              'classes': array([1]),
              'scores': array([4.99693571e+08])},
    1273: {   'boxes': array([[ 18, 286,  65, 320]]),
              'classes': array([1]),
              'scores': array([4.99691938e+08])},
    1274: {   'boxes': array([[ 28, 203,  56, 229]]),
              'classes': array([1]),
              'scores': array([4.99732284e+08])},
    1275: {   'boxes': array([[ 23, 179,  57, 207]]),
              'classes': array([1]),
              'scores': array([4.99575638e+08])},
    1276: {   'boxes': array([[ 27, 143,  56, 174]]),
              'classes': array([1]),
              'scores': array([4.99518585e+08])},
    1277: {   'boxes': array([[ 15, 118,  58, 136]]),
              'classes': array([1]),
              'scores': array([4.99626903e+08])},
    1278: {   'boxes': array([[ 27,  67,  67, 100]]),
              'classes': array([1]),
              'scores': array([4.9947741e+08])},
    1279: {   'boxes': array([[33, 29, 91, 61]]),
              'classes': array([1]),
              'scores': array([4.99783382e+08])},
    1280: {   'boxes': array([[ 31,  85,  61, 116]]),
              'classes': array([1]),
              'scores': array([4.99639228e+08])},
    1281: {   'boxes': array([[ 19, 316,  71, 341]]),
              'classes': array([1]),
              'scores': array([4.99646274e+08])},
    1282: {   'boxes': array([[ 19, 367, 104, 393]]),
              'classes': array([1]),
              'scores': array([4.99889506e+08])},
    1283: {   'boxes': array([[  1,   1, 374, 194]]),
              'classes': array([1]),
              'scores': array([4.99658389e+08])},
    1284: {   'boxes': array([[  1, 171, 182, 293]]),
              'classes': array([1]),
              'scores': array([4.99634025e+08])},
    1285: {   'boxes': array([[127, 166, 375, 373]]),
              'classes': array([1]),
              'scores': array([4.99762352e+08])},
    1286: {   'boxes': array([[ 73,  87, 351, 223]]),
              'classes': array([1]),
              'scores': array([4.99650746e+08])},
    1287: {   'boxes': array([[ 66, 247, 208, 346]]),
              'classes': array([1]),
              'scores': array([4.99636471e+08])},
    1288: {   'boxes': array([[ 39, 121, 231, 254]]),
              'classes': array([1]),
              'scores': array([4.99625794e+08])},
    1289: {   'boxes': array([[ 61, 337, 182, 494]]),
              'classes': array([1]),
              'scores': array([4.99685804e+08])},
    1290: {   'boxes': array([[ 81, 101, 362, 500]]),
              'classes': array([1]),
              'scores': array([4.99676553e+08])},
    1291: {   'boxes': array([[ 73, 177, 273, 317]]),
              'classes': array([1]),
              'scores': array([5.00125949e+08])},
    1292: {   'boxes': array([[ 39,  23, 500, 376]]),
              'classes': array([1]),
              'scores': array([5.00181468e+08])},
    1293: {   'boxes': array([[296, 294, 500, 375]]),
              'classes': array([1]),
              'scores': array([4.99621806e+08])},
    1294: {   'boxes': array([[ 50, 270, 118, 300]]),
              'classes': array([1]),
              'scores': array([5.00149173e+08])},
    1295: {   'boxes': array([[ 59, 245, 116, 269]]),
              'classes': array([1]),
              'scores': array([4.99811984e+08])},
    1296: {   'boxes': array([[ 68, 169, 311, 355]]),
              'classes': array([1]),
              'scores': array([4.99800523e+08])},
    1297: {   'boxes': array([[288, 273, 380, 343]]),
              'classes': array([1]),
              'scores': array([4.99677554e+08])},
    1298: {   'boxes': array([[ 10, 184, 305, 384]]),
              'classes': array([1]),
              'scores': array([4.99883112e+08])},
    1299: {   'boxes': array([[ 65,   3, 221, 354]]),
              'classes': array([1]),
              'scores': array([4.99975596e+08])},
    1300: {   'boxes': array([[ 83,  60, 213, 214]]),
              'classes': array([1]),
              'scores': array([5.00080084e+08])},
    1301: {   'boxes': array([[ 64,  43, 368, 298]]),
              'classes': array([1]),
              'scores': array([4.99449725e+08])},
    1302: {   'boxes': array([[183, 433, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99632201e+08])},
    1303: {   'boxes': array([[112, 408, 304, 497]]),
              'classes': array([1]),
              'scores': array([4.99665039e+08])},
    1304: {   'boxes': array([[107, 380, 241, 437]]),
              'classes': array([1]),
              'scores': array([4.99804488e+08])},
    1305: {   'boxes': array([[124, 335, 225, 394]]),
              'classes': array([1]),
              'scores': array([4.99752222e+08])},
    1306: {   'boxes': array([[122, 212, 181, 301]]),
              'classes': array([1]),
              'scores': array([5.00008899e+08])},
    1307: {   'boxes': array([[115, 120, 213, 167]]),
              'classes': array([1]),
              'scores': array([4.99752731e+08])},
    1308: {   'boxes': array([[114,  68, 225, 135]]),
              'classes': array([1]),
              'scores': array([4.99754723e+08])},
    1309: {   'boxes': array([[107,   1, 241, 121]]),
              'classes': array([1]),
              'scores': array([4.99883263e+08])},
    1310: {   'boxes': array([[181,   1, 362,  98]]),
              'classes': array([1]),
              'scores': array([4.99716295e+08])},
    1311: {   'boxes': array([[153, 220, 252, 250]]),
              'classes': array([1]),
              'scores': array([4.9956335e+08])},
    1312: {   'boxes': array([[128, 194, 166, 214]]),
              'classes': array([1]),
              'scores': array([4.99710773e+08])},
    1313: {   'boxes': array([[130, 160, 167, 188]]),
              'classes': array([1]),
              'scores': array([4.99897226e+08])},
    1314: {   'boxes': array([[ 85,  13, 500, 301]]),
              'classes': array([1]),
              'scores': array([4.99711299e+08])},
    1315: {   'boxes': array([[114, 316, 160, 334]]),
              'classes': array([1]),
              'scores': array([4.99847521e+08])},
    1316: {   'boxes': array([[ 32, 103, 320, 291]]),
              'classes': array([1]),
              'scores': array([4.99499931e+08])},
    1317: {   'boxes': array([[ 28,  88, 318, 356]]),
              'classes': array([1]),
              'scores': array([4.99600718e+08])},
    1318: {   'boxes': array([[  9,  89, 499, 375]]),
              'classes': array([1]),
              'scores': array([4.99534782e+08])},
    1319: {   'boxes': array([[ 56, 160, 215, 277]]),
              'classes': array([1]),
              'scores': array([4.9978738e+08])},
    1320: {   'boxes': array([[ 63,  32, 256, 108]]),
              'classes': array([1]),
              'scores': array([4.99708652e+08])},
    1321: {   'boxes': array([[102,  75, 266, 130]]),
              'classes': array([1]),
              'scores': array([4.99747143e+08])},
    1322: {   'boxes': array([[111,  15, 290, 105]]),
              'classes': array([1]),
              'scores': array([4.99522376e+08])},
    1323: {   'boxes': array([[103,   1, 375, 328]]),
              'classes': array([1]),
              'scores': array([5.00250794e+08])},
    1324: {   'boxes': array([[ 42, 302, 375, 500]]),
              'classes': array([1]),
              'scores': array([5.00483861e+08])},
    1325: {   'boxes': array([[ 52, 479, 184, 500]]),
              'classes': array([1]),
              'scores': array([4.99597897e+08])},
    1326: {   'boxes': array([[ 23, 471,  52, 496]]),
              'classes': array([1]),
              'scores': array([4.99360345e+08])},
    1327: {   'boxes': array([[135,   3, 321, 286]]),
              'classes': array([1]),
              'scores': array([4.99230607e+08])},
    1328: {   'boxes': array([[ 96, 288, 369, 415]]),
              'classes': array([1]),
              'scores': array([4.99777559e+08])},
    1329: {   'boxes': array([[120, 213, 285, 330]]),
              'classes': array([1]),
              'scores': array([4.99687254e+08])},
    1330: {   'boxes': array([[ 93, 169, 281, 279]]),
              'classes': array([1]),
              'scores': array([4.99305069e+08])},
    1331: {   'boxes': array([[  1, 238, 318, 472]]),
              'classes': array([1]),
              'scores': array([4.99782217e+08])},
    1332: {   'boxes': array([[131,  39, 185,  69]]),
              'classes': array([1]),
              'scores': array([4.99754335e+08])},
    1333: {   'boxes': array([[205, 121, 237, 192]]),
              'classes': array([1]),
              'scores': array([4.99349723e+08])},
    1334: {   'boxes': array([[ 82, 308, 368, 426]]),
              'classes': array([1]),
              'scores': array([5.00057207e+08])},
    1335: {   'boxes': array([[184, 390, 246, 414]]),
              'classes': array([1]),
              'scores': array([4.99902078e+08])},
    1336: {   'boxes': array([[177, 422, 247, 443]]),
              'classes': array([1]),
              'scores': array([4.99833325e+08])},
    1337: {   'boxes': array([[181, 479, 244, 500]]),
              'classes': array([1]),
              'scores': array([4.99804489e+08])},
    1338: {   'boxes': array([[193, 357, 242, 375]]),
              'classes': array([1]),
              'scores': array([4.99826657e+08])},
    1339: {   'boxes': array([[  5,  89, 333, 459]]),
              'classes': array([1]),
              'scores': array([4.99767864e+08])},
    1340: {   'boxes': array([[178, 142, 232, 183]]),
              'classes': array([1]),
              'scores': array([4.99622081e+08])},
    1341: {   'boxes': array([[ 79, 340, 228, 410]]),
              'classes': array([1]),
              'scores': array([4.99442581e+08])},
    1342: {   'boxes': array([[ 32,   1, 295,  72]]),
              'classes': array([1]),
              'scores': array([4.99949857e+08])},
    1343: {   'boxes': array([[ 19,  15, 426, 316]]),
              'classes': array([1]),
              'scores': array([5.00005879e+08])},
    1344: {   'boxes': array([[111, 194, 371, 305]]),
              'classes': array([1]),
              'scores': array([4.99736519e+08])},
    1345: {   'boxes': array([[288, 258, 500, 375]]),
              'classes': array([1]),
              'scores': array([4.99958017e+08])},
    1346: {   'boxes': array([[261,  40, 500, 284]]),
              'classes': array([1]),
              'scores': array([4.99862804e+08])},
    1347: {   'boxes': array([[142, 196, 185, 211]]),
              'classes': array([1]),
              'scores': array([4.99657211e+08])},
    1348: {   'boxes': array([[142,  19, 179,  32]]),
              'classes': array([1]),
              'scores': array([4.99758618e+08])},
    1349: {   'boxes': array([[153, 132, 183, 148]]),
              'classes': array([1]),
              'scores': array([4.99717861e+08])},
    1350: {   'boxes': array([[143,  71, 180,  82]]),
              'classes': array([1]),
              'scores': array([4.99658705e+08])},
    1351: {   'boxes': array([[117,  81, 373, 205]]),
              'classes': array([1]),
              'scores': array([4.99793818e+08])},
    1352: {   'boxes': array([[125, 245, 372, 498]]),
              'classes': array([1]),
              'scores': array([4.99730103e+08])},
    1353: {   'boxes': array([[ 92, 295, 262, 496]]),
              'classes': array([1]),
              'scores': array([4.99890682e+08])},
    1354: {   'boxes': array([[305, 151, 374, 186]]),
              'classes': array([1]),
              'scores': array([4.99680112e+08])},
    1355: {   'boxes': array([[320,  87, 374, 113]]),
              'classes': array([1]),
              'scores': array([4.99706169e+08])},
    1356: {   'boxes': array([[321, 113, 374, 139]]),
              'classes': array([1]),
              'scores': array([4.9962598e+08])},
    1357: {   'boxes': array([[316, 129, 374, 156]]),
              'classes': array([1]),
              'scores': array([4.99631715e+08])},
    1358: {   'boxes': array([[ 36, 164, 333, 386]]),
              'classes': array([1]),
              'scores': array([4.99791543e+08])},
    1359: {   'boxes': array([[259,  54, 318, 121]]),
              'classes': array([1]),
              'scores': array([4.99651603e+08])},
    1360: {   'boxes': array([[128, 254, 323, 362]]),
              'classes': array([1]),
              'scores': array([4.99874964e+08])},
    1361: {   'boxes': array([[151,  84, 220, 121]]),
              'classes': array([1]),
              'scores': array([4.99925512e+08])},
    1362: {   'boxes': array([[138, 141, 246, 225]]),
              'classes': array([1]),
              'scores': array([4.99908461e+08])},
    1363: {   'boxes': array([[167, 331, 285, 382]]),
              'classes': array([1]),
              'scores': array([4.99676787e+08])},
    1364: {   'boxes': array([[181, 440, 285, 468]]),
              'classes': array([1]),
              'scores': array([4.99788031e+08])},
    1365: {   'boxes': array([[125,  45, 461, 314]]),
              'classes': array([1]),
              'scores': array([4.99611743e+08])},
    1366: {   'boxes': array([[152,   1, 266, 227]]),
              'classes': array([1]),
              'scores': array([4.99733384e+08])},
    1367: {   'boxes': array([[ 64, 416, 293, 500]]),
              'classes': array([1]),
              'scores': array([4.99726936e+08])},
    1368: {   'boxes': array([[105, 198, 333, 307]]),
              'classes': array([1]),
              'scores': array([4.99755431e+08])},
    1369: {   'boxes': array([[135, 155, 270, 201]]),
              'classes': array([1]),
              'scores': array([4.99729034e+08])},
    1370: {   'boxes': array([[230, 425, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99652885e+08])},
    1371: {   'boxes': array([[131, 198, 368, 350]]),
              'classes': array([1]),
              'scores': array([4.99771507e+08])},
    1372: {   'boxes': array([[138, 146, 278, 262]]),
              'classes': array([1]),
              'scores': array([5.00107524e+08])},
    1373: {   'boxes': array([[124,   1, 206, 105]]),
              'classes': array([1]),
              'scores': array([4.99740947e+08])},
    1374: {   'boxes': array([[ 43,  75, 496, 206]]),
              'classes': array([1]),
              'scores': array([4.99848708e+08])},
    1375: {   'boxes': array([[108, 167, 215, 287]]),
              'classes': array([1]),
              'scores': array([4.99615645e+08])},
    1376: {   'boxes': array([[143,   7, 273,  61]]),
              'classes': array([1]),
              'scores': array([5.00026753e+08])},
    1377: {   'boxes': array([[180, 372, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99290269e+08])},
    1378: {   'boxes': array([[ 39,  86, 268, 222]]),
              'classes': array([1]),
              'scores': array([5.00174904e+08])},
    1379: {   'boxes': array([[ 74, 330, 319, 448]]),
              'classes': array([1]),
              'scores': array([4.99700503e+08])},
    1380: {   'boxes': array([[ 71, 322, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99755111e+08])},
    1381: {   'boxes': array([[ 80,  86, 375, 299]]),
              'classes': array([1]),
              'scores': array([4.99869159e+08])},
    1382: {   'boxes': array([[245, 417, 284, 433]]),
              'classes': array([1]),
              'scores': array([4.99837908e+08])},
    1383: {   'boxes': array([[254, 120, 294, 135]]),
              'classes': array([1]),
              'scores': array([4.99656184e+08])},
    1384: {   'boxes': array([[252,  99, 300, 117]]),
              'classes': array([1]),
              'scores': array([4.99657071e+08])},
    1385: {   'boxes': array([[ 48,  17, 333, 211]]),
              'classes': array([1]),
              'scores': array([4.99735825e+08])},
    1386: {   'boxes': array([[ 34, 157, 333, 448]]),
              'classes': array([1]),
              'scores': array([4.99719228e+08])},
    1387: {   'boxes': array([[175, 258, 336, 334]]),
              'classes': array([1]),
              'scores': array([4.99875697e+08])},
    1388: {   'boxes': array([[170, 237, 199, 251]]),
              'classes': array([1]),
              'scores': array([4.99899274e+08])},
    1389: {   'boxes': array([[170, 182, 199, 199]]),
              'classes': array([1]),
              'scores': array([4.99658267e+08])},
    1390: {   'boxes': array([[171, 169, 198, 182]]),
              'classes': array([1]),
              'scores': array([4.99675044e+08])},
    1391: {   'boxes': array([[181, 133, 206, 146]]),
              'classes': array([1]),
              'scores': array([4.99766469e+08])},
    1392: {   'boxes': array([[ 30, 121, 107, 150]]),
              'classes': array([1]),
              'scores': array([4.99688305e+08])},
    1393: {   'boxes': array([[ 22, 148, 106, 182]]),
              'classes': array([1]),
              'scores': array([4.99831779e+08])},
    1394: {   'boxes': array([[ 18, 201,  86, 245]]),
              'classes': array([1]),
              'scores': array([4.99860902e+08])},
    1395: {   'boxes': array([[ 23, 243,  87, 288]]),
              'classes': array([1]),
              'scores': array([4.99650163e+08])},
    1396: {   'boxes': array([[ 64, 358, 112, 396]]),
              'classes': array([1]),
              'scores': array([4.99507409e+08])},
    1397: {   'boxes': array([[ 89, 282, 288, 401]]),
              'classes': array([1]),
              'scores': array([4.99799506e+08])},
    1398: {   'boxes': array([[ 85, 242, 175, 309]]),
              'classes': array([1]),
              'scores': array([4.99607054e+08])},
    1399: {   'boxes': array([[ 73,  33, 333, 210]]),
              'classes': array([1]),
              'scores': array([4.99732982e+08])},
    1400: {   'boxes': array([[ 82, 220, 177, 263]]),
              'classes': array([1]),
              'scores': array([4.99852714e+08])},
    1401: {   'boxes': array([[ 77, 197, 175, 239]]),
              'classes': array([1]),
              'scores': array([4.99767454e+08])},
    1402: {   'boxes': array([[  7,  84, 484, 301]]),
              'classes': array([1]),
              'scores': array([4.99637762e+08])},
    1403: {   'boxes': array([[318,   1, 500,  91]]),
              'classes': array([1]),
              'scores': array([4.9978907e+08])},
    1404: {   'boxes': array([[273,  67, 348, 119]]),
              'classes': array([1]),
              'scores': array([4.99779646e+08])},
    1405: {   'boxes': array([[213, 253, 269, 272]]),
              'classes': array([1]),
              'scores': array([4.99763118e+08])},
    1406: {   'boxes': array([[ 55, 223, 242, 344]]),
              'classes': array([1]),
              'scores': array([4.99776597e+08])},
    1407: {   'boxes': array([[ 49, 110, 205, 208]]),
              'classes': array([1]),
              'scores': array([4.99772209e+08])},
    1408: {   'boxes': array([[ 91,  18, 275, 111]]),
              'classes': array([1]),
              'scores': array([4.99632697e+08])},
    1409: {   'boxes': array([[145, 208, 300, 349]]),
              'classes': array([1]),
              'scores': array([5.00117083e+08])},
    1410: {   'boxes': array([[ 95, 157, 286, 242]]),
              'classes': array([1]),
              'scores': array([4.9973406e+08])},
    1411: {   'boxes': array([[ 49, 226, 237, 316]]),
              'classes': array([1]),
              'scores': array([5.00091314e+08])},
    1412: {   'boxes': array([[131, 297, 159, 330]]),
              'classes': array([1]),
              'scores': array([4.99740531e+08])},
    1413: {   'boxes': array([[ 50, 460, 146, 482]]),
              'classes': array([1]),
              'scores': array([4.99938497e+08])},
    1414: {   'boxes': array([[ 51, 451,  82, 465]]),
              'classes': array([1]),
              'scores': array([4.9983915e+08])},
    1415: {   'boxes': array([[ 59, 419, 145, 439]]),
              'classes': array([1]),
              'scores': array([5.00206484e+08])},
    1416: {   'boxes': array([[ 58, 398, 144, 421]]),
              'classes': array([1]),
              'scores': array([4.99793622e+08])},
    1417: {   'boxes': array([[ 60, 424, 198, 470]]),
              'classes': array([1]),
              'scores': array([4.99558034e+08])},
    1418: {   'boxes': array([[ 48, 337, 135, 387]]),
              'classes': array([1]),
              'scores': array([4.99837545e+08])},
    1419: {   'boxes': array([[ 64, 282, 121, 342]]),
              'classes': array([1]),
              'scores': array([4.9989177e+08])},
    1420: {   'boxes': array([[ 70, 267, 115, 295]]),
              'classes': array([1]),
              'scores': array([4.99870761e+08])},
    1421: {   'boxes': array([[ 61, 231, 121, 269]]),
              'classes': array([1]),
              'scores': array([4.99875208e+08])},
    1422: {   'boxes': array([[ 64, 198, 140, 224]]),
              'classes': array([1]),
              'scores': array([4.99653617e+08])},
    1423: {   'boxes': array([[ 59, 143, 149, 182]]),
              'classes': array([1]),
              'scores': array([4.99742704e+08])},
    1424: {   'boxes': array([[ 67, 122, 151, 153]]),
              'classes': array([1]),
              'scores': array([4.99756666e+08])},
    1425: {   'boxes': array([[ 60,  47, 174,  86]]),
              'classes': array([1]),
              'scores': array([4.99714616e+08])},
    1426: {   'boxes': array([[ 69,  19, 183,  58]]),
              'classes': array([1]),
              'scores': array([4.99689662e+08])},
    1427: {   'boxes': array([[ 65,   2, 181,  16]]),
              'classes': array([1]),
              'scores': array([4.99623997e+08])},
    1428: {   'boxes': array([[  1,   1, 351, 397]]),
              'classes': array([1]),
              'scores': array([4.99751207e+08])},
    1429: {   'boxes': array([[ 20,  40, 298, 262]]),
              'classes': array([1]),
              'scores': array([4.99927006e+08])},
    1430: {   'boxes': array([[283, 167, 325, 176]]),
              'classes': array([1]),
              'scores': array([4.9968591e+08])},
    1431: {   'boxes': array([[288, 107, 313, 116]]),
              'classes': array([1]),
              'scores': array([4.99655864e+08])},
    1432: {   'boxes': array([[291,  98, 326, 109]]),
              'classes': array([1]),
              'scores': array([4.99664659e+08])},
    1433: {   'boxes': array([[285,  84, 335, 100]]),
              'classes': array([1]),
              'scores': array([4.99790333e+08])},
    1434: {   'boxes': array([[286, 426, 339, 446]]),
              'classes': array([1]),
              'scores': array([4.99679233e+08])},
    1435: {   'boxes': array([[287,  36, 315,  43]]),
              'classes': array([1]),
              'scores': array([4.99695233e+08])},
    1436: {   'boxes': array([[  1,   1, 352, 245]]),
              'classes': array([1]),
              'scores': array([4.99242892e+08])},
    1437: {   'boxes': array([[101, 179, 311, 246]]),
              'classes': array([1]),
              'scores': array([4.99750735e+08])},
    1438: {   'boxes': array([[176, 153, 242, 179]]),
              'classes': array([1]),
              'scores': array([4.9973345e+08])},
    1439: {   'boxes': array([[ 42, 237, 130, 380]]),
              'classes': array([1]),
              'scores': array([4.99844752e+08])},
    1440: {   'boxes': array([[ 85,   3, 375, 201]]),
              'classes': array([1]),
              'scores': array([4.99796028e+08])},
    1441: {   'boxes': array([[108, 181, 375, 341]]),
              'classes': array([1]),
              'scores': array([4.99737974e+08])},
    1442: {   'boxes': array([[148, 275, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99902426e+08])},
    1443: {   'boxes': array([[ 57, 407, 174, 453]]),
              'classes': array([1]),
              'scores': array([4.99566383e+08])},
    1444: {   'boxes': array([[102, 180, 207, 281]]),
              'classes': array([1]),
              'scores': array([4.99789906e+08])},
    1445: {   'boxes': array([[105,   2, 320, 191]]),
              'classes': array([1]),
              'scores': array([4.99852927e+08])},
    1446: {   'boxes': array([[103, 107, 320, 294]]),
              'classes': array([1]),
              'scores': array([4.99798818e+08])},
    1447: {   'boxes': array([[108, 225, 318, 349]]),
              'classes': array([1]),
              'scores': array([4.99853696e+08])},
    1448: {   'boxes': array([[ 43, 313, 320, 463]]),
              'classes': array([1]),
              'scores': array([4.99729004e+08])},
    1449: {   'boxes': array([[216, 323, 239, 334]]),
              'classes': array([1]),
              'scores': array([4.99704651e+08])},
    1450: {   'boxes': array([[353, 268, 375, 296]]),
              'classes': array([1]),
              'scores': array([4.99709262e+08])},
    1451: {   'boxes': array([[  3, 274, 324, 472]]),
              'classes': array([1]),
              'scores': array([4.99635289e+08])},
    1452: {   'boxes': array([[ 42, 373, 293, 500]]),
              'classes': array([1]),
              'scores': array([4.99657082e+08])},
    1453: {   'boxes': array([[220, 461, 378, 500]]),
              'classes': array([1]),
              'scores': array([4.99721884e+08])},
    1454: {   'boxes': array([[ 14, 212, 174, 327]]),
              'classes': array([1]),
              'scores': array([4.99756312e+08])},
    1455: {   'boxes': array([[ 34, 142, 172, 210]]),
              'classes': array([1]),
              'scores': array([4.99862558e+08])},
    1456: {   'boxes': array([[ 55,  92, 170, 156]]),
              'classes': array([1]),
              'scores': array([4.99721089e+08])},
    1457: {   'boxes': array([[ 53,   5, 368, 227]]),
              'classes': array([1]),
              'scores': array([4.9983074e+08])},
    1458: {   'boxes': array([[102, 320, 319, 500]]),
              'classes': array([1]),
              'scores': array([4.99919936e+08])},
    1459: {   'boxes': array([[105, 426, 134, 462]]),
              'classes': array([1]),
              'scores': array([4.99766005e+08])},
    1460: {   'boxes': array([[ 83,  95, 319, 309]]),
              'classes': array([1]),
              'scores': array([4.99708268e+08])},
    1461: {   'boxes': array([[169,  89, 190, 112]]),
              'classes': array([1]),
              'scores': array([4.99985635e+08])},
    1462: {   'boxes': array([[241, 442, 294, 478]]),
              'classes': array([1]),
              'scores': array([4.99968402e+08])},
    1463: {   'boxes': array([[ 32, 215, 231, 367]]),
              'classes': array([1]),
              'scores': array([4.99769649e+08])},
    1464: {   'boxes': array([[  1, 396, 316, 469]]),
              'classes': array([1]),
              'scores': array([4.99783529e+08])},
    1465: {   'boxes': array([[  1,  65, 143, 167]]),
              'classes': array([1]),
              'scores': array([4.99682903e+08])},
    1466: {   'boxes': array([[  1,   1, 169,  63]]),
              'classes': array([1]),
              'scores': array([4.99781915e+08])},
    1467: {   'boxes': array([[ 21, 199, 344, 336]]),
              'classes': array([1]),
              'scores': array([4.99763172e+08])},
    1468: {   'boxes': array([[ 27,  64, 375, 478]]),
              'classes': array([1]),
              'scores': array([5.0006827e+08])},
    1469: {   'boxes': array([[ 56,  46, 303, 132]]),
              'classes': array([1]),
              'scores': array([4.99665127e+08])},
    1470: {   'boxes': array([[136, 364, 236, 440]]),
              'classes': array([1]),
              'scores': array([4.99708574e+08])},
    1471: {   'boxes': array([[ 79,  34, 157,  55]]),
              'classes': array([1]),
              'scores': array([5.00149832e+08])},
    1472: {   'boxes': array([[ 80,   1, 161,  34]]),
              'classes': array([1]),
              'scores': array([4.99814472e+08])},
    1473: {   'boxes': array([[ 60,   1, 500, 195]]),
              'classes': array([1]),
              'scores': array([4.99674726e+08])},
    1474: {   'boxes': array([[ 63, 125, 331, 493]]),
              'classes': array([1]),
              'scores': array([4.99845537e+08])},
    1475: {   'boxes': array([[ 50,  43, 379, 208]]),
              'classes': array([1]),
              'scores': array([4.99628457e+08])},
    1476: {   'boxes': array([[172, 349, 238, 372]]),
              'classes': array([1]),
              'scores': array([5.00207028e+08])},
    1477: {   'boxes': array([[ 91, 168, 357, 246]]),
              'classes': array([1]),
              'scores': array([4.99292304e+08])},
    1478: {   'boxes': array([[210, 165, 373, 231]]),
              'classes': array([1]),
              'scores': array([4.99931465e+08])},
    1479: {   'boxes': array([[215, 316, 375, 398]]),
              'classes': array([1]),
              'scores': array([4.99848259e+08])},
    1480: {   'boxes': array([[220, 425, 368, 475]]),
              'classes': array([1]),
              'scores': array([4.99665057e+08])},
    1481: {   'boxes': array([[151, 133, 240, 196]]),
              'classes': array([1]),
              'scores': array([4.99812991e+08])},
    1482: {   'boxes': array([[ 85, 136, 333, 242]]),
              'classes': array([1]),
              'scores': array([4.99742056e+08])},
    1483: {   'boxes': array([[142, 209, 326, 309]]),
              'classes': array([1]),
              'scores': array([4.99789915e+08])},
    1484: {   'boxes': array([[131, 220, 184, 270]]),
              'classes': array([1]),
              'scores': array([4.99943368e+08])},
    1485: {   'boxes': array([[112, 244, 144, 295]]),
              'classes': array([1]),
              'scores': array([4.99588636e+08])},
    1486: {   'boxes': array([[ 82, 135, 270, 256]]),
              'classes': array([1]),
              'scores': array([4.99781633e+08])},
    1487: {   'boxes': array([[169, 321, 332, 388]]),
              'classes': array([1]),
              'scores': array([4.99720887e+08])},
    1488: {   'boxes': array([[130, 395, 171, 493]]),
              'classes': array([1]),
              'scores': array([4.99860396e+08])},
    1489: {   'boxes': array([[184,  31, 232,  51]]),
              'classes': array([1]),
              'scores': array([4.99398028e+08])},
    1490: {   'boxes': array([[184,  21, 230,  33]]),
              'classes': array([1]),
              'scores': array([4.99572192e+08])},
    1491: {   'boxes': array([[144, 223, 378, 488]]),
              'classes': array([1]),
              'scores': array([4.99864236e+08])},
    1492: {   'boxes': array([[ 18,   1, 378, 188]]),
              'classes': array([1]),
              'scores': array([4.99734908e+08])},
    1493: {   'boxes': array([[114, 150, 262, 296]]),
              'classes': array([1]),
              'scores': array([4.9969664e+08])},
    1494: {   'boxes': array([[ 50, 454, 149, 500]]),
              'classes': array([1]),
              'scores': array([4.99653036e+08])},
    1495: {   'boxes': array([[ 89, 122, 308, 359]]),
              'classes': array([1]),
              'scores': array([4.99707877e+08])},
    1496: {   'boxes': array([[ 82, 291, 357, 494]]),
              'classes': array([1]),
              'scores': array([4.9972593e+08])},
    1497: {   'boxes': array([[238, 387, 311, 500]]),
              'classes': array([1]),
              'scores': array([4.99752309e+08])},
    1498: {   'boxes': array([[ 67, 443, 181, 478]]),
              'classes': array([1]),
              'scores': array([4.99771283e+08])},
    1499: {   'boxes': array([[ 76, 475, 116, 500]]),
              'classes': array([1]),
              'scores': array([4.99733593e+08])},
    1500: {   'boxes': array([[171, 114, 375, 249]]),
              'classes': array([1]),
              'scores': array([4.9972486e+08])},
    1501: {   'boxes': array([[172,  19, 274, 111]]),
              'classes': array([1]),
              'scores': array([4.99662525e+08])},
    1502: {   'boxes': array([[258, 320, 281, 337]]),
              'classes': array([1]),
              'scores': array([4.99726557e+08])},
    1503: {   'boxes': array([[261, 295, 314, 323]]),
              'classes': array([1]),
              'scores': array([4.99880137e+08])},
    1504: {   'boxes': array([[ 52, 201, 202, 258]]),
              'classes': array([1]),
              'scores': array([4.99772384e+08])},
    1505: {   'boxes': array([[146, 311, 161, 320]]),
              'classes': array([1]),
              'scores': array([4.99639626e+08])},
    1506: {   'boxes': array([[149,  96, 170, 107]]),
              'classes': array([1]),
              'scores': array([4.9966219e+08])},
    1507: {   'boxes': array([[248, 101, 295, 121]]),
              'classes': array([1]),
              'scores': array([4.99647044e+08])},
    1508: {   'boxes': array([[193, 205, 236, 228]]),
              'classes': array([1]),
              'scores': array([4.99709783e+08])},
    1509: {   'boxes': array([[ 50, 387, 187, 500]]),
              'classes': array([1]),
              'scores': array([4.99564382e+08])},
    1510: {   'boxes': array([[174, 259, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99702164e+08])},
    1511: {   'boxes': array([[149,   1, 375, 238]]),
              'classes': array([1]),
              'scores': array([4.99694709e+08])},
    1512: {   'boxes': array([[103,   1, 170,  38]]),
              'classes': array([1]),
              'scores': array([4.994096e+08])},
    1513: {   'boxes': array([[ 44,  58, 159, 157]]),
              'classes': array([1]),
              'scores': array([5.00031338e+08])},
    1514: {   'boxes': array([[ 54, 169, 135, 253]]),
              'classes': array([1]),
              'scores': array([4.99815804e+08])},
    1515: {   'boxes': array([[ 41, 234, 153, 357]]),
              'classes': array([1]),
              'scores': array([4.99679752e+08])},
    1516: {   'boxes': array([[ 85, 302, 125, 333]]),
              'classes': array([1]),
              'scores': array([4.99714949e+08])},
    1517: {   'boxes': array([[138, 454, 179, 472]]),
              'classes': array([1]),
              'scores': array([4.99606341e+08])},
    1518: {   'boxes': array([[156,   1, 233,  19]]),
              'classes': array([1]),
              'scores': array([4.99977775e+08])},
    1519: {   'boxes': array([[156, 219, 311, 264]]),
              'classes': array([1]),
              'scores': array([4.99754724e+08])},
    1520: {   'boxes': array([[162, 362, 258, 388]]),
              'classes': array([1]),
              'scores': array([5.00113266e+08])},
    1521: {   'boxes': array([[151, 304, 203, 339]]),
              'classes': array([1]),
              'scores': array([4.99700802e+08])},
    1522: {   'boxes': array([[ 33, 126, 384, 378]]),
              'classes': array([1]),
              'scores': array([4.9988245e+08])},
    1523: {   'boxes': array([[ 33,   1, 500, 333]]),
              'classes': array([1]),
              'scores': array([4.99886535e+08])},
    1524: {   'boxes': array([[  1,   1, 181,  96]]),
              'classes': array([1]),
              'scores': array([4.99554394e+08])},
    1525: {   'boxes': array([[  1, 233, 184, 331]]),
              'classes': array([1]),
              'scores': array([4.99510526e+08])},
    1526: {   'boxes': array([[ 93, 128, 461, 316]]),
              'classes': array([1]),
              'scores': array([4.99428209e+08])},
    1527: {   'boxes': array([[ 64, 274, 322, 500]]),
              'classes': array([1]),
              'scores': array([4.99592302e+08])},
    1528: {   'boxes': array([[ 64,  41, 311, 283]]),
              'classes': array([1]),
              'scores': array([4.99646912e+08])},
    1529: {   'boxes': array([[ 95,   1, 201,  34]]),
              'classes': array([1]),
              'scores': array([4.99677142e+08])},
    1530: {   'boxes': array([[113,  61, 145,  97]]),
              'classes': array([1]),
              'scores': array([4.99557399e+08])},
    1531: {   'boxes': array([[  2,   4, 485, 223]]),
              'classes': array([1]),
              'scores': array([4.99638337e+08])},
    1532: {   'boxes': array([[145, 203, 305, 279]]),
              'classes': array([1]),
              'scores': array([4.99913467e+08])},
    1533: {   'boxes': array([[242,  23, 414, 198]]),
              'classes': array([1]),
              'scores': array([4.9968012e+08])},
    1534: {   'boxes': array([[ 77,  98, 434, 332]]),
              'classes': array([1]),
              'scores': array([4.99916335e+08])},
    1535: {   'boxes': array([[ 53, 242, 211, 306]]),
              'classes': array([1]),
              'scores': array([4.99998663e+08])},
    1536: {   'boxes': array([[142, 181, 337, 373]]),
              'classes': array([1]),
              'scores': array([4.99436e+08])},
    1537: {   'boxes': array([[ 33, 160, 348, 277]]),
              'classes': array([1]),
              'scores': array([4.99755212e+08])},
    1538: {   'boxes': array([[219,  34, 259,  70]]),
              'classes': array([1]),
              'scores': array([4.99676732e+08])},
    1539: {   'boxes': array([[124, 333, 144, 356]]),
              'classes': array([1]),
              'scores': array([4.99613027e+08])},
    1540: {   'boxes': array([[127, 358, 144, 376]]),
              'classes': array([1]),
              'scores': array([4.99713745e+08])},
    1541: {   'boxes': array([[128, 376, 148, 397]]),
              'classes': array([1]),
              'scores': array([4.996583e+08])},
    1542: {   'boxes': array([[133, 398, 150, 422]]),
              'classes': array([1]),
              'scores': array([4.9966361e+08])},
    1543: {   'boxes': array([[133, 422, 152, 445]]),
              'classes': array([1]),
              'scores': array([4.99691529e+08])},
    1544: {   'boxes': array([[100, 130, 132, 153]]),
              'classes': array([1]),
              'scores': array([4.99696658e+08])},
    1545: {   'boxes': array([[ 96, 111, 131, 130]]),
              'classes': array([1]),
              'scores': array([4.99677641e+08])},
    1546: {   'boxes': array([[15,  3, 93, 76]]),
              'classes': array([1]),
              'scores': array([4.99742657e+08])},
    1547: {   'boxes': array([[  2,   1, 142, 159]]),
              'classes': array([1]),
              'scores': array([4.99854295e+08])},
    1548: {   'boxes': array([[ 18,  44, 361, 234]]),
              'classes': array([1]),
              'scores': array([4.99758503e+08])},
    1549: {   'boxes': array([[ 18, 253, 346, 500]]),
              'classes': array([1]),
              'scores': array([5.00218654e+08])},
    1550: {   'boxes': array([[179, 104, 451, 224]]),
              'classes': array([1]),
              'scores': array([4.99727093e+08])},
    1551: {   'boxes': array([[ 15,  68, 280, 179]]),
              'classes': array([1]),
              'scores': array([4.99741992e+08])},
    1552: {   'boxes': array([[138, 231, 241, 271]]),
              'classes': array([1]),
              'scores': array([4.99633806e+08])},
    1553: {   'boxes': array([[185, 334, 259, 426]]),
              'classes': array([1]),
              'scores': array([4.99584801e+08])},
    1554: {   'boxes': array([[135, 457, 249, 491]]),
              'classes': array([1]),
              'scores': array([4.9984109e+08])},
    1555: {   'boxes': array([[157, 309, 235, 329]]),
              'classes': array([1]),
              'scores': array([5.00325753e+08])},
    1556: {   'boxes': array([[158, 384, 231, 411]]),
              'classes': array([1]),
              'scores': array([4.99635464e+08])},
    1557: {   'boxes': array([[157, 408, 188, 441]]),
              'classes': array([1]),
              'scores': array([4.99560301e+08])},
    1558: {   'boxes': array([[143,  16, 233,  56]]),
              'classes': array([1]),
              'scores': array([4.99670961e+08])},
    1559: {   'boxes': array([[189,   1, 237,  22]]),
              'classes': array([1]),
              'scores': array([4.99720311e+08])},
    1560: {   'boxes': array([[ 65,   7, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99614584e+08])},
    1561: {   'boxes': array([[ 43,  85, 470, 323]]),
              'classes': array([1]),
              'scores': array([4.99612069e+08])},
    1562: {   'boxes': array([[ 79, 207, 107, 246]]),
              'classes': array([1]),
              'scores': array([4.99660982e+08])},
    1563: {   'boxes': array([[ 59,   4, 372, 280]]),
              'classes': array([1]),
              'scores': array([4.99664414e+08])},
    1564: {   'boxes': array([[107, 240, 371, 355]]),
              'classes': array([1]),
              'scores': array([4.99850768e+08])},
    1565: {   'boxes': array([[ 79, 262, 375, 433]]),
              'classes': array([1]),
              'scores': array([4.99788091e+08])},
    1566: {   'boxes': array([[ 74, 327, 372, 498]]),
              'classes': array([1]),
              'scores': array([4.99724192e+08])},
    1567: {   'boxes': array([[ 61, 235, 333, 309]]),
              'classes': array([1]),
              'scores': array([4.99707015e+08])},
    1568: {   'boxes': array([[ 68, 144, 330, 238]]),
              'classes': array([1]),
              'scores': array([4.99770491e+08])},
    1569: {   'boxes': array([[199, 251, 442, 349]]),
              'classes': array([1]),
              'scores': array([4.99699039e+08])},
    1570: {   'boxes': array([[227, 217, 434, 274]]),
              'classes': array([1]),
              'scores': array([4.99754658e+08])},
    1571: {   'boxes': array([[327, 149, 449, 201]]),
              'classes': array([1]),
              'scores': array([4.99847996e+08])},
    1572: {   'boxes': array([[309,  96, 464, 184]]),
              'classes': array([1]),
              'scores': array([4.99901617e+08])},
    1573: {   'boxes': array([[305,  29, 474, 100]]),
              'classes': array([1]),
              'scores': array([4.99720326e+08])},
    1574: {   'boxes': array([[245,   1, 448,  74]]),
              'classes': array([1]),
              'scores': array([4.99785949e+08])},
    1575: {   'boxes': array([[220,   3, 318, 156]]),
              'classes': array([1]),
              'scores': array([4.99789532e+08])},
    1576: {   'boxes': array([[232,  72, 341, 131]]),
              'classes': array([1]),
              'scores': array([4.99717383e+08])},
    1577: {   'boxes': array([[213, 126, 245, 174]]),
              'classes': array([1]),
              'scores': array([4.99817813e+08])},
    1578: {   'boxes': array([[184, 154, 237, 206]]),
              'classes': array([1]),
              'scores': array([4.99601522e+08])},
    1579: {   'boxes': array([[179, 188, 235, 256]]),
              'classes': array([1]),
              'scores': array([4.99724984e+08])},
    1580: {   'boxes': array([[  1, 400, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99837413e+08])},
    1581: {   'boxes': array([[183, 152, 297, 213]]),
              'classes': array([1]),
              'scores': array([4.99505909e+08])},
    1582: {   'boxes': array([[132,  40, 171,  54]]),
              'classes': array([1]),
              'scores': array([4.99679569e+08])},
    1583: {   'boxes': array([[132,  54, 166,  68]]),
              'classes': array([1]),
              'scores': array([4.99697134e+08])},
    1584: {   'boxes': array([[133,  68, 168,  76]]),
              'classes': array([1]),
              'scores': array([4.99666714e+08])},
    1585: {   'boxes': array([[133,  82, 155,  93]]),
              'classes': array([1]),
              'scores': array([4.99695103e+08])},
    1586: {   'boxes': array([[133,  74, 155,  83]]),
              'classes': array([1]),
              'scores': array([4.99642684e+08])},
    1587: {   'boxes': array([[122, 196, 183, 217]]),
              'classes': array([1]),
              'scores': array([4.99693936e+08])},
    1588: {   'boxes': array([[123,  13, 182,  39]]),
              'classes': array([1]),
              'scores': array([4.99931962e+08])},
    1589: {   'boxes': array([[ 13, 287, 374, 500]]),
              'classes': array([1]),
              'scores': array([4.99746794e+08])},
    1590: {   'boxes': array([[ 45, 295, 319, 377]]),
              'classes': array([1]),
              'scores': array([4.99608682e+08])},
    1591: {   'boxes': array([[121, 256, 333, 329]]),
              'classes': array([1]),
              'scores': array([4.99549189e+08])},
    1592: {   'boxes': array([[ 10, 147, 370, 263]]),
              'classes': array([1]),
              'scores': array([4.99714785e+08])},
    1593: {   'boxes': array([[ 39,  83, 500, 231]]),
              'classes': array([1]),
              'scores': array([4.99746776e+08])},
    1594: {   'boxes': array([[ 80, 271, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.9997629e+08])},
    1595: {   'boxes': array([[131,  31, 375, 238]]),
              'classes': array([1]),
              'scores': array([4.99700682e+08])},
    1596: {   'boxes': array([[369, 103, 468, 195]]),
              'classes': array([1]),
              'scores': array([5.00065221e+08])},
    1597: {   'boxes': array([[ 78,  97, 339, 397]]),
              'classes': array([1]),
              'scores': array([4.99794357e+08])},
    1598: {   'boxes': array([[ 47, 214, 341, 410]]),
              'classes': array([1]),
              'scores': array([4.9957459e+08])},
    1599: {   'boxes': array([[ 32, 210, 235, 314]]),
              'classes': array([1]),
              'scores': array([4.9966794e+08])},
    1600: {   'boxes': array([[ 16,   4, 235, 132]]),
              'classes': array([1]),
              'scores': array([4.99890457e+08])},
    1601: {   'boxes': array([[ 28,  16, 235, 217]]),
              'classes': array([1]),
              'scores': array([4.99942261e+08])},
    1602: {   'boxes': array([[101, 159, 375, 336]]),
              'classes': array([1]),
              'scores': array([5.00056673e+08])},
    1603: {   'boxes': array([[  4,   1, 375, 234]]),
              'classes': array([1]),
              'scores': array([5.00352624e+08])},
    1604: {   'boxes': array([[ 22, 118, 292, 200]]),
              'classes': array([1]),
              'scores': array([4.99678106e+08])},
    1605: {   'boxes': array([[166, 351, 375, 461]]),
              'classes': array([1]),
              'scores': array([4.99815097e+08])},
    1606: {   'boxes': array([[  2,  65, 333, 500]]),
              'classes': array([1]),
              'scores': array([4.99640312e+08])},
    1607: {   'boxes': array([[ 75,  47, 315, 271]]),
              'classes': array([1]),
              'scores': array([4.99721494e+08])},
    1608: {   'boxes': array([[  1, 192, 326, 466]]),
              'classes': array([1]),
              'scores': array([4.99703963e+08])},
    1609: {   'boxes': array([[146,   1, 331, 128]]),
              'classes': array([1]),
              'scores': array([4.99612155e+08])},
    1610: {   'boxes': array([[ 35,  12, 192, 121]]),
              'classes': array([1]),
              'scores': array([4.99818131e+08])},
    1611: {   'boxes': array([[ 22, 134, 120, 181]]),
              'classes': array([1]),
              'scores': array([4.99574359e+08])},
    1612: {   'boxes': array([[ 64, 289, 172, 374]]),
              'classes': array([1]),
              'scores': array([4.99499801e+08])},
    1613: {   'boxes': array([[ 96, 173, 199, 263]]),
              'classes': array([1]),
              'scores': array([4.99699762e+08])},
    1614: {   'boxes': array([[114, 297, 204, 375]]),
              'classes': array([1]),
              'scores': array([4.99703975e+08])},
    1615: {   'boxes': array([[ 68, 373, 174, 427]]),
              'classes': array([1]),
              'scores': array([4.99772999e+08])},
    1616: {   'boxes': array([[ 61, 332, 161, 389]]),
              'classes': array([1]),
              'scores': array([4.99686853e+08])},
    1617: {   'boxes': array([[ 44, 163, 168, 277]]),
              'classes': array([1]),
              'scores': array([4.9939289e+08])},
    1618: {   'boxes': array([[ 74, 435, 153, 493]]),
              'classes': array([1]),
              'scores': array([4.99470623e+08])},
    1619: {   'boxes': array([[ 79, 415, 142, 449]]),
              'classes': array([1]),
              'scores': array([4.99731982e+08])},
    1620: {   'boxes': array([[ 84, 470, 148, 500]]),
              'classes': array([1]),
              'scores': array([4.99530081e+08])},
    1621: {   'boxes': array([[ 72, 121, 142, 178]]),
              'classes': array([1]),
              'scores': array([4.99601348e+08])},
    1622: {   'boxes': array([[  1,   1, 149,  36]]),
              'classes': array([1]),
              'scores': array([4.99406705e+08])},
    1623: {   'boxes': array([[125, 231, 192, 254]]),
              'classes': array([1]),
              'scores': array([4.99960551e+08])},
    1624: {   'boxes': array([[199,   3, 500, 240]]),
              'classes': array([1]),
              'scores': array([4.99199167e+08])},
    1625: {   'boxes': array([[361, 202, 471, 251]]),
              'classes': array([1]),
              'scores': array([4.99595322e+08])},
    1626: {   'boxes': array([[158, 128, 334, 322]]),
              'classes': array([1]),
              'scores': array([4.99749517e+08])},
    1627: {   'boxes': array([[ 18, 105, 335, 407]]),
              'classes': array([1]),
              'scores': array([4.99880935e+08])},
    1628: {   'boxes': array([[177, 199, 308, 259]]),
              'classes': array([1]),
              'scores': array([4.99802948e+08])},
    1629: {   'boxes': array([[ 92,  68, 360, 277]]),
              'classes': array([1]),
              'scores': array([5.00126861e+08])},
    1630: {   'boxes': array([[104, 121, 418, 317]]),
              'classes': array([1]),
              'scores': array([4.99991852e+08])},
    1631: {   'boxes': array([[139, 199, 375, 349]]),
              'classes': array([1]),
              'scores': array([4.99769279e+08])},
    1632: {   'boxes': array([[130, 344, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99811546e+08])},
    1633: {   'boxes': array([[181,  27, 341,  97]]),
              'classes': array([1]),
              'scores': array([4.99668118e+08])},
    1634: {   'boxes': array([[174,   1, 319,  32]]),
              'classes': array([1]),
              'scores': array([4.99941707e+08])},
    1635: {   'boxes': array([[ 99, 174, 234, 309]]),
              'classes': array([1]),
              'scores': array([4.99850793e+08])},
    1636: {   'boxes': array([[ 76,  29, 500, 254]]),
              'classes': array([1]),
              'scores': array([4.99672306e+08])},
    1637: {   'boxes': array([[245,   2, 453, 117]]),
              'classes': array([1]),
              'scores': array([5.00175143e+08])},
    1638: {   'boxes': array([[ 74, 291, 119, 326]]),
              'classes': array([1]),
              'scores': array([4.99845901e+08])},
    1639: {   'boxes': array([[ 57, 182, 196, 251]]),
              'classes': array([1]),
              'scores': array([4.99766655e+08])},
    1640: {   'boxes': array([[  5, 310, 167, 500]]),
              'classes': array([1]),
              'scores': array([4.99651402e+08])},
    1641: {   'boxes': array([[201, 305, 331, 429]]),
              'classes': array([1]),
              'scores': array([4.99807372e+08])},
    1642: {   'boxes': array([[ 97,  40, 307, 259]]),
              'classes': array([1]),
              'scores': array([4.99995549e+08])},
    1643: {   'boxes': array([[ 64, 158, 343, 259]]),
              'classes': array([1]),
              'scores': array([4.99923152e+08])},
    1644: {   'boxes': array([[ 75, 247, 355, 341]]),
              'classes': array([1]),
              'scores': array([4.9968667e+08])},
    1645: {   'boxes': array([[ 85,  89, 294, 220]]),
              'classes': array([1]),
              'scores': array([4.99751582e+08])},
    1646: {   'boxes': array([[ 75, 321, 372, 496]]),
              'classes': array([1]),
              'scores': array([4.99685894e+08])},
    1647: {   'boxes': array([[ 12, 132,  84, 167]]),
              'classes': array([1]),
              'scores': array([4.99750118e+08])},
    1648: {   'boxes': array([[426,  80, 500, 103]]),
              'classes': array([1]),
              'scores': array([4.99849314e+08])},
    1649: {   'boxes': array([[427,  96, 497, 117]]),
              'classes': array([1]),
              'scores': array([4.99854828e+08])},
    1650: {   'boxes': array([[102, 103, 391, 310]]),
              'classes': array([1]),
              'scores': array([4.99661708e+08])},
    1651: {   'boxes': array([[  2,   2, 500, 395]]),
              'classes': array([1]),
              'scores': array([4.99646723e+08])},
    1652: {   'boxes': array([[  9,  54, 485, 291]]),
              'classes': array([1]),
              'scores': array([4.99681055e+08])},
    1653: {   'boxes': array([[103,  79, 500, 215]]),
              'classes': array([1]),
              'scores': array([4.99745152e+08])},
    1654: {   'boxes': array([[206, 252, 248, 302]]),
              'classes': array([1]),
              'scores': array([4.99657557e+08])},
    1655: {   'boxes': array([[213, 195, 250, 234]]),
              'classes': array([1]),
              'scores': array([4.99582068e+08])},
    1656: {   'boxes': array([[183, 251, 192, 259]]),
              'classes': array([1]),
              'scores': array([4.9967014e+08])},
    1657: {   'boxes': array([[184, 239, 192, 245]]),
              'classes': array([1]),
              'scores': array([4.99646346e+08])},
    1658: {   'boxes': array([[139,  73, 289, 156]]),
              'classes': array([1]),
              'scores': array([4.99692955e+08])},
    1659: {   'boxes': array([[142, 153, 285, 237]]),
              'classes': array([1]),
              'scores': array([4.99754977e+08])},
    1660: {   'boxes': array([[127, 244, 266, 321]]),
              'classes': array([1]),
              'scores': array([4.99798998e+08])},
    1661: {   'boxes': array([[134, 352, 262, 435]]),
              'classes': array([1]),
              'scores': array([4.99633662e+08])},
    1662: {   'boxes': array([[137, 379, 296, 468]]),
              'classes': array([1]),
              'scores': array([4.99644381e+08])},
    1663: {   'boxes': array([[146, 341, 281, 397]]),
              'classes': array([1]),
              'scores': array([4.99666526e+08])},
    1664: {   'boxes': array([[ 19,  89, 494, 323]]),
              'classes': array([1]),
              'scores': array([4.99584281e+08])},
    1665: {   'boxes': array([[ 15, 196, 481, 338]]),
              'classes': array([1]),
              'scores': array([4.99744195e+08])},
    1666: {   'boxes': array([[157, 321, 358, 403]]),
              'classes': array([1]),
              'scores': array([4.99751812e+08])},
    1667: {   'boxes': array([[ 65,  69, 375, 330]]),
              'classes': array([1]),
              'scores': array([4.99938031e+08])},
    1668: {   'boxes': array([[ 91, 209, 211, 286]]),
              'classes': array([1]),
              'scores': array([4.99289121e+08])},
    1669: {   'boxes': array([[ 31, 148, 319, 290]]),
              'classes': array([1]),
              'scores': array([4.99665062e+08])},
    1670: {   'boxes': array([[193, 136, 297, 180]]),
              'classes': array([1]),
              'scores': array([4.99855029e+08])},
    1671: {   'boxes': array([[ 32,  60, 362, 262]]),
              'classes': array([1]),
              'scores': array([4.9977486e+08])},
    1672: {   'boxes': array([[ 15, 185, 374, 401]]),
              'classes': array([1]),
              'scores': array([4.99688664e+08])},
    1673: {   'boxes': array([[131,   1, 500, 274]]),
              'classes': array([1]),
              'scores': array([4.99653142e+08])},
    1674: {   'boxes': array([[ 73, 125, 333, 391]]),
              'classes': array([1]),
              'scores': array([4.99997869e+08])},
    1675: {   'boxes': array([[206, 362, 333, 472]]),
              'classes': array([1]),
              'scores': array([4.99683127e+08])},
    1676: {   'boxes': array([[ 50, 210, 333, 482]]),
              'classes': array([1]),
              'scores': array([4.99885409e+08])},
    1677: {   'boxes': array([[170, 468, 333, 500]]),
              'classes': array([1]),
              'scores': array([4.99682378e+08])},
    1678: {   'boxes': array([[146, 438, 241, 489]]),
              'classes': array([1]),
              'scores': array([4.99775313e+08])},
    1679: {   'boxes': array([[ 78,  33, 375, 192]]),
              'classes': array([1]),
              'scores': array([4.99954158e+08])},
    1680: {   'boxes': array([[ 85, 176, 375, 352]]),
              'classes': array([1]),
              'scores': array([4.99749671e+08])},
    1681: {   'boxes': array([[ 68, 309, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99998896e+08])},
    1682: {   'boxes': array([[  1, 402, 158, 500]]),
              'classes': array([1]),
              'scores': array([4.99876472e+08])},
    1683: {   'boxes': array([[  2, 308, 125, 395]]),
              'classes': array([1]),
              'scores': array([4.99657023e+08])},
    1684: {   'boxes': array([[  1,   1, 119,  47]]),
              'classes': array([1]),
              'scores': array([4.99927413e+08])},
    1685: {   'boxes': array([[ 1, 43, 92, 87]]),
              'classes': array([1]),
              'scores': array([4.99725957e+08])},
    1686: {   'boxes': array([[  1, 190, 121, 277]]),
              'classes': array([1]),
              'scores': array([4.99740763e+08])},
    1687: {   'boxes': array([[ 63,   1, 374, 328]]),
              'classes': array([1]),
              'scores': array([5.00027074e+08])},
    1688: {   'boxes': array([[177, 362, 291, 500]]),
              'classes': array([1]),
              'scores': array([4.99602666e+08])},
    1689: {   'boxes': array([[122, 441, 247, 500]]),
              'classes': array([1]),
              'scores': array([4.99545733e+08])},
    1690: {   'boxes': array([[163, 354, 256, 411]]),
              'classes': array([1]),
              'scores': array([4.99978792e+08])},
    1691: {   'boxes': array([[169, 318, 220, 363]]),
              'classes': array([1]),
              'scores': array([4.99981276e+08])},
    1692: {   'boxes': array([[163, 253, 234, 324]]),
              'classes': array([1]),
              'scores': array([4.99851108e+08])},
    1693: {   'boxes': array([[144,   1, 245,  19]]),
              'classes': array([1]),
              'scores': array([4.99746841e+08])},
    1694: {   'boxes': array([[169,  95, 214, 114]]),
              'classes': array([1]),
              'scores': array([5.00059602e+08])},
    1695: {   'boxes': array([[168,  72, 219,  88]]),
              'classes': array([1]),
              'scores': array([5.00032329e+08])},
    1696: {   'boxes': array([[170,  53, 219,  71]]),
              'classes': array([1]),
              'scores': array([5.0020222e+08])},
    1697: {   'boxes': array([[ 86,  64, 317, 184]]),
              'classes': array([1]),
              'scores': array([4.99697791e+08])},
    1698: {   'boxes': array([[ 72, 160, 284, 300]]),
              'classes': array([1]),
              'scores': array([4.99746433e+08])},
    1699: {   'boxes': array([[  1, 270, 366, 483]]),
              'classes': array([1]),
              'scores': array([4.9951563e+08])},
    1700: {   'boxes': array([[ 76, 205, 375, 445]]),
              'classes': array([1]),
              'scores': array([4.99951208e+08])},
    1701: {   'boxes': array([[119,   1, 374, 144]]),
              'classes': array([1]),
              'scores': array([5.00023427e+08])},
    1702: {   'boxes': array([[122, 387, 220, 454]]),
              'classes': array([1]),
              'scores': array([4.99697808e+08])},
    1703: {   'boxes': array([[ 26, 277, 335, 500]]),
              'classes': array([1]),
              'scores': array([4.99737787e+08])},
    1704: {   'boxes': array([[ 76,  16, 335, 343]]),
              'classes': array([1]),
              'scores': array([4.99767144e+08])},
    1705: {   'boxes': array([[ 27,  17, 375, 351]]),
              'classes': array([1]),
              'scores': array([4.99818648e+08])},
    1706: {   'boxes': array([[ 39, 315, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99650109e+08])},
    1707: {   'boxes': array([[ 34,  54, 248, 139]]),
              'classes': array([1]),
              'scores': array([4.99820333e+08])},
    1708: {   'boxes': array([[ 71, 194, 340, 463]]),
              'classes': array([1]),
              'scores': array([4.99651879e+08])},
    1709: {   'boxes': array([[ 44, 108, 188, 257]]),
              'classes': array([1]),
              'scores': array([4.99725989e+08])},
    1710: {   'boxes': array([[ 53,   1, 286, 132]]),
              'classes': array([1]),
              'scores': array([4.99673329e+08])},
    1711: {   'boxes': array([[ 84, 330, 316, 448]]),
              'classes': array([1]),
              'scores': array([4.99700809e+08])},
    1712: {   'boxes': array([[ 97,  52, 294, 191]]),
              'classes': array([1]),
              'scores': array([5.00109415e+08])},
    1713: {   'boxes': array([[ 63,   1, 298,  41]]),
              'classes': array([1]),
              'scores': array([4.99766638e+08])},
    1714: {   'boxes': array([[ 11, 115, 223, 353]]),
              'classes': array([1]),
              'scores': array([4.99991924e+08])},
    1715: {   'boxes': array([[121,  19, 259,  75]]),
              'classes': array([1]),
              'scores': array([4.99390042e+08])},
    1716: {   'boxes': array([[143, 109, 222, 179]]),
              'classes': array([1]),
              'scores': array([4.99756026e+08])},
    1717: {   'boxes': array([[166, 146, 263, 190]]),
              'classes': array([1]),
              'scores': array([5.00182073e+08])},
    1718: {   'boxes': array([[167, 176, 262, 219]]),
              'classes': array([1]),
              'scores': array([4.99656313e+08])},
    1719: {   'boxes': array([[171, 206, 258, 241]]),
              'classes': array([1]),
              'scores': array([4.99740271e+08])},
    1720: {   'boxes': array([[143, 195, 220, 268]]),
              'classes': array([1]),
              'scores': array([4.99949062e+08])},
    1721: {   'boxes': array([[100,  10, 500, 325]]),
              'classes': array([1]),
              'scores': array([5.0014329e+08])},
    1722: {   'boxes': array([[ 32,   2, 333, 270]]),
              'classes': array([1]),
              'scores': array([5.00077027e+08])},
    1723: {   'boxes': array([[ 3,  1, 79, 65]]),
              'classes': array([1]),
              'scores': array([4.99460725e+08])},
    1724: {   'boxes': array([[169, 231, 279, 316]]),
              'classes': array([1]),
              'scores': array([4.99794916e+08])},
    1725: {   'boxes': array([[170, 312, 287, 419]]),
              'classes': array([1]),
              'scores': array([4.99653744e+08])},
    1726: {   'boxes': array([[ 73, 105, 375, 231]]),
              'classes': array([1]),
              'scores': array([4.99661602e+08])},
    1727: {   'boxes': array([[ 66,  35, 375, 118]]),
              'classes': array([1]),
              'scores': array([4.99836181e+08])},
    1728: {   'boxes': array([[  4, 257, 375, 435]]),
              'classes': array([1]),
              'scores': array([4.99590285e+08])},
    1729: {   'boxes': array([[ 54, 163, 284, 289]]),
              'classes': array([1]),
              'scores': array([4.99833657e+08])},
    1730: {   'boxes': array([[ 76,  28, 376, 306]]),
              'classes': array([1]),
              'scores': array([4.99863204e+08])},
    1731: {   'boxes': array([[ 51,  97, 266, 223]]),
              'classes': array([1]),
              'scores': array([4.99675613e+08])},
    1732: {   'boxes': array([[  1, 279, 138, 497]]),
              'classes': array([1]),
              'scores': array([4.99605777e+08])},
    1733: {   'boxes': array([[ 54,   1, 290,  72]]),
              'classes': array([1]),
              'scores': array([4.99440887e+08])},
    1734: {   'boxes': array([[ 93, 430, 241, 482]]),
              'classes': array([1]),
              'scores': array([4.9964599e+08])},
    1735: {   'boxes': array([[ 95, 232, 224, 280]]),
              'classes': array([1]),
              'scores': array([4.99579752e+08])},
    1736: {   'boxes': array([[100, 279, 212, 321]]),
              'classes': array([1]),
              'scores': array([4.99587075e+08])},
    1737: {   'boxes': array([[ 91,  83, 168, 115]]),
              'classes': array([1]),
              'scores': array([4.99677063e+08])},
    1738: {   'boxes': array([[ 88, 304, 152, 338]]),
              'classes': array([1]),
              'scores': array([5.00211692e+08])},
    1739: {   'boxes': array([[ 85, 347, 198, 378]]),
              'classes': array([1]),
              'scores': array([4.99696975e+08])},
    1740: {   'boxes': array([[ 81, 376, 191, 396]]),
              'classes': array([1]),
              'scores': array([4.9972251e+08])},
    1741: {   'boxes': array([[ 74, 455, 138, 483]]),
              'classes': array([1]),
              'scores': array([4.99894635e+08])},
    1742: {   'boxes': array([[ 82, 330, 187, 359]]),
              'classes': array([1]),
              'scores': array([5.0005616e+08])},
    1743: {   'boxes': array([[ 88,  36, 245,  80]]),
              'classes': array([1]),
              'scores': array([4.99588198e+08])},
    1744: {   'boxes': array([[ 82, 132, 157, 154]]),
              'classes': array([1]),
              'scores': array([4.99818942e+08])},
    1745: {   'boxes': array([[ 80,  11, 227,  47]]),
              'classes': array([1]),
              'scores': array([4.99864118e+08])},
    1746: {   'boxes': array([[ 81, 242, 100, 251]]),
              'classes': array([1]),
              'scores': array([4.99655774e+08])},
    1747: {   'boxes': array([[ 73, 227,  91, 238]]),
              'classes': array([1]),
              'scores': array([4.99697871e+08])},
    1748: {   'boxes': array([[ 77, 213, 116, 233]]),
              'classes': array([1]),
              'scores': array([4.99785428e+08])},
    1749: {   'boxes': array([[ 87, 199, 100, 211]]),
              'classes': array([1]),
              'scores': array([4.9965588e+08])},
    1750: {   'boxes': array([[ 82, 154, 160, 171]]),
              'classes': array([1]),
              'scores': array([4.99899761e+08])},
    1751: {   'boxes': array([[ 84, 109, 157, 135]]),
              'classes': array([1]),
              'scores': array([4.99643443e+08])},
    1752: {   'boxes': array([[ 78,  96, 102, 111]]),
              'classes': array([1]),
              'scores': array([4.99900088e+08])},
    1753: {   'boxes': array([[ 97,   1, 137,  17]]),
              'classes': array([1]),
              'scores': array([4.9976894e+08])},
    1754: {   'boxes': array([[112, 453, 175, 482]]),
              'classes': array([1]),
              'scores': array([4.99721445e+08])},
    1755: {   'boxes': array([[122, 403, 143, 414]]),
              'classes': array([1]),
              'scores': array([4.99816363e+08])},
    1756: {   'boxes': array([[119, 302, 140, 312]]),
              'classes': array([1]),
              'scores': array([4.99662119e+08])},
    1757: {   'boxes': array([[125, 293, 146, 302]]),
              'classes': array([1]),
              'scores': array([4.99683175e+08])},
    1758: {   'boxes': array([[267, 309, 340, 376]]),
              'classes': array([1]),
              'scores': array([4.99563598e+08])},
    1759: {   'boxes': array([[145,   2, 192,  49]]),
              'classes': array([1]),
              'scores': array([4.9930812e+08])},
    1760: {   'boxes': array([[ 14, 261, 332, 380]]),
              'classes': array([1]),
              'scores': array([4.99954694e+08])},
    1761: {   'boxes': array([[ 29, 158, 335, 265]]),
              'classes': array([1]),
              'scores': array([4.99854658e+08])},
    1762: {   'boxes': array([[  1,   1, 426, 500]]),
              'classes': array([1]),
              'scores': array([4.99273293e+08])},
    1763: {   'boxes': array([[ 53,  97, 298, 298]]),
              'classes': array([1]),
              'scores': array([4.99786e+08])},
    1764: {   'boxes': array([[ 58,  11, 449, 112]]),
              'classes': array([1]),
              'scores': array([4.99672381e+08])},
    1765: {   'boxes': array([[ 76, 118, 431, 201]]),
              'classes': array([1]),
              'scores': array([4.99724832e+08])},
    1766: {   'boxes': array([[ 72, 187, 425, 261]]),
              'classes': array([1]),
              'scores': array([4.99722528e+08])},
    1767: {   'boxes': array([[  1,   1, 374, 206]]),
              'classes': array([1]),
              'scores': array([4.99708049e+08])},
    1768: {   'boxes': array([[201, 291, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.9976565e+08])},
    1769: {   'boxes': array([[  1,   1, 375, 205]]),
              'classes': array([1]),
              'scores': array([4.99684899e+08])},
    1770: {   'boxes': array([[ 19, 149, 333, 287]]),
              'classes': array([1]),
              'scores': array([4.99598601e+08])},
    1771: {   'boxes': array([[ 59, 107, 333, 494]]),
              'classes': array([1]),
              'scores': array([4.99679689e+08])},
    1772: {   'boxes': array([[179, 137, 443, 334]]),
              'classes': array([1]),
              'scores': array([4.99965251e+08])},
    1773: {   'boxes': array([[ 19,   2, 500, 334]]),
              'classes': array([1]),
              'scores': array([4.99757746e+08])},
    1774: {   'boxes': array([[172,  61, 372, 233]]),
              'classes': array([1]),
              'scores': array([4.99734624e+08])},
    1775: {   'boxes': array([[ 90, 181, 202, 230]]),
              'classes': array([1]),
              'scores': array([4.99833874e+08])},
    1776: {   'boxes': array([[ 24, 312, 218, 404]]),
              'classes': array([1]),
              'scores': array([5.00167364e+08])},
    1777: {   'boxes': array([[ 80, 286, 106, 314]]),
              'classes': array([1]),
              'scores': array([5.00032016e+08])},
    1778: {   'boxes': array([[114,  18, 171,  48]]),
              'classes': array([1]),
              'scores': array([4.99767978e+08])},
    1779: {   'boxes': array([[128,  46, 201,  74]]),
              'classes': array([1]),
              'scores': array([4.99630825e+08])},
    1780: {   'boxes': array([[137, 269, 221, 305]]),
              'classes': array([1]),
              'scores': array([4.99422842e+08])},
    1781: {   'boxes': array([[ 44, 126, 240, 231]]),
              'classes': array([1]),
              'scores': array([4.99777205e+08])},
    1782: {   'boxes': array([[ 58,   2, 374, 252]]),
              'classes': array([1]),
              'scores': array([5.00132608e+08])},
    1783: {   'boxes': array([[ 58, 218, 374, 500]]),
              'classes': array([1]),
              'scores': array([5.0012115e+08])},
    1784: {   'boxes': array([[ 41,  97, 333, 271]]),
              'classes': array([1]),
              'scores': array([4.99835409e+08])},
    1785: {   'boxes': array([[175, 197, 308, 282]]),
              'classes': array([1]),
              'scores': array([5.00016792e+08])},
    1786: {   'boxes': array([[162,  62, 304, 149]]),
              'classes': array([1]),
              'scores': array([4.99758064e+08])},
    1787: {   'boxes': array([[207, 398, 251, 411]]),
              'classes': array([1]),
              'scores': array([4.99712288e+08])},
    1788: {   'boxes': array([[206, 410, 250, 425]]),
              'classes': array([1]),
              'scores': array([4.99607909e+08])},
    1789: {   'boxes': array([[143, 164, 171, 178]]),
              'classes': array([1]),
              'scores': array([4.9978783e+08])},
    1790: {   'boxes': array([[153, 178, 171, 189]]),
              'classes': array([1]),
              'scores': array([4.99661055e+08])},
    1791: {   'boxes': array([[156, 236, 170, 246]]),
              'classes': array([1]),
              'scores': array([4.99655973e+08])},
    1792: {   'boxes': array([[152, 323, 167, 337]]),
              'classes': array([1]),
              'scores': array([4.99727861e+08])},
    1793: {   'boxes': array([[  4, 340,  87, 369]]),
              'classes': array([1]),
              'scores': array([4.99689554e+08])},
    1794: {   'boxes': array([[  1, 329,  47, 357]]),
              'classes': array([1]),
              'scores': array([4.99678477e+08])},
    1795: {   'boxes': array([[  1, 283,  47, 315]]),
              'classes': array([1]),
              'scores': array([4.99870762e+08])},
    1796: {   'boxes': array([[119, 285, 146, 310]]),
              'classes': array([1]),
              'scores': array([4.99734236e+08])},
    1797: {   'boxes': array([[ 80, 333, 333, 490]]),
              'classes': array([1]),
              'scores': array([4.99833023e+08])},
    1798: {   'boxes': array([[ 83, 207, 333, 362]]),
              'classes': array([1]),
              'scores': array([4.99809276e+08])},
    1799: {   'boxes': array([[ 71,   1, 333, 185]]),
              'classes': array([1]),
              'scores': array([5.00163334e+08])},
    1800: {   'boxes': array([[159, 210, 211, 236]]),
              'classes': array([1]),
              'scores': array([4.99263752e+08])},
    1801: {   'boxes': array([[166, 180, 206, 196]]),
              'classes': array([1]),
              'scores': array([4.99810597e+08])},
    1802: {   'boxes': array([[162,  10, 220,  53]]),
              'classes': array([1]),
              'scores': array([4.99515401e+08])},
    1803: {   'boxes': array([[136, 288, 181, 323]]),
              'classes': array([1]),
              'scores': array([4.99288603e+08])},
    1804: {   'boxes': array([[  2, 291, 264, 375]]),
              'classes': array([1]),
              'scores': array([4.99683036e+08])},
    1805: {   'boxes': array([[  1, 133,  90, 195]]),
              'classes': array([1]),
              'scores': array([4.99728515e+08])},
    1806: {   'boxes': array([[  1, 212,  41, 233]]),
              'classes': array([1]),
              'scores': array([4.99781853e+08])},
    1807: {   'boxes': array([[119, 346, 153, 358]]),
              'classes': array([1]),
              'scores': array([4.99672865e+08])},
    1808: {   'boxes': array([[118, 337, 150, 348]]),
              'classes': array([1]),
              'scores': array([4.99696457e+08])},
    1809: {   'boxes': array([[153, 210, 168, 229]]),
              'classes': array([1]),
              'scores': array([4.99826864e+08])},
    1810: {   'boxes': array([[150, 254, 166, 277]]),
              'classes': array([1]),
              'scores': array([4.99683605e+08])},
    1811: {   'boxes': array([[ 59, 187, 350, 380]]),
              'classes': array([1]),
              'scores': array([4.99942447e+08])},
    1812: {   'boxes': array([[ 82, 125, 244, 216]]),
              'classes': array([1]),
              'scores': array([4.99716145e+08])},
    1813: {   'boxes': array([[184, 135, 296, 288]]),
              'classes': array([1]),
              'scores': array([4.99678126e+08])},
    1814: {   'boxes': array([[ 89,  59, 278, 119]]),
              'classes': array([1]),
              'scores': array([4.99635496e+08])},
    1815: {   'boxes': array([[  1, 352, 183, 500]]),
              'classes': array([1]),
              'scores': array([4.99726086e+08])},
    1816: {   'boxes': array([[122, 226, 276, 376]]),
              'classes': array([1]),
              'scores': array([4.99858978e+08])},
    1817: {   'boxes': array([[  4,  40, 408, 312]]),
              'classes': array([1]),
              'scores': array([4.99694982e+08])},
    1818: {   'boxes': array([[ 44, 171, 226, 294]]),
              'classes': array([1]),
              'scores': array([4.99830382e+08])},
    1819: {   'boxes': array([[174,   1, 253,  38]]),
              'classes': array([1]),
              'scores': array([4.99784751e+08])},
    1820: {   'boxes': array([[200,   3, 316,  22]]),
              'classes': array([1]),
              'scores': array([4.99865748e+08])},
    1821: {   'boxes': array([[212, 259, 310, 284]]),
              'classes': array([1]),
              'scores': array([4.99955801e+08])},
    1822: {   'boxes': array([[215, 225, 310, 252]]),
              'classes': array([1]),
              'scores': array([4.99698773e+08])},
    1823: {   'boxes': array([[  1, 176, 196, 330]]),
              'classes': array([1]),
              'scores': array([4.9980363e+08])},
    1824: {   'boxes': array([[118, 192, 375, 302]]),
              'classes': array([1]),
              'scores': array([4.99806679e+08])},
    1825: {   'boxes': array([[192, 415, 324, 500]]),
              'classes': array([1]),
              'scores': array([4.99194459e+08])},
    1826: {   'boxes': array([[ 85,   8, 375, 248]]),
              'classes': array([1]),
              'scores': array([4.99471803e+08])},
    1827: {   'boxes': array([[ 27, 245, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99762942e+08])},
    1828: {   'boxes': array([[158, 146, 500, 375]]),
              'classes': array([1]),
              'scores': array([4.99848917e+08])},
    1829: {   'boxes': array([[ 52,  26, 102,  47]]),
              'classes': array([1]),
              'scores': array([4.99837663e+08])},
    1830: {   'boxes': array([[ 73, 104, 250, 143]]),
              'classes': array([1]),
              'scores': array([4.99756062e+08])},
    1831: {   'boxes': array([[ 71,   1, 375,  61]]),
              'classes': array([1]),
              'scores': array([4.99714431e+08])},
    1832: {   'boxes': array([[ 89,  45, 168,  80]]),
              'classes': array([1]),
              'scores': array([4.99641281e+08])},
    1833: {   'boxes': array([[223, 120, 375, 187]]),
              'classes': array([1]),
              'scores': array([4.99561066e+08])},
    1834: {   'boxes': array([[136,  82, 378, 322]]),
              'classes': array([1]),
              'scores': array([4.99887641e+08])},
    1835: {   'boxes': array([[ 50, 124, 279, 371]]),
              'classes': array([1]),
              'scores': array([4.99706994e+08])},
    1836: {   'boxes': array([[ 37, 287, 164, 427]]),
              'classes': array([1]),
              'scores': array([4.99923242e+08])},
    1837: {   'boxes': array([[197, 232, 315, 334]]),
              'classes': array([1]),
              'scores': array([4.99751221e+08])},
    1838: {   'boxes': array([[ 18, 170, 208, 330]]),
              'classes': array([1]),
              'scores': array([4.9979635e+08])},
    1839: {   'boxes': array([[ 37, 341, 361, 458]]),
              'classes': array([1]),
              'scores': array([4.99769387e+08])},
    1840: {   'boxes': array([[ 75, 442, 292, 500]]),
              'classes': array([1]),
              'scores': array([4.99818834e+08])},
    1841: {   'boxes': array([[104, 202, 374, 395]]),
              'classes': array([1]),
              'scores': array([4.99789686e+08])},
    1842: {   'boxes': array([[202, 410, 221, 418]]),
              'classes': array([1]),
              'scores': array([4.99631589e+08])},
    1843: {   'boxes': array([[160,  28, 375, 181]]),
              'classes': array([1]),
              'scores': array([4.99618186e+08])},
    1844: {   'boxes': array([[ 62, 131, 440, 332]]),
              'classes': array([1]),
              'scores': array([4.99683059e+08])},
    1845: {   'boxes': array([[ 11,   1, 296, 320]]),
              'classes': array([1]),
              'scores': array([4.99554805e+08])},
    1846: {   'boxes': array([[222, 132, 280, 195]]),
              'classes': array([1]),
              'scores': array([4.99386883e+08])},
    1847: {   'boxes': array([[257,   1, 326,  61]]),
              'classes': array([1]),
              'scores': array([4.99379742e+08])},
    1848: {   'boxes': array([[218, 378, 323, 500]]),
              'classes': array([1]),
              'scores': array([4.99735447e+08])},
    1849: {   'boxes': array([[121, 248, 262, 447]]),
              'classes': array([1]),
              'scores': array([4.99640164e+08])},
    1850: {   'boxes': array([[ 85,   1, 375, 239]]),
              'classes': array([1]),
              'scores': array([4.99657643e+08])},
    1851: {   'boxes': array([[312, 102, 500, 333]]),
              'classes': array([1]),
              'scores': array([4.99819547e+08])},
    1852: {   'boxes': array([[ 82,  95, 412, 269]]),
              'classes': array([1]),
              'scores': array([4.99816917e+08])},
    1853: {   'boxes': array([[ 18, 200, 163, 323]]),
              'classes': array([1]),
              'scores': array([4.99699992e+08])},
    1854: {   'boxes': array([[ 16, 146, 395, 327]]),
              'classes': array([1]),
              'scores': array([4.99828924e+08])},
    1855: {   'boxes': array([[ 17, 255, 334, 500]]),
              'classes': array([1]),
              'scores': array([4.9968062e+08])},
    1856: {   'boxes': array([[151,  55, 334, 266]]),
              'classes': array([1]),
              'scores': array([4.99700718e+08])},
    1857: {   'boxes': array([[ 92, 122, 135, 150]]),
              'classes': array([1]),
              'scores': array([4.9974674e+08])},
    1858: {   'boxes': array([[ 58, 196, 116, 244]]),
              'classes': array([1]),
              'scores': array([4.99725393e+08])},
    1859: {   'boxes': array([[125, 183, 354, 266]]),
              'classes': array([1]),
              'scores': array([4.99724121e+08])},
    1860: {   'boxes': array([[133, 238, 276, 356]]),
              'classes': array([1]),
              'scores': array([4.99957449e+08])},
    1861: {   'boxes': array([[168, 135, 285, 235]]),
              'classes': array([1]),
              'scores': array([5.00110921e+08])},
    1862: {   'boxes': array([[107, 107, 282, 421]]),
              'classes': array([1]),
              'scores': array([4.9960899e+08])},
    1863: {   'boxes': array([[105, 210, 135, 238]]),
              'classes': array([1]),
              'scores': array([4.99672236e+08])},
    1864: {   'boxes': array([[  1, 130, 192, 262]]),
              'classes': array([1]),
              'scores': array([4.99736205e+08])},
    1865: {   'boxes': array([[  2, 306, 101, 461]]),
              'classes': array([1]),
              'scores': array([4.99657074e+08])},
    1866: {   'boxes': array([[184, 132, 320, 289]]),
              'classes': array([1]),
              'scores': array([4.999464e+08])},
    1867: {   'boxes': array([[ 15, 177, 283, 445]]),
              'classes': array([1]),
              'scores': array([4.99822597e+08])},
    1868: {   'boxes': array([[202, 353, 331, 388]]),
              'classes': array([1]),
              'scores': array([4.99905041e+08])},
    1869: {   'boxes': array([[ 26,  68, 358, 185]]),
              'classes': array([1]),
              'scores': array([4.99708825e+08])},
    1870: {   'boxes': array([[  1, 211, 308, 462]]),
              'classes': array([1]),
              'scores': array([4.99744553e+08])},
    1871: {   'boxes': array([[  2,  66, 462, 295]]),
              'classes': array([1]),
              'scores': array([4.99694617e+08])},
    1872: {   'boxes': array([[ 47,  17, 198,  86]]),
              'classes': array([1]),
              'scores': array([4.99718955e+08])},
    1873: {   'boxes': array([[ 36,  48, 127,  79]]),
              'classes': array([1]),
              'scores': array([5.00163771e+08])},
    1874: {   'boxes': array([[ 31,  79, 119, 105]]),
              'classes': array([1]),
              'scores': array([4.99733403e+08])},
    1875: {   'boxes': array([[ 49,  91,  84, 107]]),
              'classes': array([1]),
              'scores': array([4.99671982e+08])},
    1876: {   'boxes': array([[ 52,  88, 116, 128]]),
              'classes': array([1]),
              'scores': array([5.0031471e+08])},
    1877: {   'boxes': array([[ 82,   2, 194,  18]]),
              'classes': array([1]),
              'scores': array([4.99644158e+08])},
    1878: {   'boxes': array([[ 24, 272,  87, 303]]),
              'classes': array([1]),
              'scores': array([4.99662062e+08])},
    1879: {   'boxes': array([[ 64, 299, 103, 322]]),
              'classes': array([1]),
              'scores': array([4.9970582e+08])},
    1880: {   'boxes': array([[ 35, 468,  78, 493]]),
              'classes': array([1]),
              'scores': array([5.00023006e+08])},
    1881: {   'boxes': array([[ 29, 464, 159, 500]]),
              'classes': array([1]),
              'scores': array([4.99728096e+08])},
    1882: {   'boxes': array([[ 43, 450,  93, 475]]),
              'classes': array([1]),
              'scores': array([5.00028571e+08])},
    1883: {   'boxes': array([[ 30, 405, 120, 433]]),
              'classes': array([1]),
              'scores': array([4.99814347e+08])},
    1884: {   'boxes': array([[241, 267, 375, 380]]),
              'classes': array([1]),
              'scores': array([5.00153399e+08])},
    1885: {   'boxes': array([[161,  50, 374, 416]]),
              'classes': array([1]),
              'scores': array([5.00264165e+08])},
    1886: {   'boxes': array([[144,   2, 375,  70]]),
              'classes': array([1]),
              'scores': array([4.996351e+08])},
    1887: {   'boxes': array([[ 84, 134, 366, 458]]),
              'classes': array([1]),
              'scores': array([4.99728055e+08])},
    1888: {   'boxes': array([[ 96,  39, 366, 143]]),
              'classes': array([1]),
              'scores': array([4.99805953e+08])},
    1889: {   'boxes': array([[ 69, 448, 239, 500]]),
              'classes': array([1]),
              'scores': array([4.99839883e+08])},
    1890: {   'boxes': array([[ 61, 235, 333, 437]]),
              'classes': array([1]),
              'scores': array([4.99992652e+08])},
    1891: {   'boxes': array([[  5,   1, 333, 466]]),
              'classes': array([1]),
              'scores': array([4.99947745e+08])},
    1892: {   'boxes': array([[ 34,   1, 155, 105]]),
              'classes': array([1]),
              'scores': array([4.99493205e+08])},
    1893: {   'boxes': array([[ 91, 433, 146, 458]]),
              'classes': array([1]),
              'scores': array([4.99613068e+08])},
    1894: {   'boxes': array([[117, 344, 208, 390]]),
              'classes': array([1]),
              'scores': array([4.9969344e+08])},
    1895: {   'boxes': array([[100, 366, 135, 403]]),
              'classes': array([1]),
              'scores': array([4.99648241e+08])},
    1896: {   'boxes': array([[ 90,  86, 131, 117]]),
              'classes': array([1]),
              'scores': array([4.99659868e+08])},
    1897: {   'boxes': array([[106, 245, 146, 268]]),
              'classes': array([1]),
              'scores': array([4.99698644e+08])},
    1898: {   'boxes': array([[138, 265, 157, 287]]),
              'classes': array([1]),
              'scores': array([4.99686355e+08])},
    1899: {   'boxes': array([[107, 206, 165, 270]]),
              'classes': array([1]),
              'scores': array([4.99821987e+08])},
    1900: {   'boxes': array([[195, 232, 279, 310]]),
              'classes': array([1]),
              'scores': array([4.99707579e+08])},
    1901: {   'boxes': array([[ 63,  66, 333, 366]]),
              'classes': array([1]),
              'scores': array([4.99771455e+08])},
    1902: {   'boxes': array([[  1,   1, 288, 106]]),
              'classes': array([1]),
              'scores': array([4.99720661e+08])},
    1903: {   'boxes': array([[110, 330, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99842843e+08])},
    1904: {   'boxes': array([[135, 147, 375, 328]]),
              'classes': array([1]),
              'scores': array([4.99690479e+08])},
    1905: {   'boxes': array([[  2,   1, 375, 112]]),
              'classes': array([1]),
              'scores': array([4.99723499e+08])},
    1906: {   'boxes': array([[123, 316, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99796499e+08])},
    1907: {   'boxes': array([[ 24, 263, 341, 384]]),
              'classes': array([1]),
              'scores': array([4.99747643e+08])},
    1908: {   'boxes': array([[145, 143, 374, 301]]),
              'classes': array([1]),
              'scores': array([4.99978588e+08])},
    1909: {   'boxes': array([[138,  35, 374, 162]]),
              'classes': array([1]),
              'scores': array([5.00018438e+08])},
    1910: {   'boxes': array([[ 49,  40, 245, 178]]),
              'classes': array([1]),
              'scores': array([4.99997702e+08])},
    1911: {   'boxes': array([[ 44, 153, 346, 340]]),
              'classes': array([1]),
              'scores': array([5.00139083e+08])},
    1912: {   'boxes': array([[  2, 401, 321, 500]]),
              'classes': array([1]),
              'scores': array([4.99739488e+08])},
    1913: {   'boxes': array([[ 45,  94, 305, 334]]),
              'classes': array([1]),
              'scores': array([4.99815226e+08])},
    1914: {   'boxes': array([[117,   6, 375, 376]]),
              'classes': array([1]),
              'scores': array([4.99976568e+08])},
    1915: {   'boxes': array([[ 98,   1, 375, 185]]),
              'classes': array([1]),
              'scores': array([4.99675363e+08])},
    1916: {   'boxes': array([[ 75, 294, 316, 432]]),
              'classes': array([1]),
              'scores': array([4.99801348e+08])},
    1917: {   'boxes': array([[ 64,  82, 327, 226]]),
              'classes': array([1]),
              'scores': array([4.99689796e+08])},
    1918: {   'boxes': array([[ 78, 412, 201, 461]]),
              'classes': array([1]),
              'scores': array([4.99718656e+08])},
    1919: {   'boxes': array([[ 36,  51, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99629374e+08])},
    1920: {   'boxes': array([[ 16, 153, 500, 291]]),
              'classes': array([1]),
              'scores': array([4.99704434e+08])},
    1921: {   'boxes': array([[ 22,  36, 500, 174]]),
              'classes': array([1]),
              'scores': array([4.99891627e+08])},
    1922: {   'boxes': array([[  5, 129, 453, 306]]),
              'classes': array([1]),
              'scores': array([4.99624836e+08])},
    1923: {   'boxes': array([[133, 113, 340, 227]]),
              'classes': array([1]),
              'scores': array([4.996927e+08])},
    1924: {   'boxes': array([[ 47, 114, 168, 149]]),
              'classes': array([1]),
              'scores': array([4.99866009e+08])},
    1925: {   'boxes': array([[ 48, 184, 276, 344]]),
              'classes': array([1]),
              'scores': array([5.00073749e+08])},
    1926: {   'boxes': array([[ 78,  61, 132,  82]]),
              'classes': array([1]),
              'scores': array([4.99725063e+08])},
    1927: {   'boxes': array([[ 81,  33, 134,  58]]),
              'classes': array([1]),
              'scores': array([4.99822038e+08])},
    1928: {   'boxes': array([[151,  20, 375, 266]]),
              'classes': array([1]),
              'scores': array([4.99646478e+08])},
    1929: {   'boxes': array([[140, 195, 264, 320]]),
              'classes': array([1]),
              'scores': array([4.9975059e+08])},
    1930: {   'boxes': array([[117, 355, 282, 487]]),
              'classes': array([1]),
              'scores': array([5.00241895e+08])},
    1931: {   'boxes': array([[100, 407, 219, 480]]),
              'classes': array([1]),
              'scores': array([4.99948357e+08])},
    1932: {   'boxes': array([[112, 126, 193, 157]]),
              'classes': array([1]),
              'scores': array([4.9987092e+08])},
    1933: {   'boxes': array([[112, 171, 192, 203]]),
              'classes': array([1]),
              'scores': array([5.00148424e+08])},
    1934: {   'boxes': array([[111, 245, 143, 271]]),
              'classes': array([1]),
              'scores': array([4.99872985e+08])},
    1935: {   'boxes': array([[116, 224, 144, 246]]),
              'classes': array([1]),
              'scores': array([4.99775505e+08])},
    1936: {   'boxes': array([[115, 249, 220, 340]]),
              'classes': array([1]),
              'scores': array([4.99886221e+08])},
    1937: {   'boxes': array([[107, 344, 260, 445]]),
              'classes': array([1]),
              'scores': array([4.99796843e+08])},
    1938: {   'boxes': array([[105, 320, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99889099e+08])},
    1939: {   'boxes': array([[130, 117, 241, 199]]),
              'classes': array([1]),
              'scores': array([4.99703826e+08])},
    1940: {   'boxes': array([[127,   2, 286, 149]]),
              'classes': array([1]),
              'scores': array([4.99686001e+08])},
    1941: {   'boxes': array([[  6,   1, 500, 352]]),
              'classes': array([1]),
              'scores': array([5.00526122e+08])},
    1942: {   'boxes': array([[183, 138, 411, 259]]),
              'classes': array([1]),
              'scores': array([4.99825926e+08])},
    1943: {   'boxes': array([[ 68, 198, 302, 324]]),
              'classes': array([1]),
              'scores': array([4.99941259e+08])},
    1944: {   'boxes': array([[ 72,  24, 172,  68]]),
              'classes': array([1]),
              'scores': array([4.99743015e+08])},
    1945: {   'boxes': array([[ 52,   6, 171,  89]]),
              'classes': array([1]),
              'scores': array([4.99714029e+08])},
    1946: {   'boxes': array([[ 97,  64, 169, 118]]),
              'classes': array([1]),
              'scores': array([4.99897164e+08])},
    1947: {   'boxes': array([[ 51,  79, 123, 125]]),
              'classes': array([1]),
              'scores': array([4.99686645e+08])},
    1948: {   'boxes': array([[ 79, 108, 159, 139]]),
              'classes': array([1]),
              'scores': array([4.99829958e+08])},
    1949: {   'boxes': array([[ 51, 128, 146, 178]]),
              'classes': array([1]),
              'scores': array([5.00132546e+08])},
    1950: {   'boxes': array([[138, 121, 167, 158]]),
              'classes': array([1]),
              'scores': array([4.99746535e+08])},
    1951: {   'boxes': array([[105, 171, 164, 216]]),
              'classes': array([1]),
              'scores': array([5.00033276e+08])},
    1952: {   'boxes': array([[ 60, 257, 107, 295]]),
              'classes': array([1]),
              'scores': array([4.99858781e+08])},
    1953: {   'boxes': array([[ 96, 320, 162, 388]]),
              'classes': array([1]),
              'scores': array([4.99706461e+08])},
    1954: {   'boxes': array([[ 49, 386, 160, 436]]),
              'classes': array([1]),
              'scores': array([4.99888544e+08])},
    1955: {   'boxes': array([[ 75, 438, 156, 486]]),
              'classes': array([1]),
              'scores': array([4.99851073e+08])},
    1956: {   'boxes': array([[ 40, 436, 155, 494]]),
              'classes': array([1]),
              'scores': array([4.9968945e+08])},
    1957: {   'boxes': array([[263, 160, 375, 318]]),
              'classes': array([1]),
              'scores': array([4.99914518e+08])},
    1958: {   'boxes': array([[245, 332, 375, 439]]),
              'classes': array([1]),
              'scores': array([4.99436817e+08])},
    1959: {   'boxes': array([[217, 415, 361, 479]]),
              'classes': array([1]),
              'scores': array([4.99820423e+08])},
    1960: {   'boxes': array([[142, 397, 347, 500]]),
              'classes': array([1]),
              'scores': array([4.9975573e+08])},
    1961: {   'boxes': array([[102, 256, 269, 383]])
,
              'classes': array([1]),
              'scores': array([5.00085462e+08])},
    1962: {   'boxes': array([[133, 181, 223, 257]]),
              'classes': array([1]),
              'scores': array([4.99888484e+08])},
    1963: {   'boxes': array([[135, 106, 186, 140]]),
              'classes': array([1]),
              'scores': array([4.99661997e+08])},
    1964: {   'boxes': array([[ 93,   1, 375, 171]]),
              'classes': array([1]),
              'scores': array([4.99702709e+08])},
    1965: {   'boxes': array([[271, 217, 307, 237]]),
              'classes': array([1]),
              'scores': array([4.99673441e+08])},
    1966: {   'boxes': array([[105, 219, 226, 271]]),
              'classes': array([1]),
              'scores': array([4.99711404e+08])},
    1967: {   'boxes': array([[119, 225, 165, 245]]),
              'classes': array([1]),
              'scores': array([4.99819472e+08])},
    1968: {   'boxes': array([[140,  58, 193,  90]]),
              'classes': array([1]),
              'scores': array([4.99670349e+08])},
    1969: {   'boxes': array([[ 72, 249, 224, 334]]),
              'classes': array([1]),
              'scores': array([4.99953085e+08])},
    1970: {   'boxes': array([[ 41, 235, 196, 296]]),
              'classes': array([1]),
              'scores': array([4.99780052e+08])},
    1971: {   'boxes': array([[164,  46, 251,  68]]),
              'classes': array([1]),
              'scores': array([4.99794638e+08])},
    1972: {   'boxes': array([[163,  13, 258,  44]]),
              'classes': array([1]),
              'scores': array([4.99831799e+08])},
    1973: {   'boxes': array([[174, 340, 252, 373]]),
              'classes': array([1]),
              'scores': array([4.99755693e+08])},
    1974: {   'boxes': array([[151, 206, 350, 261]]),
              'classes': array([1]),
              'scores': array([4.99784703e+08])},
    1975: {   'boxes': array([[130, 203, 372, 361]]),
              'classes': array([1]),
              'scores': array([4.99825054e+08])},
    1976: {   'boxes': array([[161, 323, 375, 500]]),
              'classes': array([1]),
              'scores': array([4.99728764e+08])},
    1977: {   'boxes': array([[104, 291, 281, 461]]),
              'classes': array([1]),
              'scores': array([4.99621685e+08])},
    1978: {   'boxes': array([[101, 236, 194, 331]]),
              'classes': array([1]),
              'scores': array([4.99656639e+08])},
    1979: {   'boxes': array([[110,  86, 281, 231]]),
              'classes': array([1]),
              'scores': array([4.99697505e+08])},
    1980: {   'boxes': array([[114, 171, 178, 204]]),
              'classes': array([1]),
              'scores': array([4.99789716e+08])},
    1981: {   'boxes': array([[106, 257, 160, 288]]),
              'classes': array([1]),
              'scores': array([4.99869827e+08])},
    1982: {   'boxes': array([[ 82, 224, 335, 314]]),
              'classes': array([1]),
              'scores': array([5.00054904e+08])},
    1983: {   'boxes': array([[ 55, 160, 294, 334]]),
              'classes': array([1]),
              'scores': array([4.99844657e+08])},
    1984: {   'boxes': array([[ 83, 340, 179, 385]]),
              'classes': array([1]),
              'scores': array([4.99885957e+08])},
    1985: {   'boxes': array([[162,   1, 232,  26]]),
              'classes': array([1]),
              'scores': array([4.9935845e+08])},
    1986: {   'boxes': array([[152,  24, 230,  80]]),
              'classes': array([1]),
              'scores': array([4.9951885e+08])},
    1987: {   'boxes': array([[156,  82, 188, 106]]),
              'classes': array([1]),
              'scores': array([4.99637507e+08])},
    1988: {   'boxes': array([[117, 220, 285, 280]]),
              'classes': array([1]),
              'scores': array([4.99738307e+08])},
    1989: {   'boxes': array([[114, 329, 275, 375]]),
              'classes': array([1]),
              'scores': array([4.99647637e+08])},
    1990: {   'boxes': array([[ 47,  96, 500, 318]]),
              'classes': array([1]),
              'scores': array([4.99913861e+08])},
    1991: {   'boxes': array([[185,  12, 500, 204]]),
              'classes': array([1]),
              'scores': array([4.99856208e+08])},
    1992: {   'boxes': array([[ 37,  96, 500, 293]]),
              'classes': array([1]),
              'scores': array([4.99717924e+08])},
    1993: {   'boxes': array([[130,  89, 450, 217]]),
              'classes': array([1]),
              'scores': array([4.99647191e+08])},
    1994: {   'boxes': array([[ 43,   1, 282, 302]]),
              'classes': array([1]),
              'scores': array([4.99878272e+08])},
    1995: {   'boxes': array([[ 85, 298, 278, 500]]),
              'classes': array([1]),
              'scores': array([4.99743406e+08])},
    1996: {   'boxes': array([[215, 201, 374, 416]]),
              'classes': array([1]),
              'scores': array([4.99634839e+08])},
    1997: {   'boxes': array([[184,  47, 283,  95]]),
              'classes': array([1]),
              'scores': array([4.99732629e+08])},
    1998: {   'boxes': array([[194, 221, 273, 253]]),
              'classes': array([1]),
              'scores': array([4.99653959e+08])},
    1999: {   'boxes': array([[261, 393, 301, 418]]),
              'classes': array([1]),
              'scores': array([4.99721668e+08])},
    2000: {   'boxes': array([[258, 380, 288, 394]]),
              'classes': array([1]),
              'scores': array([4.99752256e+08])},
    2001: {   'boxes': array([[247, 344, 291, 358]]),
              'classes': array([1]),
              'scores': array([4.99727263e+08])},
    2002: {   'boxes': array([[258, 476, 309, 500]]),
              'classes': array([1]),
              'scores': array([4.99700535e+08])},
    2003: {   'boxes': array([[ 88, 237, 326, 490]]),
              'classes': array([1]),
              'scores': array([4.99929916e+08])},
    2004: {   'boxes': array([[  1,   1, 257, 157]]),
              'classes': array([1]),
              'scores': array([4.99588097e+08])},
    2005: {   'boxes': array([[120, 184, 192, 235]]),
              'classes': array([1]),
              'scores': array([4.99703691e+08])},
    2006: {   'boxes': array([[  2,   2, 333, 267]]),
              'classes': array([1]),
              'scores': array([4.99784633e+08])},
    2007: {   'boxes': array([[ 79, 233, 333, 496]]),
              'classes': array([1]),
              'scores': array([4.99630389e+08])}}
Received these standard args: Namespace(accuracy_only=True, annotations_dir='/home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/caffe2-pose-estimation/annotations/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages', batch_size=1, benchmark_dir='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks', benchmark_only=False, checkpoint=None, data_location='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/pascal_voc_tfrecord/tfrecord-voc.record', data_num_inter_threads=None, data_num_intra_threads=None, disable_tcmalloc=True, framework='caffe', input_graph='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.prototxt', input_weights='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.caffemodel', intelai_models='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/detection_softmax', mode='inference', model_args=[], model_name='detection_softmax', model_source_dir='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet', mpi=None, num_cores=2, num_instances=1, num_inter_threads=1, num_intra_threads=1, num_mpi=1, output_dir='/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/tensorflow_object_detection_create_coco_tfrecord', output_results=False, precision='fp32', risk_difference=0.5, socket_id=0, tcmalloc_large_alloc_report_threshold=2147483648, use_case='object_detection', verbose=True)
Received these custom args: []
Current directory: /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks
Running: numactl --cpunodebind=0 --membind=0 /usr/bin/python3.6 /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/detection_softmax/inference/fp32/infer_detections.py -i 1000 -w 200 -a 1 -e 1 -g /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.prototxt -weight /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.caffemodel -rd 0.5 -r True -b 1 -d /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/pascal_voc_tfrecord/tfrecord-voc.record --annotations_dir /home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/caffe2-pose-estimation/annotations/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages
PYTHONPATH: :/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/detection_softmax
RUNCMD: /usr/bin/python3.6 common/caffe/run_tf_benchmark.py --framework=caffe --use-case=object_detection --model-name=detection_softmax --precision=fp32 --mode=inference --benchmark-dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks --intelai-models=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/models/benchmarks/../models/object_detection/caffe/detection_softmax --num-cores=2 --batch-size=1 --socket-id=0 --output-dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/tensorflow_object_detection_create_coco_tfrecord --annotations_dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge/Repository/caffe2-pose-estimation/annotations/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages --accuracy-only   --verbose --model-source-dir=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet --in-graph=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.prototxt --in-weights=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Repository/keras-arcface/models/squeezenet/squeezenet.caffemodel --data-location=/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/pascal_voc_tfrecord/tfrecord-voc.record --num-inter-threads=1 --num-intra-threads=1 --disable-tcmalloc=True                   
Batch Size: 1
Ran inference with batch size 1
Log location outside container: /home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/tensorflow_object_detection_create_coco_tfrecord/benchmark_detection_softmax_inference_fp32_20200523_094755.log