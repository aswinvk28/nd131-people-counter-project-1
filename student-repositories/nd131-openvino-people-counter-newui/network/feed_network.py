import argparse
### TODO: Load the necessary libraries
import os
from inference import Network
from openvino.inference_engine.ie_api import IENetLayer
import numpy as np
import cv2

CPU_EXTENSION = "/home/aswin/Documents/Courses/Udacity/Intel-Edge-Phase2/Projects/People-Counter-App/Repository/nd131-openvino-people-counter-newui/custom_layers/arcface/cl_pnorm/user_ie_extensions/cpu/build/libpnorm_cpu_extension.so"
VPU_EXTENSION = "/opt/intel/openvino_2019.3.376/deployment_tools/inference_engine/lib/intel64/libmyriadPlugin.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Load an IR into the Inference Engine")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"

    # -- Create the arguments
    parser.add_argument("-m", help=m_desc)
    parser.add_argument('-l', required=False, type=str)
    parser.add_argument('-xp', required=False, type=str)
    parser.add_argument('-d', required=False, type=str)
    parser.add_argument('--img', required=False, type=str)
    parser.add_argument('--batch_size', required=False, type=int, default=1)
    parser.add_argument('--factor', required=False, type=float, default=1e-2)
    args = parser.parse_args()

    return args

def load_to_IE(args, model_xml, img_path):
    ### TODO: Load the Inference Engine API
    # plugin = IECore()

    network = Network()

    CPU_EXTENSION = args.l

    def exec_f(l):
        pass

    network.load_core(model_xml, args.d, cpu_extension=CPU_EXTENSION, args=args)

    if "MYRIAD" in args.d:
        network.feed_custom_layers(args, {'xml_path': args.xp}, exec_f)

    if "CPU" in args.d:
        network.feed_custom_parameters(args, exec_f)

    network.load_model(model_xml, args.d, cpu_extension=CPU_EXTENSION, args=args)

    # ### TODO: Load IR files into their related class
    # model_bin = os.path.splitext(model_xml)[0] + ".bin"
    # net = IENetwork(model=model_xml, weights=model_bin)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (160,160))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(1,1,160,160)

    img = img.astype(np.float64) - img.min()*args.factor
    img = img.astype(np.uint8)

    network.sync_inference(img)

    print(network.extract_output())

    network.check_layers(args)

    # feed_custom_layers(args, plugin, net, args.xp, exec_f)

    ### TODO: Add a CPU extension, if applicable. It's suggested to check
    ###       your code for unsupported layers for practice before 
    ###       implementing this. Not all of the models may need it.
    # if "CPU" in args.d:
    #     plugin.add_extension(args.l, "CPU")

    # if "MYRIAD" in args.d:
    #     plugin.add_extension(VPU_EXTENSION, "MYRIAD")

    ### TODO: Get the supported layers of the network
    # supported_layers = plugin.query_network(network=net, device_name=args.d)

    # input_blob = next(iter(net.inputs))
    # output_blob = next(iter(net.outputs))

    ### TODO: Load the network into the Inference Engine
    # exec_network = plugin.load_network(net, args.d)
    # exec_network.infer({input_blob: np.random.randint(0,255,(64,3,224,224))})

    # print(exec_network.requests[0].outputs)

    ### TODO: Check for any unsupported layers, and let the user
    ###       know if anything is missing. Exit the program, if so.
    # unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]
    # if len(unsupported_layers) != 0:
    #     print("Unsupported layers found: {}".format(unsupported_layers))
    #     print("Check whether extensions are available to add to IECore.")
    #     exit(1)

    return


def main():
    args = get_args()
    load_to_IE(args, args.m, args.img)


if __name__ == "__main__":
    main()
