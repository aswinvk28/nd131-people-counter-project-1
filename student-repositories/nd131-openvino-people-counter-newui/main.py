import argparse
import cv2
from inference import Network

INPUT_STREAM = "test_video.mp4"
CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"
    ### TODO: Add additional arguments and descriptions for:
    ###       1) Different confidence thresholds used to draw bounding boxes
    ###       2) The user choosing the color of the bounding boxes
    c_desc = "The color of the bounding boxes to draw; RED, GREEN or BLUE"
    ct_desc = "The confidence threshold to use with the bounding boxes"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    optional.add_argument("-c", help=c_desc, default='BLUE')
    optional.add_argument("-ct", help=ct_desc, default=0.5)
    optional.add_argument("-th", help=ct_desc, default=2, type=int)
    args = parser.parse_args()

    return args


def convert_color(color_string):
    '''
    Get the BGR value of the desired bounding box color.
    Defaults to Blue if an invalid color is given.
    '''
    colors = {"BLUE": (255,0,0), "GREEN": (0,255,0), "RED": (0,0,255)}
    out_color = colors.get(color_string)
    if out_color:
        return out_color
    else:
        return colors['BLUE']


def draw_boxes(frame, boxes, args, width, height, y_pixel):
    '''
    Draw bounding boxes onto the frame.
    '''
    identified = False
    for box in result[0][0]: # Output shape is 1x1x100x7
        xmin = int(box[3] * width)
        ymin = int(box[4] * height)
        xmax = int(box[5] * width)
        ymax = int(box[6] * height)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), args.c, args.th)
        frame = cv2.putText(frame, out_text, (15, y_pixel), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return frame

def preprocess_outputs(frame, result, confidence_level=0.0):
        '''
        TODO: This method needs to be completed by you
        '''
        boxes = []
        confs = []
        height, width = frame.shape[2:]
        if len(result[0][0]) > 0:
            for res in result[0][0]:
                _, label, conf, xmin, ymin, xmax, ymax = res
                if conf > confidence_level:
                    xmin = int(xmin*width)
                    ymin = int(ymin*height)
                    xmax = int(xmax*width)
                    ymax = int(ymax*height)
                    boxes.append([xmin, ymin, xmax, ymax])
                    confs.append(conf)
        
        return boxes, confs


def infer_on_video(args):
    # Convert the args for color and confidence
    args.c = convert_color(args.c)
    args.ct = float(args.ct)

    ### TODO: Initialize the Inference Engine
    plugin = Network()

    ### TODO: Load the network model into the IE
    plugin.load_model(args.m, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)

    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    # The second argument should be `cv2.VideoWriter_fourcc('M','J','P','G')`
    # on Mac, and `0x00000021` on Linux
    out = cv2.VideoWriter('out.mp4', 0x00000021, 30, (width,height))

    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        ### TODO: Pre-process the frame
        p_frame = cv2.resize(frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        ### TODO: Perform inference on the frame
        plugin.async_inference(p_frame)

        ### TODO: Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()
            boxes, confs = preprocess_outputs(p_frame, result)
            ### TODO: Update the frame to include detected bounding boxes
            # frame = draw_boxes(frame, result, args, width, height)
            # Write out the frame
            draw_boxes(frame, )
            out.write(frame)

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    # out.release()
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
