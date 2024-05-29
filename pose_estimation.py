import cv2
import numpy as np
import pyrealsense2 as rs
import math
import torch


# Im new to github
#test git push lin -- ignore

fovh = 58
fovw = 87
resh = 720
resw = 1280

color_frame = None
depth_frame = None
color_image = None
depth_image = None

yaw_estimate = 0  # Initial estimate of yaw angle
yaw_estimate_error = 1  # Initial estimate error
process_variance = 0.01  # Process variance (tune this parameter)
measurement_variance = 1  # Measurement variance (tune this parameter)

def horizontal_distance(x, y, depth):
    global hfov, vfov, hres, vres
    v_theta = abs(640-y) * fovh/resh
    h_theta = abs(640-x) * fovw/resw
    h_theta = math.radians(h_theta)
    h_theta = round(h_theta, 2)
    #print(h_theta)

    hor_dist = math.cos(h_theta) * depth
    adj_dist = math.sin(h_theta) * depth

    #print(depth)

    #print(adj_dist)

    return hor_dist, adj_dist

def plot_boxes(results, frame):
    global color_frame, depth_frame, color_image, depth_image, yaw_estimate, yaw_estimate_error, process_variance, measurement_variance

    labels, cord = (
        results.xyxyn[0][:, -1].cpu().numpy(),
        results.xyxyn[0][:, :-1].cpu().numpy(),
    )

    for i in range(len(labels)):
        row = cord[i]
        if row[4] >= 0.2:
            x1, y1, x2, y2 = (
                int(row[0] * frame.shape[1]),
                int(row[1] * frame.shape[0]),
                int(row[2] * frame.shape[1]),
                int(row[3] * frame.shape[0]),
            )
            # Draw bounding box on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            #print(x1, " ", x2, " ", y1, " ", y2)
            centroid_coordinates = [int((x1+x2)/2), int((y1+y2)/2)]
            offset = (x2 - x1)/10
            #left_coordinates = [int(x1+offset), int((y1+y2)/2)]
            #right_coordinates = [int(x2-offset), int((y1+y2)/2)]

            left_coordinates = [int(x1+offset), int((y1+y2)/2)]
            right_coordinates = [int(x2-offset), int((y1+y2)/2)]

            cv2.circle(frame, (centroid_coordinates[0], centroid_coordinates[1]), 5, (0, 255, 0), -1)
            cv2.circle(frame, (left_coordinates[0], left_coordinates[1]), 5, (0, 255, 0), -1)
            cv2.circle(frame, (right_coordinates[0], right_coordinates[1]), 5, (0, 255, 0), -1)

            depth_target_left = depth_frame.get_distance(left_coordinates[0], left_coordinates[1])
            depth_target_center = depth_frame.get_distance(centroid_coordinates[0], centroid_coordinates[1])
            depth_target_right = depth_frame.get_distance(right_coordinates[0], right_coordinates[1])

            left_dist, left_adj = horizontal_distance(left_coordinates[0], left_coordinates[1], depth_target_left)

            center_dist, center_adj = horizontal_distance(centroid_coordinates[0], centroid_coordinates[1], depth_target_center)

            right_dist, right_adj = horizontal_distance(right_coordinates[0], right_coordinates[1], depth_target_right)

            '''print("Left depth",depth_target_left)
            print("Right depth",depth_target_right)
            print("Left adjacent",left_dist)
            print("Right adjacent",right_dist)'''




            x_center = int(frame.shape[1]/2)

            adj_dist = 1

            '''print("Left coordinates ",left_coordinates[0])
            print("Right coordinates ",right_coordinates[0])'''


            if left_coordinates[0] <= x_center and right_coordinates[0] >= x_center:
                adj_dist = abs(left_adj) + abs(right_adj)
                print("Box center")
            elif left_coordinates[0] <= x_center and right_coordinates[0] <= x_center:
                adj_dist = abs(left_adj) - abs(right_adj)
                print("Box left")    
            elif left_coordinates[0] >= x_center and right_coordinates[0] >= x_center:
                adj_dist = abs(right_adj) - abs(left_adj)
                print("Box right")              

            

            yaw_theta = 0

            if left_dist > right_dist:
                obj_depth_dist = left_dist - right_dist
                #print("Opposite distance", obj_depth_dist)
                yaw_theta = math.atan(obj_depth_dist/adj_dist)

            elif right_dist > left_dist:
                obj_depth_dist = right_dist -left_dist
                #print("Opposite distance", obj_depth_dist)
                yaw_theta = -math.atan(obj_depth_dist/adj_dist)

            yaw_theta_deg = math.degrees(yaw_theta)

            # Apply the Kalman filter
            innovation = yaw_theta_deg - yaw_estimate
            innovation_covariance = yaw_estimate_error + measurement_variance
            kalman_gain = yaw_estimate_error / innovation_covariance

            # Update the estimate
            yaw_estimate += kalman_gain * innovation

            # Update the estimate error
            yaw_estimate_error = (1 - kalman_gain) * yaw_estimate_error + process_variance

            # Print or use the filtered yaw_estimate value
            yaw_theta_deg = yaw_estimate

            print(i," yaw angle: ", yaw_theta_deg)
            



            # Put label on the bounding box
            text = f"Class {int(labels[i])}: {row[4]:.2f}"
            cv2.putText(
                frame,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

    return frame




def main():
    global color_frame, depth_frame, color_image, depth_image
    # Configure RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device('239222302533')
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 200)
    config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)

    # Start streaming
    pipeline.start(config)

    left_coordinates = [540, 360]
    center_coordinates = [640,360]
    right_coordinates = [740,360]

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='/home/muthukumar/ros2_ws/src/my_project/best.pt', force_reload=True)
    model.conf = 0.1  # confidence threshold (0-1)
    model.iou = 0.1  # NMS IoU threshold (0-1)


    try:
        while True:

            # Wait for the next set of frames
            frames = pipeline.wait_for_frames()

            # Get the color and depth frames
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            # Convert the frames to NumPy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            img = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)


            results = model(img)


            color_image = plot_boxes(results, color_image)

            '''# Get the depth values at the target coordinates
            depth_target_left = depth_frame.get_distance(left_coordinates[0], left_coordinates[1])
            depth_target_center = depth_frame.get_distance(center_coordinates[0], center_coordinates[1])
            depth_target_right = depth_frame.get_distance(right_coordinates[0], right_coordinates[1])


            # Print the depth values at the target coordinates
            #print(f"Left Depth (320, 360): {depth_target_left} meters")
            #print(f"Centre depth (640, 360): {depth_target_center} meters")
            #print(f"Right depth (960, 360): {depth_target_right} meters")

            # print(depth_target_left, " ", depth_target_center, " ", depth_target_right)

            left_dist, left_adj = horizontal_distance(left_coordinates[0], left_coordinates[1], depth_target_left)

            center_dist, center_adj = horizontal_distance(center_coordinates[0], center_coordinates[1], depth_target_center)

            right_dist, right_adj = horizontal_distance(right_coordinates[0], right_coordinates[1], depth_target_right)

            adj_dist = left_adj + right_adj

            #print("Adjacent distance", adj_dist)


            #print("Left distance: ", left_dist)
            #print("Right distance: ", right_dist)

            yaw_theta = 0

            if left_dist > right_dist:
                obj_depth_dist = left_dist - right_dist
                #print("Opposite distance", obj_depth_dist)
                yaw_theta = math.atan(obj_depth_dist/adj_dist)

            elif right_dist > left_dist:
                obj_depth_dist = right_dist -left_dist
                #print("Opposite distance", obj_depth_dist)
                yaw_theta = -math.atan(obj_depth_dist/adj_dist)
                
            

            yaw_theta_deg = math.degrees(yaw_theta)

            print("Yaw angle", yaw_theta_deg)'''

            # Draw circles at the target coordinates on the color image
            #cv2.circle(color_image, (left_coordinates[0], left_coordinates[1]), 5, (0, 255, 0), -1)
            #cv2.circle(color_image, (center_coordinates[0], center_coordinates[1]), 5, (0, 255, 0), -1)
            #cv2.circle(color_image, (right_coordinates[0], right_coordinates[1]), 5, (0, 255, 0), -1)
            

            # Apply depth colormap
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )
            depth_gray = cv2.cvtColor(depth_colormap, cv2.COLOR_BGR2GRAY)

            # Ensure that both images have the same number of channels
            depth_colormap = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2RGB)

            # Draw circles at the target coordinates on the depth image
            #cv2.circle(depth_colormap, (left_coordinates[0], left_coordinates[1]), 5, (0, 255, 0), -1)
            #cv2.circle(depth_colormap, (center_coordinates[0], center_coordinates[1]), 5, (0, 255, 0), -1)
            #cv2.circle(depth_colormap, (right_coordinates[0], right_coordinates[1]), 5, (0, 255, 0), -1)
            

            # Stack color and depth side by side
            combined_image = np.hstack((color_image, depth_colormap))

            # Resize the combined image to reduce width
            height_reduction = 0.5
            width_reduction = 0.5  # Set the width reduction factor
            resized_image = cv2.resize(combined_image, (0, 0), fx=width_reduction, fy=height_reduction)

            # Display the combined frame using OpenCV
            cv2.imshow('RealSense D455 Feed', resized_image)

            # Check for the 'q' key to exit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    finally:
        # Stop streaming
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
