# Function to process images and YOLO annotations with confidence threshold
def process_images_and_annotations(image_folder, annotation_folder):
    inname = [i.name for i in session.get_inputs()]
    outname = [i.name for i in session.get_outputs()]

    # Get a list of image and annotation file paths
    image_paths = glob.glob(os.path.join(image_folder, '*.jpg'))
    annotation_paths = [os.path.join(annotation_folder, os.path.splitext(os.path.basename(image_path))[0] + '.txt') for image_path in image_paths]

    n=0
    for image_path, annotation_path in zip(image_paths, annotation_paths):
        # Load the image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        n=n+1
        print(n)

        # Load YOLO annotation from the file (assuming YOLO format) and remove empty lines

        if not os.path.isfile(annotation_path):
            # If it doesn't exist, create an empty file
            open(annotation_path, 'a').close()


        with open(annotation_path, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        # Extract the bounding box information for existing annotations
        existing_annotations = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                cls_id, x0, y0, width, height = map(float, parts[:5])
                x1, y1 = x0 + width, y0 + height
                existing_annotations.append((cls_id, x0, y0, x1, y1))

        # Perform YOLO inference
        image, ratio, dwdh = letterbox(img, auto=False)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, 0)
        image = np.ascontiguousarray(image)
        im = image.astype(np.float32)
        im /= 255
        dw, dh = dwdh
        inp = {inname[0]: im}

        outputs = session.run(outname, inp)[0]

        image_width = img.shape[1]
        image_height = img.shape[0]


        for output in outputs:
            # Check if the detected object is a person, pet, or vehicle
            if len(output) >= 6:
                x0, y0, x1, y1, cls_id, score = output[:6]
                cls_id = int(cls_id)
                #print(x0,y0,x1,y1)
                if cls_id in [0, 1, 2] and score >= confidence_threshold:
                    # Convert coordinates back to the original image space
                    # x0, y0, x1, y1 = x0 + int(dwdh[0] * 2 / ratio), y0 + int(dwdh[1] * 2 / ratio), x1 + int(dwdh[0] * 2 / ratio), y1 + int(dwdh[1] * 2 / ratio)
                    # x0, y0, x1, y1 = int(x0 * ratio), int(y0 * ratio), int(x1 * ratio), int(y1 * ratio)

                    # # Normalize the coordinates to YOLO format using the image size

                    # x0 /= image_width
                    # x1 /= image_width
                    # y0 /= image_height
                    # y1 /= image_height

                    # Check if the detection overlaps with existing annotations
                    overlaps_existing = False
                    # for ex_cls_id, ex_x0, ex_y0, ex_x1, ex_y1 in existing_annotations:
                    #     intersection_x0 = max(x0, ex_x0)
                    #     intersection_y0 = max(y0, ex_y0)
                    #     intersection_x1 = min(x1, ex_x1)
                    #     intersection_y1 = min(y1, ex_y1)
                    #     if intersection_x0 < intersection_x1 and intersection_y0 < intersection_y1:
                    #         overlaps_existing = True
                    #         break

                    # If it doesn't overlap with existing annotations, add it to the YOLO annotation
                    # x0 -= dw / 2  # unpad
                    # y0 -= dh / 2  # unpad
                    # x1 -= dw / 2  # unpad
                    # y1 -= dh / 2  # unpad

                    # x0=x0*(image_width/640)
                    # x1=x1*(image_width/640)
                    # y0=y0*(image_height/640)
                    # y1=y1*(image_height/640)  

                    # x0 -= dwdh*2
                    # x1 -= dwdh*2
                    # y0 -= dwdh*2
                    # y1 -= dwdh*2


                    # x0 /= ratio  
                    # x1 /= ratio 
                    # y0 /= ratio 
                    # y1 /= ratio             

                    # x0 = x0 / image_width
                    # y0 = y0 / image_height
                    # x1 = x1 / image_width
                    # y1 = y1 / image_height

                    box = np.array([x0,y0,x1,y1])
                    box -= np.array(dwdh*2)
                    box /= ratio
                    box = box.round().astype(np.int32).tolist()
                    #print(box)


                    x0= box[0]
                    y0= box[1]
                    x1= box[2]
                    y1= box[3]

                    x0 = x0 / image_width
                    y0 = y0 / image_height
                    x1 = x1 / image_width
                    y1 = y1 / image_height

                    x_centre = (x0 + x1) / 2
                    y_centre = (y0 + y1) / 2
                    x_width = abs(x1 - x0)
                    y_height = abs(y1 - y0)
                    if x0>0 and y0>0 and x1>0 and y1>0: 
                        new_annotation = f'{cls_id} {x_centre} {y_centre} {x_width} {y_height}\n'
                        lines.append(new_annotation)

        # Save the updated YOLO annotation

        # Check if the file exists
        if not os.path.isfile(annotation_path):
            # If it doesn't exist, create an empty file
            open(annotation_path, 'a').close()

        with open(annotation_path, 'w') as f:
            for line in lines:
                f.write(line + '\n')

# Specify the folders containing images and YOLO annotations
#image_folder = "/Users/pranav.naik/Downloads/cat_combined_auto_anno"
#annotation_folder = "/Users/pranav.naik/Downloads/cat_combined_auto_anno"

image_folder = "C:/Users/L4ZER/Downloads/work/Batch-4_2/Batch-4_2/baby_smile_0703_38"   #path to face images folder
annotation_folder = "C:/Users/L4ZER/Downloads/work/Batch-4_2/Batch-4_2/38-output" #path to face images folder

# Process images and annotations
process_images_and_annotations(image_folder, annotation_folder)
