import os
import cv2

train_or_val = ["training", "validation"] #training or validation OR test
for dir in train_or_val:
  image_directory = f"./ADEChallengeData2016/images/{dir}/"
  annotation_directory = f"./ADEChallengeData2016/annotations/{dir}/"

  image_list = os.listdir(image_directory)
  annotation_list = os.listdir(annotation_directory)

  with open(f"{dir}.odgt", "w") as file:
      for image in image_list:
          img = cv2.imread(image_directory + image)
          if image.startswith("image"):
            searchable = image.split(".")[0]
          else:
            searchable= image.split("_")[2].split(".")[0]
          eq_annotations_list = [i for i in annotation_list if i.__contains__(searchable)]
          if len(eq_annotations_list) == 0:
              print(f"{image} doesn't have annotation!")
              continue
          pngimg = eq_annotations_list[0]
          file.write("{"+f"\"fpath_img\": \"{image_directory+image}\", \"fpath_segm\": \"{annotation_directory+pngimg}\", \"width\": {img.shape[1]}, \"height\": {img.shape[0]}" + "}\n")
  print("Done.")
