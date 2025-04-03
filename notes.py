



# @markdown Plot instance masks.
def plot_mask(color, alpha, original_image, mask):
  rgb = ImageColor.getrgb(color)
  pil_image = Image.fromarray(original_image)

  solid_color = np.expand_dims(
    np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
  pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert("RGBA")
  pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert("L")
  pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
  img_w_mask = np.array(pil_image.convert("RGB"))
  return img_w_mask



def display_image(path_or_array, size=(10, 10)):
  if isinstance(path_or_array, str):
    image = np.asarray(Image.open(open(image_path, "rb")).convert("RGB"))
  else:
    image = path_or_array

  plt.figure(figsize=size)
  plt.imshow(image)
  plt.axis("off")
  plt.show()

  # @markdown Define ViLD forward pass.

def vild(image_path, category_name_string, params, plot_on=True, prompt_swaps=[]):
  #################################################################
  # Preprocessing categories and get params
  for a, b in prompt_swaps:
    category_name_string = category_name_string.replace(a, b)
  category_names = [x.strip() for x in category_name_string.split(";")]
  category_names = ["background"] + category_names
  categories = [{"name": item, "id": idx + 1, } for idx, item in enumerate(category_names)]
  category_indices = {cat["id"]: cat for cat in categories}

  max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area, max_box_area = params
  fig_size_h = min(max(5, int(len(category_names) / 2.5)), 10)

  #################################################################
  # Obtain results and read image
  roi_boxes, roi_scores, detection_boxes, scores_unused, box_outputs, detection_masks, visual_features, image_info = session.run(
    ["RoiBoxes:0", "RoiScores:0", "2ndStageBoxes:0", "2ndStageScoresUnused:0", "BoxOutputs:0", "MaskOutputs:0",
     "VisualFeatOutputs:0", "ImageInfo:0"],
    feed_dict={"Placeholder:0": [image_path, ]})

  roi_boxes = np.squeeze(roi_boxes, axis=0)  # squeeze
  # no need to clip the boxes, already done
  roi_scores = np.squeeze(roi_scores, axis=0)

  detection_boxes = np.squeeze(detection_boxes, axis=(0, 2))
  scores_unused = np.squeeze(scores_unused, axis=0)
  box_outputs = np.squeeze(box_outputs, axis=0)
  detection_masks = np.squeeze(detection_masks, axis=0)
  visual_features = np.squeeze(visual_features, axis=0)

  image_info = np.squeeze(image_info, axis=0)  # obtain image info
  image_scale = np.tile(image_info[2:3, :], (1, 2))
  image_height = int(image_info[0, 0])
  image_width = int(image_info[0, 1])

  rescaled_detection_boxes = detection_boxes / image_scale  # rescale

  # Read image
  image = np.asarray(Image.open(open(image_path, "rb")).convert("RGB"))
  assert image_height == image.shape[0]
  assert image_width == image.shape[1]

  #################################################################
  # Filter boxes

  # Apply non-maximum suppression to detected boxes with nms threshold.
  nmsed_indices = nms(
    detection_boxes,
    roi_scores,
    thresh=nms_threshold
  )

  # Compute RPN box size.
  box_sizes = (rescaled_detection_boxes[:, 2] - rescaled_detection_boxes[:, 0]) * (
            rescaled_detection_boxes[:, 3] - rescaled_detection_boxes[:, 1])

  # Filter out invalid rois (nmsed rois)
  valid_indices = np.where(
    np.logical_and(
      np.isin(np.arange(len(roi_scores), dtype=np.int), nmsed_indices),
      np.logical_and(
        np.logical_not(np.all(roi_boxes == 0., axis=-1)),
        np.logical_and(
          roi_scores >= min_rpn_score_thresh,
          np.logical_and(
            box_sizes > min_box_area,
            box_sizes < max_box_area
          )
        )
      )
    )
  )[0]

  detection_roi_scores = roi_scores[valid_indices][:max_boxes_to_draw, ...]
  detection_boxes = detection_boxes[valid_indices][:max_boxes_to_draw, ...]
  detection_masks = detection_masks[valid_indices][:max_boxes_to_draw, ...]
  detection_visual_feat = visual_features[valid_indices][:max_boxes_to_draw, ...]
  rescaled_detection_boxes = rescaled_detection_boxes[valid_indices][:max_boxes_to_draw, ...]

  #################################################################
  # Compute text embeddings and detection scores, and rank results
  text_features = build_text_embedding(categories)

  raw_scores = detection_visual_feat.dot(text_features.T)
  if FLAGS.use_softmax:
    scores_all = softmax(FLAGS.temperature * raw_scores, axis=-1)
  else:
    scores_all = raw_scores

  indices = np.argsort(-np.max(scores_all, axis=1))  # Results are ranked by scores
  indices_fg = np.array([i for i in indices if np.argmax(scores_all[i]) != 0])

  #################################################################
  # Print found_objects
  found_objects = []
  for a, b in prompt_swaps:
    category_names = [name.replace(b, a) for name in category_names]  # Extra prompt engineering.
  for anno_idx in indices[0:int(rescaled_detection_boxes.shape[0])]:
    scores = scores_all[anno_idx]
    if np.argmax(scores) == 0:
      continue
    found_object = category_names[np.argmax(scores)]
    if found_object == "background":
      continue
    print("Found a", found_object, "with score:", np.max(scores))
    found_objects.append(category_names[np.argmax(scores)])
  if not plot_on:
    return found_objects

  #################################################################
  # Plot detected boxes on the input image.
  ymin, xmin, ymax, xmax = np.split(rescaled_detection_boxes, 4, axis=-1)
  processed_boxes = np.concatenate([xmin, ymin, xmax - xmin, ymax - ymin], axis=-1)
  segmentations = paste_instance_masks(detection_masks, processed_boxes, image_height, image_width)

  if len(indices_fg) == 0:
    display_image(np.array(image), size=overall_fig_size)
    print("ViLD does not detect anything belong to the given category")

  else:
    image_with_detections = visualize_boxes_and_labels_on_image_array(
      np.array(image),
      rescaled_detection_boxes[indices_fg],
      valid_indices[:max_boxes_to_draw][indices_fg],
      detection_roi_scores[indices_fg],
      numbered_category_indices,
      instance_masks=segmentations[indices_fg],
      use_normalized_coordinates=False,
      max_boxes_to_draw=max_boxes_to_draw,
      min_score_thresh=min_rpn_score_thresh,
      skip_scores=False,
      skip_labels=True)

    # plt.figure(figsize=overall_fig_size)
    plt.imshow(image_with_detections)
    # plt.axis("off")
    plt.title("ViLD detected objects and RPN scores.")
    plt.show()

  return found_objects



category_names = ['blue block',
                  'red block',
                  'green block',
                  'orange block',
                  'yellow block',
                  'purple block',
                  'pink block',
                  'cyan block',
                  'brown block',
                  'gray block',

                  'blue bowl',
                  'red bowl',
                  'green bowl',
                  'orange bowl',
                  'yellow bowl',
                  'purple bowl',
                  'pink bowl',
                  'cyan bowl',
                  'brown bowl',
                  'gray bowl']
image_path = 'tmp.jpg'

#@markdown ViLD settings.
category_name_string = ";".join(category_names)
max_boxes_to_draw = 8 #@param {type:"integer"}

# Extra prompt engineering: swap A with B for every (A, B) in list.
prompt_swaps = [('block', 'cube')]

nms_threshold = 0.4 #@param {type:"slider", min:0, max:0.9, step:0.05}
min_rpn_score_thresh = 0.4  #@param {type:"slider", min:0, max:1, step:0.01}
min_box_area = 10 #@param {type:"slider", min:0, max:10000, step:1.0}
max_box_area = 3000  #@param {type:"slider", min:0, max:10000, step:1.0}
vild_params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area, max_box_area
found_objects = vild(image_path, category_name_string, vild_params, plot_on=True, prompt_swaps=prompt_swaps)
